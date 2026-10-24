"""Frozen-head evaluation engine, source-only parameter selection and paired reporting.

Data flow for one target RX (EXPERIMENT_PROTOCOL.md §5):
    support IDs (manifest, draw, m) -> z_support, y_support -> head.adapt
    query IDs (manifest, fixed)     -> z_query                -> cal.predict_proba
    y_query is looked up only afterwards, inside score_metrics.
Selection (alpha of H2, kappa0 of H5c, temperatures) uses source_selection RX only; the reported gate
on those RX uses leave-one-selection-RX-out cross-fitting so no RX scores the parameter it chose.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from .heads import FAMILIES
from .sampling import support_ids
from .statistics import MissingClassError

BUDGETS = (0, 1, 2, 5, 10)


@dataclass
class FeatureSet:
    """Frozen features aligned with a RecordTable (row i of z <-> table.ids[i])."""
    z: np.ndarray
    table: object
    meta: dict

    def rows(self, ids):
        return self.table.index_of(ids)


# ------------------------------------------------------------------------------------------ metrics
def score_metrics(prob, y, num_classes, classes=None, n_bins=15):
    """Accuracy, macro-F1 / per-class recall over the evaluated classes, NLL, Brier, ECE."""
    prob = np.asarray(prob, dtype=np.float64)
    y = np.asarray(y)
    classes = list(range(num_classes)) if classes is None else list(classes)
    pred = prob.argmax(1)
    acc = float((pred == y).mean())
    rec, f1 = {}, []
    for t in classes:
        tp = float(((pred == t) & (y == t)).sum())
        p_t, y_t = float((pred == t).sum()), float((y == t).sum())
        r = tp / y_t if y_t else math.nan
        pr = tp / p_t if p_t else 0.
        rec[t] = r
        f1.append(0. if tp == 0 else 2 * pr * r / (pr + r))
    p_true = np.clip(prob[np.arange(len(y)), y], 1e-300, None)
    onehot = np.eye(num_classes)[y]
    conf = prob.max(1)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return dict(acc=acc, macro_f1=float(np.mean(f1)), nll=float(-np.log(p_true).mean()),
                brier=float(((prob - onehot) ** 2).sum(1).mean()), ece=float(ece),
                recall={int(k): v for k, v in rec.items()})


# ------------------------------------------------------------------------------------- evaluation
def fit_heads(heads, source: FeatureSet, num_classes, log=None):
    """Fit every head on source fitting records. Heads that are not applicable are dropped and logged."""
    z = torch.as_tensor(source.z, dtype=torch.float64)
    fitted, skipped = [], []
    for h in heads:
        try:
            fitted.append(h.fit(z, source.table.tx, _codes(source.table.rx), num_classes))
        except (ValueError, MissingClassError) as e:
            skipped.append(dict(head=h.name, reason=str(e)))
            if log:
                log(f'head {h.name} not applicable: {e}')
    return fitted, skipped


def _codes(a):
    return np.unique(np.asarray(a), return_inverse=True)[1]


def evaluate_target(heads, target: FeatureSet, manifest, num_classes, budgets=BUDGETS, keep_logits=False):
    """All heads x budgets x draws on one target RX. Returns (rows, logits) where logits maps
    (head, m, draw) -> (n_query, T) array (only if keep_logits)."""
    if not manifest['full_coverage']:
        raise ValueError(f'rx {manifest["rx"]} lacks full class coverage; use the partial-coverage table')
    q_rows = target.rows(manifest['query_ids'])
    zq = torch.as_tensor(target.z[q_rows], dtype=torch.float64)
    rows, logits = [], {}
    for m in budgets:
        draws = range(1) if m == 0 else range(manifest['n_draws'])
        for k in draws:
            s_rows = target.rows(support_ids(manifest, k, m)) if m else np.zeros(0, np.int64)
            zs = torch.as_tensor(target.z[s_rows], dtype=torch.float64)
            ys = target.table.tx[s_rows]
            for h in heads:
                if m == 0 and h.needs_support:
                    continue
                t0 = time.perf_counter()
                cal = h.adapt(zs, ys)
                t1 = time.perf_counter()
                lg = cal.log_scores(zq).detach()
                t2 = time.perf_counter()
                if not bool(torch.isfinite(lg).all()):
                    raise FloatingPointError(f'{h.name}: non-finite scores on rx {manifest["rx"]} m={m}')
                prob = torch.softmax(lg, -1).numpy()
                met = score_metrics(prob, target.table.tx[q_rows], num_classes)   # labels only here
                rec = met.pop('recall')
                rows.append(dict(rx=manifest['rx'], head=h.name, m=m, draw=k, n_query=len(q_rows),
                                 support_labels=int(m * num_classes), adapt_s=t1 - t0, predict_s=t2 - t1,
                                 **met, **{f'recall_{c}': v for c, v in rec.items()}))
                if keep_logits:
                    logits[(h.name, m, k)] = lg.numpy()
    return rows, logits


# -------------------------------------------------------------------------------------- selection
def family_of(name):
    for fam, pred in FAMILIES.items():
        if pred(name):
            return fam
    return None


def _rx_mean(rows, key='acc'):
    acc = defaultdict(list)
    for r in rows:
        acc[(r['head'], r['m'], r['rx'])].append(r[key])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _grid_value(name):
    return float(name.split('=')[1].rstrip(']'))


def select_family_members(rows, exclude_rx=None):
    """For every family and budget, the grid member with the highest receiver-equal mean accuracy
    on the given (selection) rows. Ties -> the smaller grid value (deterministic)."""
    per = _rx_mean([r for r in rows if r['rx'] != exclude_rx])
    choice = {}
    by = defaultdict(lambda: defaultdict(list))
    for (h, m, rx), a in per.items():
        fam = family_of(h)
        if fam:
            by[(fam, m)][h].append(a)
    for (fam, m), members in by.items():
        choice[(fam, m)] = max(members, key=lambda h: (np.mean(members[h]), -_grid_value(h)))
    return choice


def crossfit_family_rows(rows):
    """Leave-one-selection-RX-out: for each RX q, report the family member chosen on the other RX."""
    out = []
    for q in sorted(set(r['rx'] for r in rows)):
        ch = select_family_members(rows, exclude_rx=q)
        for r in rows:
            if r['rx'] != q:
                continue
            fam = family_of(r['head'])
            if fam and ch.get((fam, r['m'])) == r['head']:
                out.append(dict(r, head=fam + '*', selected=r['head']))
    return out


def fit_temperature(logit_list, y_list, lo=-5., hi=7., iters=60):
    """1-D NLL temperature scaling (golden section on log T). Accuracy is unaffected."""
    L = np.concatenate(logit_list)
    y = np.concatenate(y_list)

    def nll(logt):
        z = L / math.exp(logt)
        z = z - z.max(1, keepdims=True)
        return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean())

    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(iters):
        if nll(c) < nll(d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return math.exp((a + b) / 2)


# -------------------------------------------------------------------------------------- reporting
def aggregate(rows, key='acc'):
    """Per (head, m): receiver-equal mean, worst RX, number of RX (draw-means per RX first)."""
    per = _rx_mean(rows, key)
    by = defaultdict(list)
    for (h, m, rx), v in per.items():
        by[(h, m)].append(v)
    return {k: dict(mean=float(np.mean(v)), worst=float(np.min(v)), n_rx=len(v)) for k, v in by.items()}


def paired(rows, head_a, head_b, m, key='acc'):
    """Per-RX paired difference a - b (draw-means first; draws are not independent RX).
    t-interval over RX and the count of RX improved. Returns None if either head is missing."""
    per = _rx_mean(rows, key)
    rxs = sorted(set(rx for (h, mm, rx) in per if mm == m))
    d = [per[(head_a, m, rx)] - per[(head_b, m, rx)] for rx in rxs
         if (head_a, m, rx) in per and (head_b, m, rx) in per]
    if not d:
        return None
    d = np.array(d)
    n = len(d)
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    tq = {1: math.nan, 2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26,
          11: 2.23, 12: 2.20, 13: 2.18, 14: 2.16}.get(n, 2.0)
    return dict(a=head_a, b=head_b, m=m, n_rx=n, mean=float(d.mean()), lo=float(d.mean() - tq * se),
                hi=float(d.mean() + tq * se), n_improved=int((d > 0).sum()), worst=float(d.min()))
