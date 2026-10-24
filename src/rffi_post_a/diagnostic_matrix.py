"""Run the fixed D1 matrix on one target task, and the metric-layer diagnostics that may use query
labels (plan §6.6 items 5-7). Prediction never sees labels; labels are looked up only in
`_metrics` and in functions whose names start with `metric_layer_`, whose outputs are written to the
report and never fed back into adaptation or slot choice.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from rffi_calibration.evaluator import score_metrics
from rffi_calibration.heads import (GaussianShrink, LegacyP2EProto, PosteriorPredictive, SimpleCNAPSFrozen,
                                    SimpleShotCL2N, TargetPrototype)
from rffi_calibration.sampling import support_ids
from rffi_calibration.statistics import DTYPE, StatsConfig, _as64, support_means

from .diagnostic_heads import adapt_variant, diagnostic_variants, score_terms
from .diagnostic_stats import NotApplicable

REPLAY_HEADS = {
    'H1': lambda: TargetPrototype('pooled'), 'H4': lambda: PosteriorPredictive(StatsConfig()),
    'H2L': lambda: LegacyP2EProto(), 'H4-pooled': lambda: PosteriorPredictive(StatsConfig(pooled=True)),
    'H4-AK': lambda: PosteriorPredictive(StatsConfig(finite_center=False)), 'H3': lambda: GaussianShrink(),
    'H5a': lambda: SimpleShotCL2N(), 'H5b': lambda: SimpleCNAPSFrozen(),
}


def fit_replays(names, z_src, y_src, rx_codes, T):
    out = {}
    for n in names:
        out[n] = REPLAY_HEADS[n]().fit(torch.as_tensor(z_src, dtype=DTYPE), y_src, rx_codes, T)
    return out


def _metrics(prob, yq, T):
    met = score_metrics(prob, yq, T)
    rec = met.pop('recall')
    return dict(met, **{f'recall_{c}': v for c, v in rec.items()},
                **{f'pred_frac_{c}': float((prob.argmax(1) == c).mean()) for c in range(T)})


def run_task(bundle, matrix, replays, z_target, y_target, ids_target, manifest, T, keep_scores=False):
    """All registered slots + replays x budgets x draws. Labels of the query are looked up after scoring.

    z_target/y_target/ids_target: target feature rows, their labels and sample IDs (IDs used only to
    locate rows; the IDs and labels never reach a scorer)."""
    if not manifest['full_coverage']:
        raise ValueError(f'RX {manifest["rx"]}: manifest without full class coverage (no silent partial table)')
    lut = {s: i for i, s in enumerate(ids_target)}
    rows_of = lambda ids: np.array([lut[i] for i in ids], dtype=np.int64)
    q = rows_of(manifest['query_ids'])
    zq = torch.as_tensor(z_target[q], dtype=DTYPE)
    yq = np.asarray(y_target)[q]
    variants = diagnostic_variants(bundle, matrix)
    out, na, scores = [], [], {}
    for m in matrix['budgets']:
        for k in (range(1) if m == 0 else range(manifest['n_draws'])):
            sr = rows_of(support_ids(manifest, k, m)) if m else np.zeros(0, np.int64)
            zs = torch.as_tensor(z_target[sr], dtype=DTYPE)
            ys = np.asarray(y_target)[sr]
            entries = [(n, v, 'slot') for n, v in variants.items()] + [(n, h, 'replay') for n, h in replays.items()]
            for name, obj, kind in entries:
                t0 = time.perf_counter()
                try:
                    if kind == 'slot':
                        terms = score_terms(adapt_variant(obj, zs, ys), zq)
                    else:
                        if m == 0 and obj.needs_support:
                            raise NotApplicable('needs support')
                        cal = obj.adapt(zs, ys)
                        terms = dict(score=cal.log_scores(zq), maha=None, logdet=None)
                except NotApplicable as e:
                    if k == 0:
                        na.append(dict(slot=name, m=m, reason=str(e)))
                    continue
                prob = torch.softmax(terms['score'], -1).numpy()
                row = dict(rx=manifest['rx'], slot=name, kind=kind, m=m, draw=k, n_query=len(q),
                           seconds=time.perf_counter() - t0, **_metrics(prob, yq, T))
                if terms.get('logdet') is not None:
                    ld = terms['logdet']
                    row.update(logdet_range=float(ld.max() - ld.min()),
                               maha_median=float(terms['maha'].median()))
                out.append(row)
                if keep_scores:
                    scores[(name, m, k)] = terms['score'].numpy()
    return out, na, scores


def anchor_check(scores, pairs, tol=1e-8):
    """Registered equivalences (e.g. D_CC==H4, M_SP==H1, C_PF_L==H2L): max |score diff| after removing
    each row's max (softmax-invariant), per (m, draw)."""
    res = {}
    for a, b in pairs:
        worst = 0.
        keys = [(m, k) for (n, m, k) in scores if n == a and (b, m, k) in scores]
        for m, k in keys:
            x, y = scores[(a, m, k)], scores[(b, m, k)]
            x = x - x.max(1, keepdims=True)
            y = y - y.max(1, keepdims=True)
            worst = max(worst, float(np.abs(x - y).max()))
        res[f'{a}=={b}'] = dict(max_abs=worst, n=len(keys), ok=bool(keys) and worst <= tol)
    return res


# ------------------------------------------------------------------- metric-layer diagnostics
def metric_layer_center_errors(bundle, z_target, y_target, ids_target, manifest, budgets=(1, 5)):
    """Plan §6.6 items 5-6. Per draw: distance (in source pooled-variance units) from mu, the support
    mean s and the H4 posterior centre theta to each of two FIXED halves of the query set (query ID
    order). Query labels are used here as a diagnostic statistic only."""
    lut = {s: i for i, s in enumerate(ids_target)}
    q = np.array([lut[i] for i in manifest['query_ids']])
    y = np.asarray(y_target)
    sp = bundle.sigma_pool
    halves = []
    for t in range(bundle.T):
        qt = q[y[q] == t]
        h = len(qt) // 2
        halves.append((qt[:h], qt[h:]))
    zt = torch.as_tensor(z_target, dtype=DTYPE)
    hm = [torch.stack([zt[torch.as_tensor(h[i])].mean(0) for h in halves]) for i in (0, 1)]
    dist = lambda a, b: (((a - b) ** 2) / sp).mean(1)               # (T,)
    from .diagnostic_heads import _posterior, _Kc, _Sc
    rows = []
    noise = float(dist(hm[0], hm[1]).mean())
    for m in budgets:
        for k in range(manifest['n_draws']):
            sr = np.array([lut[i] for i in support_ids(manifest, k, m)])
            s, mm, _ = support_means(zt[torch.as_tensor(sr)], y[sr], bundle.T)
            theta, _ = _posterior(bundle, _Kc(bundle), _Sc(bundle), s, mm)
            for name, c in (('mu', bundle.mu), ('s', s), ('theta', theta)):
                e = [float(dist(c, hm[i]).mean()) for i in (0, 1)]
                rows.append(dict(rx=manifest['rx'], m=m, draw=k, center=name, err_half0=e[0], err_half1=e[1],
                                 query_half_disagreement=noise))
    return rows


def metric_layer_flip_analysis(scores, y_query, a='D_CC', b='M_SP', budgets=(1, 5)):
    """Plan §6.6 item 7: how often H4 (a) and H1 (b) disagree, and in which direction."""
    out = []
    for m in budgets:
        keys = sorted(k for (n, mm, k) in scores if n == a and mm == m and (b, m, k) in scores)
        for k in keys:
            pa, pb = scores[(a, m, k)].argmax(1), scores[(b, m, k)].argmax(1)
            ca, cb = pa == y_query, pb == y_query
            out.append(dict(m=m, draw=k, disagree=float((pa != pb).mean()), a_right_b_wrong=float((ca & ~cb).mean()),
                            a_wrong_b_right=float((~ca & cb).mean())))
    return out
