"""Receiver-episodic training objectives for T2 (ProtoNet) and T3 (posterior-predictive risk).

Stage B only (not authorised in this round): this module defines the episode schedule and the two
query losses so that their mathematics, gradients and leakage rules can be tested on synthetic data.

One step (EXPERIMENT_PROTOCOL.md §6), identical record schedule for every arm T0-T3:
  1. pseudo-target RX: uniform rotation over the fitting RX (seeded permutation cycles)
  2. reference RX: all other fitting RX (>= 2 required), n_ref records per class each
  3. pseudo-target: m support + n_query query per class, disjoint; m uniform from BUDGETS
  4. every arm computes native CE on ALL records of the step (same anchor);
     T2 adds CE of the ProtoNet query scores (support means, squared Euclidean / temperature);
     T3 adds CE of the posterior-predictive query scores, whose source statistics are fitted on the
     step's REFERENCE features only (never on pseudo-target support/query).
Reference features, support means and the posterior keep autograd (no detached epoch caches).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .statistics import StatsConfig, diag_gaussian_logpdf, fit_source_stats, posterior_predictive, support_means

BUDGETS = (1, 2, 5, 10)


@dataclass
class Episode:
    step: int
    pseudo_target: int
    m: int
    ref_idx: np.ndarray
    sup_idx: np.ndarray
    qry_idx: np.ndarray

    def all_idx(self):
        return np.concatenate([self.ref_idx, self.sup_idx, self.qry_idx])


class EpisodeSchedule:
    """Deterministic record schedule shared by all training arms (same seed -> same records)."""

    def __init__(self, y, rx, num_classes, seed, n_ref=16, n_query=8, budgets=BUDGETS):
        self.y, self.rx = np.asarray(y), np.asarray(rx)
        self.T, self.seed = int(num_classes), int(seed)
        self.n_ref, self.n_query, self.budgets = n_ref, n_query, tuple(budgets)
        self.rxs = np.unique(self.rx)
        if len(self.rxs) < 3:
            raise ValueError('episodic training needs >= 3 fitting RX (1 pseudo-target + >= 2 reference)')
        self.cells = {(t, r): np.flatnonzero((self.y == t) & (self.rx == r)) for t in range(self.T) for r in self.rxs}
        need = max(self.n_ref, max(self.budgets) + self.n_query)
        short = [(t, int(r), len(v)) for (t, r), v in self.cells.items() if len(v) < need]
        if short:
            raise ValueError(f'cells smaller than {need} records: {short[:5]}')

    def episode(self, step):
        rng = np.random.default_rng([self.seed, step])
        cycle, pos = divmod(step, len(self.rxs))
        order = np.random.default_rng([self.seed, 10 ** 6 + cycle]).permutation(self.rxs)
        tgt = order[pos]
        m = int(rng.choice(self.budgets))
        ref, sup, qry = [], [], []
        for t in range(self.T):
            for r in self.rxs:
                cell = self.cells[(t, r)]
                if r == tgt:
                    pick = rng.choice(cell, m + self.n_query, replace=False)
                    sup.append(pick[:m])
                    qry.append(pick[m:])
                else:
                    ref.append(rng.choice(cell, self.n_ref, replace=False))
        return Episode(step, int(tgt), m, np.concatenate(ref), np.concatenate(sup), np.concatenate(qry))


def protonet_query_logits(z_sup, y_sup, z_qry, num_classes, temperature=1.0):
    """T2: squared Euclidean distance to the support mean (Snell et al. 2017)."""
    s, m, _ = support_means(z_sup, y_sup, num_classes)
    if bool((m == 0).any()):
        raise ValueError('ProtoNet episode without support for some class')
    zq = z_qry.to(s.dtype)
    d2 = (zq ** 2).sum(1, keepdim=True) - 2 * zq @ s.T + (s ** 2).sum(1)[None]
    return -d2 / temperature


def posterior_query_logits(z_ref, y_ref, c_ref, z_sup, y_sup, z_qry, num_classes,
                           cfg: StatsConfig = StatsConfig(), temperature=1.0):
    """T3: differentiable H4 posterior predictive with statistics from the reference features only."""
    stats = fit_source_stats(z_ref, y_ref, c_ref, num_classes, cfg)
    s, m, _ = support_means(z_sup, y_sup, num_classes)
    theta, Q = posterior_predictive(stats, s, m)
    return diag_gaussian_logpdf(z_qry, theta, Q) / temperature


def episode_loss(arm, z, y, ep: Episode, rx, num_classes, native_logits, cfg=StatsConfig(),
                 temperature=1.0, weight=1.0, aux_fn=None):
    """Total loss of one step. z/native_logits are aligned with ep.all_idx() (ref, sup, qry order).
    T1 needs aux_fn(z, y, rx) = the historical ccalign term (scripts/receiver_p2/p2_losses.ccalign),
    imported unchanged by the trainer so that T1 matches the published ccalign arm."""
    y_all = torch.as_tensor(y[ep.all_idx()], device=z.device)
    loss = F.cross_entropy(native_logits, y_all)
    if arm == 'T0':
        return loss, dict(native=float(loss.detach()))
    if arm == 'T1':
        if aux_fn is None:
            raise ValueError('T1 requires the ccalign aux_fn')
        rx_all = torch.as_tensor(rx[ep.all_idx()], device=z.device)
        a = aux_fn(z, y_all, rx_all)
        return loss + weight * a, dict(native=float(loss.detach()), ccalign=float(a.detach()))
    nr, ns = len(ep.ref_idx), len(ep.sup_idx)
    z_ref, z_sup, z_qry = z[:nr], z[nr:nr + ns], z[nr + ns:]
    ys = torch.as_tensor(y[ep.sup_idx], device=z.device)
    yq = torch.as_tensor(y[ep.qry_idx], device=z.device)
    if arm == 'T2':
        lg = protonet_query_logits(z_sup, ys, z_qry, num_classes, temperature)
    elif arm == 'T3':
        yr = torch.as_tensor(y[ep.ref_idx], device=z.device)
        cr = torch.as_tensor(rx[ep.ref_idx], device=z.device)
        lg = posterior_query_logits(z_ref, yr, cr, z_sup, ys, z_qry, num_classes, cfg, temperature)
    else:
        raise ValueError(f'unknown arm {arm}')
    q = F.cross_entropy(lg, yq)          # balanced query (n_query per class) -> TX-equal
    return loss + weight * q, dict(native=float(loss.detach()), episode=float(q.detach()))
