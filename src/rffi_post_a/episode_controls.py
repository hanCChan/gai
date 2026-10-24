"""R0 pilot controls: one common native batch per step and two pre-declared ProtoNet routings
(plan §7.3). All three arms (C_MATCH, P_MIX, P_RX) see exactly the same records (same all_ids SHA)
and the same native CE; they differ only in whether, and how, a ProtoNet query loss is added.

Step s:
  * combo = (pseudo-target physical RX p, budget m) from a seeded permutation of fit_rx x budgets,
    one permutation per cycle, so each (RX, m) appears once per cycle (20 steps for 10 RX x {1,5});
  * p: per class m support records from date 0 and n_query query records from date 1;
  * every other fit RX: per class n_ref_per_date records from date 0 and from date 1;
  * all records are distinct sample IDs from the fit positions.
P_RX routing: support/query = the pseudo-target's support/query (receiver-grouped episode).
P_MIX routing: inside the SAME all_ids, per class t a seeded RX order cyclically shifted by t; the
support takes one date-0 record from each of the first m RX, the query one date-1 record from each of
the first n_query RX (independent orders for support and query). Same counts and date rule as P_RX.
Episode RNG is numpy-only, so Torch's dropout stream is untouched by the sampler.

ProtoNet (Snell et al. 2017; author code jakesnell/prototypical-networks@c9bb4d2, few_shot.py): class
centroid = mean support embedding, logits = -squared Euclidean distance / temperature.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

ARMS = ('C_MATCH', 'P_MIX', 'P_RX')


@dataclass
class ControlEpisode:
    step: int
    pseudo_target: str
    m: int
    all_idx: np.ndarray            # indices into the metadata arrays; native CE uses all of them
    rx_support: np.ndarray         # positions inside all_idx
    rx_query: np.ndarray
    mix_support: np.ndarray
    mix_query: np.ndarray
    mix_rx_support: list           # per class: RX used by the MIX support
    mix_rx_query: list

    def ids_sha(self, ids):
        return hashlib.sha256('|'.join(map(str, np.asarray(ids)[self.all_idx])).encode()).hexdigest()


class ControlSchedule:
    def __init__(self, tx, rx, date, num_classes, seed, budgets=(1, 5), n_query=8, n_ref_per_date=8):
        self.tx, self.rx, self.date = np.asarray(tx), np.asarray(rx).astype(str), np.asarray(date)
        self.T, self.seed = int(num_classes), int(seed)
        self.budgets, self.n_query, self.n_ref = tuple(budgets), int(n_query), int(n_ref_per_date)
        self.rxs = np.unique(self.rx)
        if len(self.rxs) < max(self.n_query, max(self.budgets)) or len(self.rxs) < 3:
            raise ValueError(f'P_MIX needs >= max(n_query, max m) = {max(self.n_query, max(self.budgets))} '
                             f'distinct fit RX; got {len(self.rxs)}')
        if set(np.unique(self.date)) != {0, 1}:
            raise ValueError('control episodes need exactly dates 0 (support/reference) and 1 (query/reference)')
        self.cells = {}
        need = {0: max(self.n_ref, max(self.budgets)), 1: max(self.n_ref, self.n_query)}
        short = []
        for t in range(self.T):
            for r in self.rxs:
                for d in (0, 1):
                    ix = np.flatnonzero((self.tx == t) & (self.rx == r) & (self.date == d))
                    self.cells[(t, r, d)] = ix
                    if len(ix) < need[d]:
                        short.append((t, r, d, len(ix)))
        if short:
            raise ValueError(f'cells below the required counts (no change of sampling rule): {short[:5]}')
        self.combos = [(r, m) for r in self.rxs for m in self.budgets]

    def _combo(self, step):
        cycle, pos = divmod(step, len(self.combos))
        perm = np.random.default_rng([self.seed, 1, cycle]).permutation(len(self.combos))
        return self.combos[perm[pos]]

    def episode(self, step):
        p, m = self._combo(step)
        rng = np.random.default_rng([self.seed, 2, step])
        parts, where = [], {}
        for t in range(self.T):
            for r in self.rxs:
                for d in (0, 1):
                    k = (m if d == 0 else self.n_query) if r == p else self.n_ref
                    pick = rng.choice(self.cells[(t, r, d)], k, replace=False)
                    where[(t, r, d)] = (sum(map(len, parts)), len(pick))
                    parts.append(pick)
        all_idx = np.concatenate(parts)
        pos = lambda t, r, d: np.arange(where[(t, r, d)][0], where[(t, r, d)][0] + where[(t, r, d)][1])
        rx_s = np.concatenate([pos(t, p, 0) for t in range(self.T)])
        rx_q = np.concatenate([pos(t, p, 1) for t in range(self.T)])
        mrng = np.random.default_rng([self.seed, 3, step])
        order_s, order_q = mrng.permutation(self.rxs), mrng.permutation(self.rxs)
        mix_s, mix_q, comp_s, comp_q = [], [], [], []
        for t in range(self.T):
            rs = np.roll(order_s, -t)[:m]
            rq = np.roll(order_q, -t)[:self.n_query]
            mix_s += [int(mrng.choice(pos(t, r, 0))) for r in rs]
            mix_q += [int(mrng.choice(pos(t, r, 1))) for r in rq]
            comp_s.append(rs.tolist())
            comp_q.append(rq.tolist())
        return ControlEpisode(step, str(p), int(m), all_idx, rx_s, rx_q, np.array(mix_s), np.array(mix_q), comp_s, comp_q)


def make_control_episode(metadata, step, schedule_seed):
    """Same all_ids; distinct predeclared MIX/RX support-query routing."""
    sch = ControlSchedule(metadata['tx'], metadata['rx'], metadata['date'], metadata['num_classes'], schedule_seed,
                          metadata.get('budgets', (1, 5)), metadata.get('n_query', 8), metadata.get('n_ref_per_date', 8))
    return sch.episode(step)


def proto_logits(z_support, y_support, z_query, num_classes, temperature=1.0):
    """-||f(q) - c_t||^2 / temperature with c_t the mean support embedding of class t."""
    y = torch.as_tensor(y_support, device=z_support.device).long()
    cnt = torch.bincount(y, minlength=num_classes).to(z_support.dtype)
    if bool((cnt == 0).any()):
        raise ValueError('ProtoNet episode without support for some class')
    c = torch.zeros(num_classes, z_support.shape[1], dtype=z_support.dtype, device=z_support.device)
    c = c.index_add(0, y, z_support) / cnt[:, None]
    d2 = ((z_query[:, None, :] - c[None]) ** 2).sum(-1)
    return -d2 / temperature


def pilot_loss(arm, z, native_logits, y_all, ep: ControlEpisode, num_classes, weight=1.0, temperature=1.0):
    """z / native_logits / y_all aligned with ep.all_idx. Returns (loss, parts) where parts holds the
    separate native and proto terms (tensors) for logging and gradient probes."""
    y_all = torch.as_tensor(y_all, device=z.device).long()
    native = F.cross_entropy(native_logits, y_all)
    if arm == 'C_MATCH':
        return native, dict(native=native)
    if arm == 'P_RX':
        s, q = ep.rx_support, ep.rx_query
    elif arm == 'P_MIX':
        s, q = ep.mix_support, ep.mix_query
    else:
        raise ValueError(f'unknown arm {arm}')
    s, q = torch.as_tensor(s, device=z.device), torch.as_tensor(q, device=z.device)
    lg = proto_logits(z[s], y_all[s], z[q], num_classes, temperature)
    proto = F.cross_entropy(lg, y_all[q])
    acc = (lg.argmax(1) == y_all[q]).float().mean()
    return native + weight * proto, dict(native=native, proto=proto, proto_acc=acc.detach())


def grad_probe(params, loss_a, loss_b):
    """Observation only: norms of and cosine between the two loss gradients (autograd.grad, so the
    gradients used by backward() are not touched)."""
    params = [p for p in params if p.requires_grad]
    ga = torch.autograd.grad(loss_a, params, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(loss_b, params, retain_graph=True, allow_unused=True)
    flat = lambda g: torch.cat([(x if x is not None else torch.zeros_like(p)).flatten() for x, p in zip(g, params)])
    a, b = flat(ga), flat(gb)
    return dict(norm_native=float(a.norm()), norm_proto=float(b.norm()),
                cosine=float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-30)))
