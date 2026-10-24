"""WiSig ManyRx adapter over the audited per-(RX, date) npz cache (receiver_r01.data.wisig layout).

Cache file rx{RR}_day{D}.npz: x float32 (N,1,256,2), y (TX 0..9), rx, position (0..199 inside the cell).
Metadata reads (coverage) load only y/position; x is loaded only after AccessGuard.check(numeric=True).
"""
from __future__ import annotations

import numpy as np

from ..access import AccessGuard
from ..sampling import RecordTable


def cache_path(reg, rx, date):
    d = reg.dataset('wisig')
    return reg.resolve(d['cache']) / d['cache_file_pattern'].format(rx=int(rx), date=int(date))


def load(reg, guard: AccessGuard, rxs, dates, positions=(0, 200), numeric=False, purpose=''):
    """RecordTable (+ x if numeric) for the given RX x dates, positions [lo, hi)."""
    tx, rxa, da, pa, xs = [], [], [], [], []
    for r in rxs:
        for d in dates:
            p = cache_path(reg, r, d)
            if numeric:
                guard.check('wisig', p, rx=r, date=d, purpose=purpose)
            else:
                guard.check_metadata('wisig', p, rx=r, date=d, purpose=purpose or 'coverage')
            with np.load(p, allow_pickle=False) as a:
                pos = a['position']
                keep = (pos >= positions[0]) & (pos < positions[1])
                tx.append(a['y'][keep].astype(np.int64))
                pa.append(pos[keep].astype(np.int64))
                if numeric:
                    x = a['x'][keep]
                    if tuple(x.shape[1:]) != (1, 256, 2) or x.dtype != np.float32:
                        raise ValueError(f'{p}: expected float32 [N,1,256,2], got {x.dtype} {x.shape}')
                    xs.append(x)
            n = int(keep.sum())
            rxa.append(np.full(n, int(r)))
            da.append(np.full(n, int(d)))
    T = reg.num_classes('wisig')
    y = np.concatenate(tx) if tx else np.zeros(0, np.int64)
    if len(y) and (y.min() < 0 or y.max() >= T):
        raise KeyError(f'WiSig label outside the registered {T} classes')
    rx_all = np.concatenate(rxa)
    date_all = np.concatenate(da)
    table = RecordTable('wisig', y, rx_all, date_all, np.concatenate(pa), 'cache_position',
                        capture=np.array([f'rx{r:02d}_day{d}' for r, d in zip(rx_all, date_all)]))
    return table, (np.concatenate(xs) if numeric else None)


def coverage(reg, guard, rxs, dates):
    """Counts per (rx, date, tx) from labels only; used before any score to decide eligibility."""
    table, _ = load(reg, guard, rxs, dates, numeric=False, purpose='coverage')
    T = reg.num_classes('wisig')
    out = {}
    for r in rxs:
        for d in dates:
            m = (table.rx == r) & (table.date == d)
            out[(int(r), int(d))] = np.bincount(table.tx[m], minlength=T).tolist()
    return out
