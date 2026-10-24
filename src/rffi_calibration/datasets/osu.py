"""OSU LoRa (Diff_Receivers_Setup_Indoor_SameTx) adapter: block-first window planning.

Each (TX, RX) is ONE continuous 20 M-sample cf32 capture at 1 MS/s. Fixed raw-sample blocks
(roles.json osu.blocks) are assigned to fit / support / query BEFORE any window exists; windows are
placed on the 8192-sample grid strictly inside a block at >= guard samples from its edges, so no
window crosses a block boundary and support/query windows are separated by the block gaps.
Window selection by periodic-phase coherence (G1 build_osu front end) is applied inside each block at
extraction time; it uses the signal only, never labels or scores. A window is an analysis unit, not a
detected packet.
"""
from __future__ import annotations

import json

import numpy as np

from ..access import AccessGuard
from ..sampling import RecordTable, check_windows

WIN = 8192


def paths(reg, rx, tx):
    base = reg.resolve(reg.dataset('osu')['root']) / rx
    return base / f'Device{tx}_IQ.dat', base / f'Device{tx}_IQ.sigmf-meta'


def read_meta(reg, guard: AccessGuard, rx, tx):
    dat, meta = paths(reg, rx, tx)
    guard.check_metadata('osu', meta, rx=rx, purpose='sigmf metadata')
    m = json.loads(meta.read_text())['_metadata']
    g = m['global']
    ann = m['annotations'][0]
    out = dict(datatype=g['core:datatype'], sample_rate=g['core:sample_rate'],
               sample_count=int(ann.get('core:sample_count', 0)),
               transmitter=int(ann['wines:transmitter']['ID']['Transmitter ID']),
               bytes=dat.stat().st_size)
    if out['datatype'] != 'cf32' or out['sample_rate'] != '1 MS/s':
        raise ValueError(f'{meta}: unexpected datatype/rate {out}')
    if out['transmitter'] != tx:
        raise ValueError(f'{meta}: transmitter id {out["transmitter"]} != file device {tx}')
    if out['bytes'] != out['sample_count'] * 8:
        raise ValueError(f'{dat}: {out["bytes"]} bytes but metadata says {out["sample_count"]} cf32 samples')
    return out


def block_ranges(n_samples, blocks_cfg):
    """[(lo, hi, role)] in raw samples from the registered fractions."""
    return [(int(round(blocks_cfg[r][0] * n_samples)), int(round(blocks_cfg[r][1] * n_samples)), r)
            for r in ('fit', 'support', 'query')]


def plan_windows(n_samples, blocks_cfg, win=WIN):
    """Grid windows fully inside guarded blocks. Returns starts, stops, roles; verified by check_windows."""
    guard = int(blocks_cfg['guard_samples'])
    blocks = block_ranges(n_samples, blocks_cfg)
    starts, roles = [], []
    for lo, hi, role in blocks:
        first = -(-(lo + guard) // win) * win
        s = np.arange(first, hi - guard - win + 1, win)
        starts.append(s)
        roles += [role] * len(s)
    starts = np.concatenate(starts).astype(np.int64)
    stops = starts + win
    probs = check_windows(starts, stops, blocks, guard)
    if probs:
        raise AssertionError(f'window plan violates block integrity: {probs[:3]}')
    return starts, stops, np.array(roles)


def planned_table(reg, guard, rx, num_tx=25):
    """Metadata-only RecordTable of all planned windows of one RX (before coherence filtering)."""
    blocks = reg.roles['osu']['blocks']
    tx, st, sp, ro, cap = [], [], [], [], []
    for k in range(1, num_tx + 1):
        meta = read_meta(reg, guard, rx, k)
        s, e, r = plan_windows(meta['sample_count'], blocks)
        tx.append(np.full(len(s), reg.label('osu', k)))
        st.append(s)
        sp.append(e)
        ro.append(r)
        cap.append(np.full(len(s), f'osu_{rx}_dev{k}'))
    n = sum(map(len, st))
    return RecordTable('osu', np.concatenate(tx), np.full(n, rx), np.full(n, -1), np.concatenate(st), 'window',
                       capture=np.concatenate(cap), raw_stop=np.concatenate(sp), split_role=np.concatenate(ro))


def extract(reg, guard: AccessGuard, table: RecordTable, coherence_min, purpose='osu stage A'):
    """Numeric: coherence filter + G1 Strict CIS for the planned windows. Requires an active reservation.
    Returns (filtered RecordTable, cis float32 [N,1,52,126])."""
    import sys
    from ..registry import WORKTREE
    sys.path.insert(0, str(WORKTREE / 'src'))
    from shen_p2a.representations import paired_maps
    from shen_stage1.frequency import periodic_phase_offset
    ib = np.abs(np.fft.fftfreq(WIN, d=1e-6)) <= 62500
    keep, cis = [], []
    for cap in np.unique(table.capture):
        rx, dev = cap.split('_')[1], int(cap.split('_dev')[1])
        dat, _ = paths(reg, rx, dev)
        guard.check('osu', dat, rx=rx, purpose=purpose)
        mm = np.memmap(dat, dtype='<c8', mode='r')
        ix = np.flatnonzero(table.capture == cap)
        z = np.stack([np.asarray(mm[s:s + WIN]) for s in table.position[ix]])
        ok = periodic_phase_offset(z, 1024, 1e6)['coherence'] >= coherence_min
        z = z[ok].astype(np.complex128)
        keep.append(ix[ok])
        if len(z):
            cis.append(paired_maps(np.fft.ifft(np.fft.fft(z, axis=-1) * ib, axis=-1))[1].astype(np.float32))
    keep = np.concatenate(keep)
    sub = RecordTable('osu', table.tx[keep], table.rx[keep], table.date[keep], table.position[keep], 'window',
                      capture=table.capture[keep], raw_stop=table.raw_stop[keep], split_role=table.split_role[keep])
    return sub, np.concatenate(cis)
