"""Shen LoRa adapter over the G1 Strict CIS cache (receiver_r01.data.shen layout).

Cache dir shen_<rx>_train: strict_cis.npy (2000,1,52,126) float32, labels.npy (0..9), rows.npy (original
HDF5 row), manifest.json. Partition 'fit' = first 160 occurrences of each TX in file order, 'val' = the
last 40 (identical to receiver_r01.data.shen). Only the 10 source RX are cached on this machine.
"""
from __future__ import annotations

import json

import numpy as np

from ..access import AccessGuard
from ..sampling import RecordTable


def cache_dir(reg, rx):
    return reg.resolve(reg.dataset('shen')['cache']) / f'shen_{rx}_train'


def _partition_index(labels, partition, T):
    sl = {'fit': slice(0, 160), 'val': slice(160, 200), 'all': slice(None)}[partition]
    return np.concatenate([np.flatnonzero(labels == t)[sl] for t in range(T)])


def load(reg, guard: AccessGuard, rxs, partition='all', numeric=False, scaler=None, purpose=''):
    """RecordTable (+ scaled x if numeric). scaler = {'mean','std'} from the checkpoint / fitting records."""
    T = reg.num_classes('shen')
    tx, rxa, pa, xs = [], [], [], []
    for r in rxs:
        d = cache_dir(reg, r)
        if numeric:
            guard.check('shen', d / 'strict_cis.npy', rx=r, purpose=purpose)
        else:
            guard.check_metadata('shen', d / 'labels.npy', rx=r, purpose=purpose or 'coverage')
        info = json.loads((d / 'manifest.json').read_text())
        if info['source']['role'] != 'train' or info.get('status') != 'completed':
            raise ValueError(f'{d}: not a completed source-RX cache')
        lab = np.load(d / 'labels.npy', allow_pickle=False)
        if not np.array_equal(np.bincount(lab, minlength=T), np.full(T, 200)):
            raise ValueError(f'{d}: expected 200 rows per TX')
        ix = _partition_index(lab, partition, T)
        rows = np.load(d / 'rows.npy', allow_pickle=False)
        tx.append(lab[ix].astype(np.int64))
        pa.append(rows[ix].astype(np.int64))
        rxa.append(np.full(len(ix), r, dtype=object))
        if numeric:
            a = np.load(d / 'strict_cis.npy', mmap_mode='r', allow_pickle=False)
            if a.shape != (2000, 1, 52, 126):
                raise ValueError(f'{d}: unexpected CIS shape {a.shape}')
            xs.append(np.asarray(a[ix]))
    rx_all = np.concatenate(rxa).astype(str)
    table = RecordTable('shen', np.concatenate(tx), rx_all, np.full(len(rx_all), -1), np.concatenate(pa), 'packet')
    x = None
    if numeric:
        x = np.concatenate(xs)
        if scaler is None:
            raise ValueError('Shen numeric load needs the checkpoint scaler (fitted on fitting records only)')
        x = ((x - scaler['mean']) / scaler['std']).astype(np.float32)
    return table, x


def fit_scaler(x_fit):
    """Global mean/std over the actual fitting packets only (receiver_r01.data.shen convention)."""
    return dict(mean=float(np.mean(x_fit, dtype=np.float64)), std=float(np.std(x_fit, dtype=np.float64)))
