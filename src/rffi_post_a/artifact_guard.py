"""Fail-closed lineage checks for D1 feature caches and R0 artifacts (plan §6.1, §10 "门禁").

verify_lineage is called BEFORE any weight, feature or signal member is opened. It checks only
metadata dictionaries and the reservation; it never deserialises a checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CACHE_KEY_FIELDS = ('checkpoint_sha256', 'encoder_layer', 'scaler_sha256', 'source_asset_sha256', 'dataset',
                    'rx', 'dates', 'positions', 'manifest_sha256', 'code_sha256', 'dtype', 'role')


class LineageError(PermissionError):
    pass


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def cache_key(meta):
    missing = [k for k in CACHE_KEY_FIELDS if k not in meta]
    if missing:
        raise LineageError(f'cache metadata misses {missing}')
    return canonical_sha({k: meta[k] for k in CACHE_KEY_FIELDS})


def verify_lineage(checkpoint_meta, cache_meta, manifest_meta, reservation, roles=None, task=None):
    """Fail closed before opening weights, features or signal members.

    checkpoint_meta: {'task', 'checkpoint_sha256'} as recorded in status.json (not the weights)
    cache_meta:      the cache's metadata dict incl. 'key' (or None before extraction)
    manifest_meta:   {'rx': manifest sha256} for the task's targets
    reservation:     active reservation of the requested stage
    task:            the registered task entry from diagnostic_matrix.json
    """
    if not reservation.get('active'):
        raise LineageError('reservation inactive')
    if reservation.get('stage') != 'D1':
        raise LineageError(f'reservation stage {reservation.get("stage")!r} is not D1')
    if not reservation.get('allow_feature_extraction'):
        raise LineageError('feature extraction not allowed by the reservation')
    if task is None or task['task'] not in reservation.get('allowed_job_ids', []):
        raise LineageError('task is not in the reservation allowlist')
    if checkpoint_meta.get('checkpoint_sha256') != task['checkpoint_sha256']:
        raise LineageError('checkpoint SHA differs from the registered Stage A status')
    if set(map(str, task['held_rx'])) & set(map(str, task['fit_rx'])):
        raise LineageError('held RX inside fit RX')
    if roles is not None:
        r = roles[task['dataset']]
        forbidden = set(map(str, r.get('final_reserved', []) + r.get('development', [])))
        used = set(map(str, task['held_rx'] + task['fit_rx']))
        if used & forbidden:
            raise LineageError(f'sealed/development RX requested: {sorted(used & forbidden)}')
        if set(map(str, used)) - set(map(str, r['source'])):
            raise LineageError('unknown RX requested')
    for rx, pref in task['manifest_sha256_prefix'].items():
        sha = manifest_meta.get(str(rx))
        if sha is None or not sha.startswith(pref):
            raise LineageError(f'manifest of RX {rx} does not match the registered Stage A manifest')
    if cache_meta is not None:
        if cache_meta.get('key') != cache_key(cache_meta):
            raise LineageError('cache key does not match its metadata (forged or stale cache)')
        if cache_meta['checkpoint_sha256'] != task['checkpoint_sha256']:
            raise LineageError('cache built from another checkpoint')
        allowed = set(map(str, task['fit_rx'])) if cache_meta['role'] == 'source' else set(map(str, task['held_rx']))
        if not set(map(str, cache_meta['rx'])) <= allowed:
            raise LineageError(f'cache RX {cache_meta["rx"]} outside the {cache_meta["role"]} RX of the task')
    return True


class ArtifactGuard:
    """Opens cache files only under a fixed root, after lineage verification, never through symlinks
    that leave the root."""

    def __init__(self, root):
        self.root = Path(os.path.realpath(root))

    def path(self, name):
        p = Path(name)
        if p.is_absolute() or '..' in p.parts:
            raise LineageError(f'cache name {name} must be relative without ..')
        real = Path(os.path.realpath(self.root / p))
        if real != self.root and self.root not in real.parents:
            raise LineageError(f'{name} resolves outside the cache root')
        return real

    def load_meta(self, name):
        meta = json.loads(self.path(name + '.meta.json').read_text())
        if meta.get('key') != cache_key(meta):
            raise LineageError('cache key mismatch')
        return meta

    def open_features(self, name, task, reservation, manifest_meta, roles=None):
        import numpy as np
        meta = self.load_meta(name)
        verify_lineage(dict(checkpoint_sha256=meta['checkpoint_sha256']), meta, manifest_meta, reservation, roles, task)
        p = self.path(name + '.npz')
        with np.load(p, allow_pickle=False) as a:
            if a['z'].dtype != np.float64:
                raise LineageError('cached features must be float64')
            return meta, {k: a[k] for k in a.files}

    def write(self, name, meta, **arrays):
        import numpy as np
        meta = dict(meta, key=cache_key(meta))
        p = self.path(name + '.npz')
        if p.exists():
            raise LineageError(f'{p} exists: caches are write-once')
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **arrays)
        self.path(name + '.meta.json').write_text(json.dumps(meta, indent=1, default=str))
        return meta
