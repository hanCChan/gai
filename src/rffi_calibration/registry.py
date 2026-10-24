"""Configuration registry with explicit path variables (no hard-coded machine paths in code)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
CONFIG_DIR = WORKTREE / 'configs/rffi_calibration'


def _load(name, config_dir=None):
    return json.loads((Path(config_dir or CONFIG_DIR) / name).read_text())


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


class Registry:
    def __init__(self, config_dir=None):
        self.config_dir = Path(config_dir or CONFIG_DIR)
        self.datasets_cfg = _load('datasets.json', self.config_dir)
        self.roles = _load('roles.json', self.config_dir)
        self.protocol = _load('registration.json', self.config_dir)
        self.vars = {k: os.environ.get(k, v) for k, v in self.datasets_cfg['path_variables'].items()}

    def resolve(self, s):
        def sub(m):
            k = m.group(1)
            if k not in self.vars:
                raise KeyError(f'unknown path variable {k}')
            return self.vars[k]
        return Path(re.sub(r'\$\{(\w+)\}', sub, s))

    def dataset(self, name):
        if name not in self.datasets_cfg['datasets']:
            raise KeyError(f'unknown dataset {name}')
        return self.datasets_cfg['datasets'][name]

    def num_classes(self, name):
        return int(self.dataset(name)['num_classes'])

    def path(self, name, key):
        return self.resolve(self.dataset(name)[key])

    def reserved(self, name):
        return list(self.roles[name].get('final_reserved', []))

    def role_of(self, name, rx):
        r = self.roles[name]
        if rx in r.get('final_reserved', []):
            return 'final_reserved'
        if rx in r.get('development', []):
            return 'development'
        if rx in r.get('source', []) or rx in r.get('physical_rx', []):
            return 'source'
        if name == 'osu' and rx in ('RX1', 'RX2'):
            return 'transfer_stress'
        raise KeyError(f'{name}: RX {rx!r} is not registered')

    def label(self, name, raw):
        """Task label from a raw dataset label; unknown labels fail loudly."""
        T = self.num_classes(name)
        if name == 'shen':
            t = int(raw) - 31 if int(raw) >= 31 else int(raw)
        elif name == 'osu':
            t = int(raw) - 1
        else:
            t = int(raw)
        if not 0 <= t < T:
            raise KeyError(f'{name}: raw label {raw} outside the registered {T} classes')
        return t

    def reservation(self):
        return dict(self.protocol['reservation'])

    def fingerprint(self):
        h = hashlib.sha256()
        for n in ('datasets.json', 'roles.json', 'registration.json'):
            h.update((self.config_dir / n).read_bytes())
        return h.hexdigest()
