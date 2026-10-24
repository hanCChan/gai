"""Numeric-access guard: role checks happen BEFORE a numeric file is opened.

Every request names dataset, RX, date and purpose. Refused when
  * the RX/date is final_reserved and the active reservation does not allow final numeric access;
  * the resolved path leaves the dataset's registered root (path traversal, symlink escape);
  * numeric access is requested while the reservation does not allow feature extraction
    (metadata reads -- sizes, sidecar JSON, label/position arrays used for coverage -- are logged separately).
The log never contains numeric content, only path, role, purpose and decision.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .registry import Registry


class AccessDenied(PermissionError):
    pass


class AccessGuard:
    def __init__(self, registry: Registry, reservation=None, log_path=None, stage='E3'):
        self.reg = registry
        self.res = dict(reservation if reservation is not None else registry.reservation())
        self.stage = stage
        self.log_path = Path(log_path) if log_path else None

    # ---------------------------------------------------------------- helpers
    def _log(self, **kw):
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open('a') as f:
            f.write(json.dumps(dict(t=time.time(), stage=self.stage, **kw), default=str) + '\n')

    def _roots(self, dataset):
        d = self.reg.dataset(dataset)
        keys = [k for k in ('cache', 'root', 'archive') if k in d]
        return [self.reg.resolve(d[k]) for k in keys]

    def _inside(self, path, dataset):
        real = Path(os.path.realpath(path))
        for root in self._roots(dataset):
            rr = Path(os.path.realpath(root))
            if real == rr or rr in real.parents:
                return real
        raise AccessDenied(f'{path} resolves to {real}, outside the registered roots of {dataset}')

    def _sealed(self, dataset, rx, date):
        r = self.reg.roles[dataset]
        if rx is not None and rx in r.get('final_reserved', []):
            return True
        if date is not None and date in r.get('reserved_dates', []):
            return True
        return False

    # ---------------------------------------------------------------- public
    def check(self, dataset, path, rx=None, date=None, purpose='', numeric=True):
        """Return the resolved path if the access is allowed; raise AccessDenied otherwise."""
        if '..' in Path(path).parts:
            self._log(kind='numeric' if numeric else 'metadata', dataset=dataset, path=str(path), decision='deny_traversal')
            raise AccessDenied(f'path traversal component in {path}')
        real = self._inside(path, dataset)
        if self._sealed(dataset, rx, date) and numeric and not self.res.get('allow_final_numeric_access', False):
            self._log(kind='numeric', dataset=dataset, path=str(path), rx=rx, date=date, purpose=purpose, decision='deny_sealed')
            raise AccessDenied(f'{dataset} RX {rx} / date {date} is final_reserved; numeric access is sealed')
        if numeric and rx in self.reg.roles[dataset].get('development', []) and not self.res.get(
                'allow_development_numeric_access', False):
            self._log(kind='numeric', dataset=dataset, path=str(path), rx=rx, date=date,
                      purpose=purpose, decision='deny_development')
            raise AccessDenied(f'{dataset} RX {rx} is development; numeric access is not authorized')
        if numeric and not (self.res.get('active') and self.res.get('allow_feature_extraction', False)):
            self._log(kind='numeric', dataset=dataset, path=str(path), rx=rx, date=date, purpose=purpose, decision='deny_inactive')
            raise AccessDenied('numeric access requires an active reservation with allow_feature_extraction=true')
        self._log(kind='numeric' if numeric else 'metadata', dataset=dataset, path=str(real), rx=rx, date=date,
                  purpose=purpose, decision='allow')
        return real

    def check_metadata(self, dataset, path, rx=None, date=None, purpose=''):
        """Metadata (sizes, sidecars, label/position members for coverage). Sealed RX/dates are still
        refused for anything but existence/size, which callers obtain with os.stat themselves."""
        if self._sealed(dataset, rx, date):
            self._log(kind='metadata', dataset=dataset, path=str(path), rx=rx, date=date, purpose=purpose, decision='deny_sealed')
            raise AccessDenied(f'{dataset} RX {rx} / date {date} is final_reserved')
        return self.check(dataset, path, rx, date, purpose, numeric=False)
