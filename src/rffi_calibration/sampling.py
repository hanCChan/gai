"""Sample identity, fixed support/query manifests and leakage checks (EXPERIMENT_PROTOCOL.md §3).

Every record carries a stable sample ID built from dataset / capture / TX / original position, never
from an array index of a rebuilt cache. Unknown fields stay None (a cache position is not a timestamp
and an analysis window is not a packet).

Support/query rule (frozen before any target score):
  * per (target RX, class): a fixed QUERY set and a disjoint SUPPORT POOL, defined by a registered
    region rule (different date / capture block / order region) -- never "all remaining packets";
  * draw k = an ordered random subset of size pool_max (10) of the support pool, seeded by
    (dataset, rx, class, k) only; budget m uses the first m of that order, so m=1 c m=2 c m=5 c m=10;
  * the query set does not depend on m or on the draw; every head/encoder uses the same IDs.
Coverage eligibility uses counts only (support_pool >= pool_max, query >= q_per_class), never scores.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields

import numpy as np

SAMPLE_UNITS = ('packet', 'cache_position', 'window')


@dataclass(frozen=True)
class SampleRecord:
    dataset_id: str
    tx_id: int                       # task label 0..T-1 (registry label map)
    physical_rx_id: str
    condition_id: str
    capture_id: str | None
    packet_id: str | None
    raw_sample_start: int | None
    raw_sample_stop: int | None
    original_row: int | None
    date_id: int | None
    split_role: str
    sample_unit: str

    def __post_init__(self):
        if self.sample_unit not in SAMPLE_UNITS:
            raise ValueError(f'unknown sample unit {self.sample_unit}')

    @property
    def sample_id(self):
        return sample_id(self.dataset_id, self.capture_id or self.physical_rx_id, self.tx_id, self.date_id,
                         self.original_row if self.original_row is not None else self.raw_sample_start)


def sample_id(dataset, capture, tx, date, position):
    if position is None:
        raise ValueError('a sample ID needs an original row or a raw sample start')
    return f'{dataset}|{capture}|tx{int(tx)}|d{"-" if date is None else int(date)}|p{int(position)}'


@dataclass
class RecordTable:
    """Column store of SampleRecords aligned with a feature matrix (row i <-> feature row i)."""
    dataset_id: str
    tx: np.ndarray
    rx: np.ndarray                   # physical RX id (str)
    date: np.ndarray                 # int, -1 = unknown
    position: np.ndarray             # original row / raw sample start
    sample_unit: str
    capture: np.ndarray | None = None
    raw_stop: np.ndarray | None = None
    split_role: np.ndarray | None = None
    ids: np.ndarray = field(init=False)

    def __post_init__(self):
        n = len(self.tx)
        for f in ('rx', 'date', 'position'):
            if len(getattr(self, f)) != n:
                raise ValueError(f'RecordTable column {f} has wrong length')
        cap = self.capture if self.capture is not None else self.rx
        self.ids = np.array([sample_id(self.dataset_id, c, t, None if d < 0 else d, p)
                             for c, t, d, p in zip(cap, self.tx, self.date, self.position)])
        dup = duplicates(self.ids)
        if dup:
            raise ValueError(f'duplicate sample IDs, e.g. {dup[:3]}')

    def __len__(self):
        return len(self.tx)

    def index_of(self, ids):
        lut = {s: i for i, s in enumerate(self.ids)}
        try:
            return np.array([lut[s] for s in ids], dtype=np.int64)
        except KeyError as e:
            raise KeyError(f'sample ID not in table: {e}') from None


def duplicates(ids):
    u, c = np.unique(np.asarray(ids), return_counts=True)
    return u[c > 1].tolist()


def _seed(*parts):
    h = hashlib.sha256('|'.join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], 'little')


def _region(table, idx, rule, region):
    """Indices (into table) of the records of one (rx, class) group inside a registered region.
    rule types:
      by_date   : {'support_dates': [...], 'query_dates': [...]}
      by_order  : {'support_frac': [a, b], 'query_frac': [c, d], 'guard': g}   order = (date, position)
      by_role   : {'support_role': 'support', 'query_role': 'query'} (records tagged upstream, e.g. OSU blocks)
    """
    t = rule['type']
    if t == 'by_date':
        dates = rule[f'{region}_dates']
        return idx[np.isin(table.date[idx], dates)]
    if t == 'by_order':
        o = idx[np.lexsort((table.position[idx], table.date[idx]))]
        n = len(o)
        a, b = rule[f'{region}_frac']
        g = int(rule.get('guard', 0))
        lo, hi = int(np.floor(a * n)), int(np.floor(b * n))
        if region == 'query':
            lo = min(hi, lo + g)               # guard records between the two regions are dropped
        return o[lo:hi]
    if t == 'by_role':
        return idx[table.split_role[idx] == rule[f'{region}_role']]
    raise ValueError(f'unknown support/query rule {t}')


def _even_subset(ix, k, table):
    """Deterministic, order-spread subset of size k (no RNG, no dependence on any score)."""
    o = ix[np.lexsort((table.position[ix], table.date[ix]))]
    if len(o) <= k:
        return o
    return o[np.linspace(0, len(o) - 1, k).round().astype(int)]


def build_manifest(table: RecordTable, target_rx, num_classes, rule, q_per_class=20, pool_max=10,
                   n_draws=20, seed_tag='v1'):
    """Fixed query + nested support draws for one target RX. Pure function of metadata."""
    rows = np.flatnonzero(table.rx == target_rx)
    cov, query, pools, draws = {}, [], {}, [[] for _ in range(n_draws)]
    eligible_classes = []
    for t in range(num_classes):
        ix = rows[table.tx[rows] == t]
        sup = _region(table, ix, rule, 'support')
        qry = _region(table, ix, rule, 'query')
        if np.intersect1d(sup, qry).size:
            raise ValueError(f'support and query regions overlap for rx {target_rx} class {t}')
        q = _even_subset(qry, q_per_class, table)
        ok = len(sup) >= pool_max and len(q) >= q_per_class
        cov[t] = dict(support_pool=int(len(sup)), query_region=int(len(qry)), query=int(len(q)), eligible=bool(ok))
        if not ok:
            continue
        eligible_classes.append(t)
        query += table.ids[q].tolist()
        pools[t] = table.ids[np.sort(sup)].tolist()
        for k in range(n_draws):
            rng = np.random.default_rng(_seed(table.dataset_id, target_rx, t, k, seed_tag))
            draws[k].append(rng.permutation(len(sup))[:pool_max].tolist())
    full = len(eligible_classes) == num_classes
    man = dict(dataset=table.dataset_id, rx=str(target_rx), rule=rule, q_per_class=q_per_class,
               pool_max=pool_max, n_draws=n_draws, seed_tag=seed_tag, coverage=cov,
               eligible_classes=eligible_classes, full_coverage=bool(full),
               query_ids=query, support_pool_ids={str(t): v for t, v in pools.items()},
               draws=[{str(t): d[i] for i, t in enumerate(eligible_classes)} for d in draws])
    validate_manifest(man)
    return man


def support_ids(manifest, draw, m):
    """Sample IDs of the m-shot support of one draw (nested prefixes)."""
    if m > manifest['pool_max']:
        raise ValueError(f'm={m} exceeds the registered support pool size {manifest["pool_max"]}')
    out = []
    for t, order in manifest['draws'][draw].items():
        pool = manifest['support_pool_ids'][t]
        out += [pool[i] for i in order[:m]]
    return out


def validate_manifest(man):
    q = set(man['query_ids'])
    if len(q) != len(man['query_ids']):
        raise ValueError('duplicate query IDs')
    allpool = set(i for v in man['support_pool_ids'].values() for i in v)
    if q & allpool:
        raise ValueError('query IDs inside the support pool')
    for d in man['draws']:
        for t, order in d.items():
            if len(set(order)) != len(order) or len(order) != man['pool_max']:
                raise ValueError('support draw is not an ordered subset of size pool_max')
    return True


def manifest_sha(man):
    return hashlib.sha256(json.dumps(man, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------------- window/block integrity
def check_windows(starts, stops, blocks, guard=0):
    """Windows [start, stop) must lie inside exactly one block (lo, hi, role) at >= guard samples from
    its edges, and windows of the same capture must not overlap. Returns list of problems."""
    starts, stops = np.asarray(starts), np.asarray(stops)
    probs = []
    for i, (s, e) in enumerate(zip(starts, stops)):
        inside = [b for b in blocks if s >= b[0] + guard and e <= b[1] - guard]
        if len(inside) != 1:
            probs.append(f'window {i} [{s},{e}) not inside exactly one guarded block')
    o = np.argsort(starts)
    ov = np.flatnonzero(starts[o][1:] < stops[o][:-1])
    probs += [f'windows {o[k]} and {o[k + 1]} overlap' for k in ov]
    return probs


def record_fields():
    return [f.name for f in fields(SampleRecord)]
