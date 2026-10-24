"""Support/query manifests, sample IDs, evaluator leakage checks and the historical cc-leak regression."""
import numpy as np
import pytest
import torch

from rffi_calibration.evaluator import (FeatureSet, aggregate, crossfit_family_rows, evaluate_target, fit_heads,
                                        paired, score_metrics, select_family_members)
from rffi_calibration.heads import PosteriorPredictive, ScalarShrink, TargetPrototype, default_heads
from rffi_calibration.sampling import (RecordTable, build_manifest, check_windows, duplicates, manifest_sha,
                                       support_ids, validate_manifest)


def _table(n_rx=3, T=4, per=(40, 40), seed=0, d=5, informative=True):
    """Two dates per RX; per = packets per (rx, class, date)."""
    rng = np.random.default_rng(seed)
    proto = 3 * rng.standard_normal((T, d))
    tx, rx, date, pos, z = [], [], [], [], []
    for r in range(n_rx):
        shift = rng.standard_normal((T, d))
        for di, n in enumerate(per):
            for t in range(T):
                tx += [t] * n
                rx += [r] * n
                date += [di] * n
                pos += list(range(t * 1000, t * 1000 + n))
                mean = proto[t] + shift[t] if informative else np.zeros(d)
                z.append(mean + rng.standard_normal((n, d)))
    tab = RecordTable('toy', np.array(tx), np.array(rx), np.array(date), np.array(pos), 'cache_position',
                      capture=np.array([f'rx{r}_day{d}' for r, d in zip(rx, date)]))
    return tab, np.concatenate(z)


RULE = {'type': 'by_date', 'support_dates': [0], 'query_dates': [1]}


def test_sample_ids_are_stable_and_unique():
    tab, _ = _table()
    tab2, _ = _table()
    assert (tab.ids == tab2.ids).all()
    assert not duplicates(tab.ids)
    with pytest.raises(ValueError, match='duplicate'):
        RecordTable('toy', np.array([0, 0]), np.array([1, 1]), np.array([0, 0]), np.array([5, 5]), 'packet')


def test_manifest_disjoint_nested_and_budget_independent_query():
    tab, _ = _table()
    man = build_manifest(tab, 1, 4, RULE, q_per_class=20, pool_max=10, n_draws=5)
    assert man['full_coverage']
    q = set(man['query_ids'])
    for k in range(5):
        prev = set()
        for m in (1, 2, 5, 10):
            s = set(support_ids(man, k, m))
            assert not (s & q)                                  # support never in the query
            assert prev <= s and len(s) == 4 * m                # nested prefixes, m per class
            prev = s
    # identical manifest on rebuild (pure function of metadata)
    assert manifest_sha(man) == manifest_sha(build_manifest(tab, 1, 4, RULE, q_per_class=20, pool_max=10, n_draws=5))
    with pytest.raises(ValueError):
        support_ids(man, 0, 11)


def test_manifest_rejects_overlap_and_reports_partial_coverage():
    tab, _ = _table(per=(40, 5))
    man = build_manifest(tab, 0, 4, RULE, q_per_class=20)
    assert not man['full_coverage'] and man['eligible_classes'] == []
    assert man['coverage'][0]['query'] == 5
    bad = {'type': 'by_date', 'support_dates': [0, 1], 'query_dates': [1]}
    with pytest.raises(ValueError, match='overlap'):
        build_manifest(_table()[0], 0, 4, bad)
    man = build_manifest(_table()[0], 0, 4, RULE)
    man['query_ids'].append(man['support_pool_ids']['0'][0])
    with pytest.raises(ValueError):
        validate_manifest(man)


def test_by_order_rule_with_guard():
    tab, _ = _table(per=(60,))
    rule = {'type': 'by_order', 'support_frac': [0., .5], 'query_frac': [.5, 1.], 'guard': 2}
    man = build_manifest(tab, 0, 4, rule, q_per_class=20)
    pool = set(i for v in man['support_pool_ids'].values() for i in v)
    assert not pool & set(man['query_ids'])
    rows = tab.index_of(man['query_ids'])
    sup_rows = tab.index_of(sorted(pool))
    for t in range(4):
        assert tab.position[rows][tab.tx[rows] == t].min() > tab.position[sup_rows][tab.tx[sup_rows] == t].max() + 2


def test_window_integrity_checks():
    blocks = [(0, 100, 'fit'), (120, 200, 'query')]
    assert check_windows([0, 10], [8, 18], blocks) == []
    assert check_windows([95], [105], blocks)                    # crosses a block edge
    assert check_windows([0, 4], [8, 12], blocks)                # overlapping windows
    assert check_windows([1], [9], blocks, guard=2)              # violates the guard


def _evaluate(tab, z, heads, target_rx=2, T=4, n_draws=3):
    src = np.isin(tab.rx, [0, 1])
    sub = lambda m: RecordTable('toy', tab.tx[m], tab.rx[m], tab.date[m], tab.position[m], 'cache_position',
                                capture=np.array([f'rx{r}_day{d}' for r, d in zip(tab.rx[m], tab.date[m])]))
    source = FeatureSet(z[src], sub(src), {})
    fitted, skipped = fit_heads(heads, source, T)
    tgt = tab.rx == target_rx
    target = FeatureSet(z[tgt], sub(tgt), {})
    man = build_manifest(target.table, target_rx, T, RULE, q_per_class=20, n_draws=n_draws)
    return evaluate_target(fitted, target, man, T, keep_logits=True), man, target


def test_query_label_permutation_changes_metrics_only():
    """End-to-end: permuting the stored query labels (sample identities unchanged) leaves every logit
    unchanged; only the metrics move."""
    import copy
    tab, z = _table()
    heads = default_heads()
    (rows, logits), man, target = _evaluate(tab, z, heads)
    qrows = target.rows(man['query_ids'])
    perm_table = copy.deepcopy(target.table)
    perm_table.tx = perm_table.tx.copy()
    perm_table.tx[qrows] = np.random.default_rng(2).permutation(perm_table.tx[qrows])
    rows2, logits2 = evaluate_target(heads, FeatureSet(target.z, perm_table, {}), man, 4, keep_logits=True)
    assert logits.keys() == logits2.keys()
    for k in logits:
        np.testing.assert_array_equal(logits[k], logits2[k])
    assert any(a['acc'] != b['acc'] for a, b in zip(rows, rows2))


def test_uninformative_features_stay_at_chance():
    """Regression for P2E audit A1: on features without class information, every label-free head must be
    near chance (the leaky h - D[y_true] decode scored 100% here)."""
    tab, z = _table(informative=False, per=(40, 200))
    (rows, _), _, _ = _evaluate(tab, z, default_heads(), n_draws=5)
    for r in rows:
        assert r['acc'] < 0.45, r                         # chance = 0.25 with 4 classes, 80 queries


def test_metrics_and_paired_reporting():
    p = np.array([[.9, .1], [.2, .8], [.6, .4]])
    y = np.array([0, 1, 1])
    m = score_metrics(p, y, 2)
    assert abs(m['acc'] - 2 / 3) < 1e-12 and m['recall'][1] == .5
    assert abs(m['nll'] - (-np.log([.9, .8, .4]).mean())) < 1e-12
    rows = [dict(head=h, m=1, rx=r, draw=k, acc=a) for h, base in (('A', .6), ('B', .5)) for r in range(4)
            for k in range(3) for a in [base + .01 * r]]
    pr = paired(rows, 'A', 'B', 1)
    assert pr['n_rx'] == 4 and abs(pr['mean'] - .1) < 1e-12 and pr['n_improved'] == 4
    ag = aggregate(rows)
    assert ag[('A', 1)]['n_rx'] == 4 and abs(ag[('A', 1)]['worst'] - .6) < 1e-12


def test_family_selection_is_source_only_and_crossfitted():
    rows = []
    for rx, best in (('a', .25), ('b', .25), ('c', .75)):
        for alpha in (0., .25, .5, .75, 1.):
            rows.append(dict(head=f'H2[a={alpha:g}]', m=1, rx=rx, draw=0, acc=1 - abs(alpha - best)))
    ch = select_family_members(rows)
    assert ch[('H2', 1)] == 'H2[a=0.25]'
    cf = crossfit_family_rows(rows)
    sel = {r['rx']: r['selected'] for r in cf}
    assert sel['c'] == 'H2[a=0.25]'           # chosen without c's own score
    assert sel['a'] in ('H2[a=0.25]', 'H2[a=0.5]')
