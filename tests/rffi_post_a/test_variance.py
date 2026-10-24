import copy

import numpy as np
import pytest

from conftest import load_by_path
from rffi_post_a.variance import SEEDS, check_reservation, cross_pair_checks, expected_jobs


def active():
    return dict(stage='RV1', active=True, allowed_job_ids=[j['job_id'] for j in expected_jobs()],
                max_fits=12, max_reserved_gpu_hours=.6, allow_feature_extraction=True,
                allow_failed_fit_retry=False, allow_final_numeric_access=False, allow_development_numeric_access=False)


def test_variance_gate_exact_matrix_and_no_other_stage():
    good = active()
    matrix = dict(jobs=expected_jobs())
    check_reservation(good, matrix)
    for bad in (dict(good, active=False), dict(good, stage='R0'), dict(good, max_fits=13),
                dict(good, max_reserved_gpu_hours=.7), dict(good, allow_feature_extraction=False),
                dict(good, allow_failed_fit_retry=True), dict(good, allow_final_numeric_access=True),
                dict(good, allowed_job_ids=good['allowed_job_ids'][:-1])):
        with pytest.raises(SystemExit, match='nothing was read'):
            check_reservation(bad, matrix)
    bad = copy.deepcopy(matrix)
    bad['jobs'][0]['arm'] = 'P_MIX'
    with pytest.raises(SystemExit):
        check_reservation(good, bad)


def test_pairs_are_grouped_by_fold_and_seed_not_fold_alone():
    statuses = {j['job_id']: dict(init_state_sha256=str(j['seed']),
                                all_ids_chain_sha256=j['split'] + str(j['seed'])) for j in expected_jobs()}
    checks = cross_pair_checks(statuses, dict(jobs=expected_jobs()))
    assert len(checks) == 6 and all(c['same_init'] and c['same_all_ids'] for c in checks.values())
    del statuses[expected_jobs()[0]['job_id']]
    assert not cross_pair_checks(statuses, dict(jobs=expected_jobs()))['w0|20261025']['same_init']


def synthetic_rows(deltas):
    rp = load_by_path('rv_summary_test', 'scripts/rffi_post_a/report_variance.py')
    rows = []
    for seed, d in zip(SEEDS, deltas):
        for split, rxs in rp.HELD.items():
            for arm in ('C_MATCH', 'P_RX'):
                for rx in rxs:
                    for head in rp.HEADS:
                        for m in ((0,) if head == 'H0' else (1, 5)):
                            acc = .5 + (d / 100 if arm == 'P_RX' and head != 'H0' else 0)
                            rows.append(dict(split=split, seed=seed, arm=arm, rx=rx, head=head,
                                             m=m, draw=0, checkpoint='last.pt', acc=acc))
    return rp, rows


def test_paired_summary_and_conditional_seed_sd_no_pseudoreplication():
    rp, rows = synthetic_rows((3., 4., 5.))
    result = rp.cohort_summary(rows, SEEDS)
    assert all(v['two_arm_pass'] for v in result['by_seed'].values())
    for split in rp.HELD:
        s = result['folds'][split]['paired_lowshot']
        assert s['n_seeds'] == 3
        np.testing.assert_allclose([s['mean_pp'], s['sd_pp']], [4., 1.], atol=1e-10)
    assert result['combined_paired']['n_seeds'] == 3
    np.testing.assert_allclose(result['combined_paired']['sd_pp'], 1., atol=1e-10)
    # Repeating support draws must not change the replication count or uncertainty.
    repeated = rp.cohort_summary(rows * 20, SEEDS)
    np.testing.assert_allclose(repeated['combined_paired']['ci95_pp'], result['combined_paired']['ci95_pp'])


def test_negative_replication_cannot_be_rescued_by_mean():
    rp, rows = synthetic_rows((-1., 8., 8.))
    result = rp.cohort_summary(rows, SEEDS)
    assert result['seed_mean_checks']['gain']
    assert not all(v['two_arm_pass'] for v in result['by_seed'].values())


def test_missing_pair_fails_instead_of_silently_reducing_matrix():
    rp, rows = synthetic_rows((3., 4., 5.))
    with pytest.raises(KeyError):
        rp.cohort_summary([r for r in rows if not (r['seed'] == SEEDS[0] and r['arm'] == 'P_RX')], SEEDS)


def test_complete_score_verifier_detects_missing_key_and_accuracy_tampering(tmp_path):
    import csv
    import json

    rp = load_by_path('rv_score_test', 'scripts/rffi_post_a/report_variance.py')
    job = expected_jobs()[0]
    rxs = rp.HELD['w0']
    y = np.repeat(np.arange(10), 20)
    scores = {f'{rx}|y_query': y for rx in rxs}
    rows = []
    for rx, h, m, draw in sorted(rp.expected_keys(rxs)):
        scores[f'{rx}|{h}|{m}|{draw}'] = np.eye(10)[y]
        rows.append(dict(rx=rx, head=h, m=m, draw=draw, **job, checkpoint='last.pt', n_query=200, acc=1.))

    def write_rows():
        with (tmp_path / 'rows.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_rows()
    np.savez(tmp_path / 'scores.npz', **scores)
    mans = {rx: 'a' * 64 for rx in rxs}
    (tmp_path / 'meta.json').write_text(json.dumps(dict(manifests=mans, checkpoint_sha256='b' * 64, skipped=[])))
    _, result = rp.verify_evaluation(tmp_path, job, 'last.pt', mans, 'b' * 64)
    assert result['checked'] == 2248 and result['max_abs'] == 0
    rows[0]['acc'] = .99
    write_rows()
    with pytest.raises(ValueError, match='accuracy mismatch'):
        rp.verify_evaluation(tmp_path, job, 'last.pt', mans, 'b' * 64)
    rows[0]['acc'] = 1.
    write_rows()
    del scores[next(k for k in scores if 'y_query' not in k)]
    np.savez(tmp_path / 'scores.npz', **scores)
    with pytest.raises(ValueError, match='score coverage'):
        rp.verify_evaluation(tmp_path, job, 'last.pt', mans, 'b' * 64)
