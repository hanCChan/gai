"""Independent checkpoint verification uses synthetic artifacts only."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from rffi_calibration.registry import WORKTREE, file_sha

spec = importlib.util.spec_from_file_location('stage_a_runner', WORKTREE / 'scripts/rffi_calibration/run_stage_a.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_fit_verifier_checks_hash_and_selection(tmp_path):
    job = dict(steps=4, job_id='synthetic')
    curve = [dict(step=2, val_ce=.5), dict(step=4, val_ce=.7)]
    torch.save(dict(job=job, step=2, val_ce=.5, model={'weight': torch.ones(2)}), tmp_path / 'selected.pt')
    status = dict(state='completed', job=job, registry_fingerprint='test',
                  checkpoint_sha256=file_sha(tmp_path / 'selected.pt'), best=dict(step=2, ce=.5))
    (tmp_path / 'status.json').write_text(json.dumps(status))
    (tmp_path / 'curve.json').write_text(json.dumps(curve))
    assert runner.verify_fit(tmp_path, job, 'test') == status
    status['best']['step'] = 4
    (tmp_path / 'status.json').write_text(json.dumps(status))
    with pytest.raises(AssertionError):
        runner.verify_fit(tmp_path, job, 'test')


def test_one_rx_cannot_select_a_crossfitted_family():
    from rffi_calibration.evaluator import crossfit_family_rows
    rows = [dict(rx='single', head=f'H2[a={a}]', m=1, draw=0, acc=.5) for a in (0, 1)]
    assert crossfit_family_rows(rows) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU inference integration test')
def test_gpu_wrapper_exports_cpu_native_head():
    from rffi_calibration.encoders import build_shen_featurecnn, build_wisig_official
    from rffi_calibration.heads import NativeHead
    for wrapper in (build_wisig_official(), build_shen_featurecnn()):
        wrapper = wrapper.to('cuda')
        weight, bias = wrapper.linear_head()
        assert weight.device.type == bias.device.type == 'cpu'
        z = torch.randn(3, wrapper.feature_dim, dtype=torch.float64)
        head = NativeHead(weight, bias).fit(z, [0, 1, 2], [0, 0, 0], 10)
        assert torch.isfinite(head.adapt().log_scores(z)).all()


def test_evaluation_ledger_recovery_has_no_execution(tmp_path):
    spec = importlib.util.spec_from_file_location(
        'stage_a_eval', WORKTREE / 'scripts/rffi_calibration/run_stage_a_evaluation.py')
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    progress = dict(records=[dict(verified=True) for _ in range(12)],
                    pending=[], running=[], reserved_gpu_seconds=900.)
    (tmp_path / 'evaluation_progress.json').write_text(json.dumps(progress))
    ledger = evaluation.recover_ledger(tmp_path, 'synthetic', 1.5)
    assert ledger['total_reserved_gpu_hours'] == .25 and ledger['ledger_recovered']
    with pytest.raises(AssertionError):
        evaluation.recover_ledger(tmp_path, 'synthetic', 1.5)


def test_independent_metrics_match_the_reference():
    import numpy as np
    from rffi_calibration.evaluator import score_metrics
    spec = importlib.util.spec_from_file_location(
        'stage_a_eval_metrics', WORKTREE / 'scripts/rffi_calibration/run_stage_a_evaluation.py')
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    scores = np.array([[2., 0., 1.], [0., 3., 1.], [1., 2., 0.], [0., 0., 1.]])
    labels = np.array([0, 1, 2, 2])
    actual, confusion = evaluation.independent_metrics(scores, labels)
    prob = torch.softmax(torch.as_tensor(scores), -1).numpy()
    expected = score_metrics(prob, labels, 3)
    for key in ('acc', 'macro_f1', 'brier', 'ece', 'nll'):
        assert abs(actual[key] - expected[key]) < 1e-12
    assert confusion.sum() == 4
