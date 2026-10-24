"""Registry/label maps, sealed-access guard, queue state machine, encoder wrapper parity."""
import json
import os
import shutil
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from rffi_calibration.access import AccessDenied, AccessGuard
from rffi_calibration.queue import BudgetExceeded, JobQueue
from rffi_calibration.registry import CONFIG_DIR, Registry


@pytest.fixture
def reg(tmp_path):
    """Registry pointed at a temporary data tree (no real data touched)."""
    cfg = tmp_path / 'cfg'
    shutil.copytree(CONFIG_DIR, cfg)
    d = json.loads((cfg / 'datasets.json').read_text())
    d['path_variables'] = {'GAI_DATA': str(tmp_path / 'data'), 'RETRAIN_ROOT': str(tmp_path / 'retrain')}
    (cfg / 'datasets.json').write_text(json.dumps(d))
    protocol = json.loads((cfg / 'registration.json').read_text())
    protocol['reservation'].update(active=False, allow_feature_extraction=False)
    (cfg / 'registration.json').write_text(json.dumps(protocol))
    cache = tmp_path / 'retrain/outputs/wisig_w01/cache'
    cache.mkdir(parents=True)
    for r, day in ((6, 0), (29, 0), (3, 3)):
        (cache / f'rx{r:02d}_day{day}.npz').write_bytes(b'x')
    (tmp_path / 'outside').mkdir()
    (tmp_path / 'outside/secret.npz').write_bytes(b'x')
    os.symlink(tmp_path / 'outside/secret.npz', cache / 'rx07_day0.npz')
    return Registry(cfg)


def test_label_maps_and_unknown_classes(reg):
    assert reg.num_classes('wisig') == 10 and reg.num_classes('shen') == 10 and reg.num_classes('osu') == 25
    assert reg.label('shen', 31) == 0 and reg.label('shen', 40) == 9
    assert reg.label('osu', 1) == 0 and reg.label('osu', 25) == 24
    for ds, raw in (('shen', 41), ('osu', 26), ('osu', 0), ('wisig', 10)):
        with pytest.raises(KeyError):
            reg.label(ds, raw)
    with pytest.raises(KeyError):
        reg.dataset('oracle')


def test_roles_are_disjoint_and_match_the_registered_lists(reg):
    w = reg.roles['wisig']
    assert len(w['source']) == 14 and len(w['development']) == 10 and len(w['final_reserved']) == 8
    assert not set(w['source']) & set(w['development']) and not set(w['final_reserved']) & set(w['source'] + w['development'])
    for f in w['folds'].values():
        assert not set(f['fit']) & set(f['source_selection'])
        assert set(f['fit']) | set(f['source_selection']) == set(w['source'])
    s = reg.roles['shen']
    assert set(s['outer_groups']['A']) | set(s['outer_groups']['B']) == set(s['source'])
    assert not set(s['final_reserved']) & set(s['source'] + s['development'])
    for g, sel in s['source_selection_tasks'].items():
        assert set(sel) <= set(s['outer_groups'][g])


def test_sealed_rx_and_date_cannot_be_opened(reg, tmp_path):
    cache = tmp_path / 'retrain/outputs/wisig_w01/cache'
    open_res = dict(reg.reservation(), active=True, allow_feature_extraction=True)
    g = AccessGuard(reg, open_res, log_path=tmp_path / 'access.jsonl')
    assert g.check('wisig', cache / 'rx06_day0.npz', rx=6, date=0, purpose='t') == (cache / 'rx06_day0.npz').resolve()
    with pytest.raises(AccessDenied, match='final_reserved'):
        g.check('wisig', cache / 'rx29_day0.npz', rx=29, date=0)
    with pytest.raises(AccessDenied, match='final_reserved'):
        g.check('wisig', cache / 'rx03_day3.npz', rx=3, date=3)
    with pytest.raises(AccessDenied, match='development'):
        g.check('wisig', cache / 'rx03_day0.npz', rx=3, date=0)
    with pytest.raises(AccessDenied, match='outside'):
        g.check('wisig', cache / 'rx07_day0.npz', rx=7, date=0)            # symlink escape
    with pytest.raises(AccessDenied, match='traversal'):
        g.check('wisig', str(cache) + '/../cache/rx06_day0.npz', rx=6, date=0)
    with pytest.raises(AccessDenied):
        g.check_metadata('wisig', cache / 'rx29_day0.npz', rx=29, date=0)
    # inactive reservation: no numeric access at all
    with pytest.raises(AccessDenied, match='active reservation'):
        AccessGuard(reg).check('wisig', cache / 'rx06_day0.npz', rx=6, date=0)
    with pytest.raises(AccessDenied, match='active reservation'):
        AccessGuard(reg, dict(open_res, active=False)).check('wisig', cache / 'rx06_day0.npz', rx=6, date=0)
    log = (tmp_path / 'access.jsonl').read_text()
    assert 'deny_sealed' in log and 'allow' in log and 'x' not in json.loads(log.splitlines()[0]).get('content', '')


def _queue(tmp_path, n=3, pred=100.):
    q = JobQueue(tmp_path / 'q.sqlite')
    for i in range(n):
        q.add(job_id=f'j{i}', stage='E4', method_id='T0', dataset_id='wisig', seed=i, predicted_gpu_seconds=pred)
    return q


def test_queue_no_double_claim_and_reservation_guard(tmp_path):
    q = _queue(tmp_path, n=6)
    res = dict(active=True, allowed_job_ids=[f'j{i}' for i in range(6)], max_reserved_gpu_hours=1.0)
    got, lock = [], threading.Lock()

    def worker(gpu):
        while True:
            j = q.claim(gpu, res)
            if j is None:
                return
            with lock:
                got.append(j['job_id'])

    ts = [threading.Thread(target=worker, args=(g,)) for g in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(got) == [f'j{i}' for i in range(6)]                  # each job exactly once
    assert q.claim(0, dict(res, active=False)) is None


def test_queue_budget_and_allowlist(tmp_path):
    q = _queue(tmp_path, n=3, pred=2000.)
    res = dict(active=True, allowed_job_ids=['j0', 'j1'], max_reserved_gpu_hours=1.0)   # 3600 s
    assert q.claim(0, res)['job_id'] == 'j0'
    with pytest.raises(BudgetExceeded):
        q.claim(1, res)                                                 # 2000 + 2000 > 3600
    assert q.get('j2')['status'] == 'pending'                           # not allow-listed, untouched
    assert q.claim(0, dict(res, allowed_job_ids=[])) is None


def test_queue_completion_requires_verification_and_sigterm_is_not_completed(tmp_path):
    q = _queue(tmp_path, n=3)
    res = dict(active=True, allowed_job_ids=['j0', 'j1', 'j2'], max_reserved_gpu_hours=1.0)
    for _ in range(3):
        q.claim(0, res)
    assert q.complete('j0', 0, artifacts_ok=True, verifier_ok=True) == 'completed'
    assert q.complete('j1', 0, artifacts_ok=True, verifier_ok=False) == 'failed'
    q.cancel('j2', gpu_reserved_seconds=17.)
    assert q.get('j2')['status'] == 'cancelled'
    assert q.get('j2')['gpu_reserved_seconds'] == 17.
    with pytest.raises(ValueError):
        q.complete('j1', 0, True, True)                                 # failed jobs are never re-run in place
    assert q.claim(0, res) is None
    assert q.export_jsonl(tmp_path / 'jobs.jsonl') == 3


def test_wisig_wrapper_matches_native_logits_and_feature_layer():
    from rffi_calibration.encoders import build_wisig_official
    torch.manual_seed(0)
    w = build_wisig_official(classes=10)
    x = torch.randn(7, 1, 256, 2)
    z = w.encode(x)
    assert z.shape == (7, 80) and (z >= 0).all()
    np.testing.assert_allclose(w.native_logits(x).detach().numpy(), w.model.eval()(x).detach().numpy(), atol=1e-6)
    W, b = w.linear_head()
    np.testing.assert_allclose((z.double() @ W.T + b).detach().numpy(), w.model(x).double().detach().numpy(), atol=1e-5)
    with pytest.raises(ValueError):
        w.encode(torch.randn(2, 1, 2, 256))


def test_shen_wrapper_uses_the_128d_preclassifier_and_refuses_silent_class_change():
    from rffi_calibration.encoders import build_shen_featurecnn
    torch.manual_seed(0)
    w = build_shen_featurecnn()
    x = torch.randn(3, 1, 52, 126)
    z = w.encode(x)
    assert z.shape == (3, 128) and (z >= 0).all()
    np.testing.assert_allclose(w.native_logits(x).detach().numpy(), w.model(x).detach().numpy(), atol=1e-6)
    W, b = w.linear_head()
    np.testing.assert_allclose((z.double() @ W.T + b).detach().numpy(), w.model(x).double().detach().numpy(), atol=1e-5)
    assert w.historical_512d(x).shape == (3, 512)
    with pytest.raises(ValueError, match='new_head'):
        build_shen_featurecnn(num_classes=25)
    w25 = build_shen_featurecnn(state_dict=w.model.state_dict(), num_classes=25, new_head=True)
    assert w25.native_logits(x).shape == (3, 25)
    np.testing.assert_allclose(w25.encode(x).detach().numpy(), z.detach().numpy(), atol=1e-6)  # trunk unchanged


def test_osu_window_plan_respects_blocks(reg):
    from rffi_calibration.datasets.osu import plan_windows
    blocks = reg.roles['osu']['blocks']
    s, e, roles = plan_windows(20_000_000, blocks)
    assert set(roles) == {'fit', 'support', 'query'}
    for role, (a, b) in (('fit', blocks['fit']), ('support', blocks['support']), ('query', blocks['query'])):
        m = roles == role
        assert s[m].min() >= a * 2e7 + blocks['guard_samples'] and e[m].max() <= b * 2e7 - blocks['guard_samples']
    assert s[roles == 'query'].min() - e[roles == 'support'].max() >= 0.05 * 2e7
