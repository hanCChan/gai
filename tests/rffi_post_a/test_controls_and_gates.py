"""R0 controls (sampler, ProtoNet, arm equivalences), gates, scheduler, independent replay, D0 helpers,
synthetic D1 end-to-end. Synthetic data only; nothing here opens real data, features or weights."""
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import ROOT, load_by_path
from rffi_post_a.artifact_guard import ArtifactGuard, LineageError, cache_key, verify_lineage
from rffi_post_a.episode_controls import ControlSchedule, grad_probe, make_control_episode, pilot_loss, proto_logits

DIAG = json.loads((ROOT / 'configs/rffi_post_a/diagnostic_matrix.json').read_text())
PILOT = json.loads((ROOT / 'configs/rffi_post_a/pilot_matrix.json').read_text())
REG = json.loads((ROOT / 'configs/rffi_post_a/registration.json').read_text())


def meta(T=10, R=10, n=20):
    tx, rx, date = [], [], []
    for t in range(T):
        for r in range(R):
            for d in (0, 1):
                tx += [t] * n
                rx += [r] * n
                date += [d] * n
    tx, rx, date = np.array(tx), np.array(rx), np.array(date)
    return dict(tx=tx, rx=rx, date=date, num_classes=T, ids=np.array([f'w|{i}' for i in range(len(tx))]))


# ------------------------------------------------------------------------------------- sampler
def test_cycle_covers_every_rx_budget_once_and_is_deterministic():
    md = meta()
    s = ControlSchedule(md['tx'], md['rx'], md['date'], 10, seed=5)
    combos = [(s.episode(k).pseudo_target, s.episode(k).m) for k in range(20)]
    assert sorted(combos) == sorted((str(r), m) for r in range(10) for m in (1, 5))
    e1, e2 = s.episode(7), ControlSchedule(md['tx'], md['rx'], md['date'], 10, seed=5).episode(7)
    assert (e1.all_idx == e2.all_idx).all() and (e1.mix_support == e2.mix_support).all()


@pytest.mark.parametrize('step', [0, 3, 11])
def test_routing_counts_dates_disjointness_and_mix_composition(step):
    md = meta()
    ep = make_control_episode(md, step, 9)
    a = ep.all_idx
    assert len(set(md['ids'][a])) == len(a)                                   # unique sample IDs
    for s_, q_ in ((ep.rx_support, ep.rx_query), (ep.mix_support, ep.mix_query)):
        assert len(s_) == 10 * ep.m and len(q_) == 10 * 8
        assert (md['date'][a[s_]] == 0).all() and (md['date'][a[q_]] == 1).all()
        assert not set(a[s_]) & set(a[q_])
        assert (np.bincount(md['tx'][a[s_]], minlength=10) == ep.m).all()
    assert set(md['rx'][a[ep.rx_support]].astype(str)) == {ep.pseudo_target}
    for t in range(10):                                                         # MIX: distinct RX per class
        assert len(set(ep.mix_rx_query[t])) == 8 and len(set(ep.mix_rx_support[t])) == ep.m
    assert len({tuple(x) for x in ep.mix_rx_query}) > 1                         # not stuck on one RX set


def test_insufficient_coverage_or_rx_count_fails_loudly():
    md = meta(R=6)
    with pytest.raises(ValueError, match='distinct fit RX'):
        ControlSchedule(md['tx'], md['rx'], md['date'], 10, seed=0)
    md = meta(n=4)
    with pytest.raises(ValueError, match='required counts'):
        ControlSchedule(md['tx'], md['rx'], md['date'], 10, seed=0)


# ------------------------------------------------------------------------------------ ProtoNet
def test_proto_logits_match_the_author_implementation():
    """jakesnell/prototypical-networks few_shot.py: z_proto = mean support, dists = euclidean_dist
    (squared), log_p = log_softmax(-dists)."""
    torch.manual_seed(0)
    zs, ys = torch.randn(6, 5), torch.tensor([0, 0, 1, 1, 2, 2])
    zq = torch.randn(4, 5)
    proto = zs.view(3, 2, 5).mean(1)

    def euclidean_dist(x, y):                                                   # verbatim author helper
        n, m, d = x.size(0), y.size(0), x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return torch.pow(x - y, 2).sum(2)
    ref = torch.log_softmax(-euclidean_dist(zq, proto), 1)
    np.testing.assert_allclose(torch.log_softmax(proto_logits(zs, ys, zq, 3), 1).numpy(), ref.numpy(), atol=1e-6)


def _model_step(arm, weight, md, ep, seed=0):
    from rffi_calibration.encoders import build_wisig_official
    torch.manual_seed(seed)
    w = build_wisig_official(classes=10)
    init = copy.deepcopy(w.state_dict())
    opt = torch.optim.Adam(w.parameters(), lr=5e-4, eps=1e-7)
    x = torch.as_tensor(np.random.default_rng(1).standard_normal((len(md['tx']), 1, 256, 2)).astype(np.float32))
    torch.manual_seed(123)                                                      # dropout stream
    w.train()
    z, logits = w.train_forward(x[ep.all_idx])
    rng_after = torch.get_rng_state().clone()
    loss, parts = pilot_loss(arm, z, logits, md['tx'][ep.all_idx], ep, 10, weight)
    (loss + w.regularization()).backward()
    sup_grad = None
    torch.nn.utils.clip_grad_norm_(w.parameters(), 5.)
    opt.step()
    return w, init, rng_after, parts, z


def test_same_init_and_dropout_stream_across_arms_and_weight0_equals_c_match():
    md = meta(n=10)
    ep = ControlSchedule(md['tx'], md['rx'], md['date'], 10, seed=1, n_ref_per_date=8).episode(0)
    runs = {arm: _model_step(arm, 1.0, md, ep) for arm in ('C_MATCH', 'P_MIX', 'P_RX')}
    inits = [r[1] for r in runs.values()]
    for k in inits[0]:
        assert all(torch.equal(inits[0][k], i[k]) for i in inits[1:])
    rngs = [r[2] for r in runs.values()]
    assert all(torch.equal(rngs[0], r) for r in rngs[1:])
    natives = [float(r[3]['native']) for r in runs.values()]
    assert natives[0] == natives[1] == natives[2]
    w0, *_ = _model_step('P_RX', 0.0, md, ep)
    wc = runs['C_MATCH'][0]
    for (k, a), (_, b) in zip(w0.state_dict().items(), wc.state_dict().items()):
        assert torch.allclose(a, b, atol=0, rtol=0), k
    wp = runs['P_RX'][0]
    assert any(not torch.equal(a, b) for a, b in zip(wp.state_dict().values(), wc.state_dict().values()))


def test_support_receives_gradient_and_query_labels_only_in_loss():
    md = meta(n=10)
    ep = ControlSchedule(md['tx'], md['rx'], md['date'], 10, seed=2, n_ref_per_date=8).episode(1)
    z = torch.randn(len(ep.all_idx), 6, requires_grad=True)
    lg = torch.zeros(len(ep.all_idx), 10, requires_grad=True)
    loss, parts = pilot_loss('P_RX', z, lg, md['tx'][ep.all_idx], ep, 10)
    loss.backward()
    assert z.grad[ep.rx_support].abs().sum() > 0 and z.grad[ep.rx_query].abs().sum() > 0
    # permuting query labels changes the loss (supervision) but not the logits of the query
    y2 = md['tx'][ep.all_idx].copy()
    y2[ep.rx_query] = np.random.default_rng(0).permutation(y2[ep.rx_query])
    l2, p2 = pilot_loss('P_RX', z.detach(), lg.detach(), y2, ep, 10)
    s = torch.as_tensor(ep.rx_support)
    q = torch.as_tensor(ep.rx_query)
    np.testing.assert_array_equal(proto_logits(z.detach()[s], torch.as_tensor(md['tx'][ep.all_idx])[s], z.detach()[q], 10).numpy(),
                                  proto_logits(z.detach()[s], torch.as_tensor(y2)[s], z.detach()[q], 10).numpy())
    assert float(l2) != float(loss)


def test_grad_probe_does_not_touch_backward_gradients():
    lin = torch.nn.Linear(4, 3)
    x = torch.randn(8, 4)
    la, lb = lin(x).pow(2).mean(), lin(x).sum()
    probe = grad_probe(lin.parameters(), la, lb)
    assert lin.weight.grad is None and -1 <= probe['cosine'] <= 1
    (la + lb).backward()
    ref = torch.nn.Linear(4, 3)
    ref.load_state_dict(lin.state_dict())
    (ref(x).pow(2).mean() + ref(x).sum()).backward()
    assert torch.allclose(lin.weight.grad, ref.weight.grad)


def test_encoders_have_no_batchnorm():
    from rffi_calibration.encoders import build_shen_featurecnn, build_wisig_official
    for w in (build_wisig_official(classes=10), build_shen_featurecnn()):
        assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in w.modules())


# --------------------------------------------------------------------------------- lineage gate
TASK = DIAG['tasks'][0]
ACTIVE = dict(REG['requests_not_active']['D1'], active=True)
ROLES = json.loads((ROOT / 'configs/rffi_post_a/roles.json').read_text())
MANS = {rx: p + '0' * 48 for rx, p in TASK['manifest_sha256_prefix'].items()}


def test_lineage_refuses_before_opening_anything():
    ck = dict(checkpoint_sha256=TASK['checkpoint_sha256'])
    assert verify_lineage(ck, None, MANS, ACTIVE, ROLES, TASK)
    for res, msg in ((dict(ACTIVE, active=False), 'inactive'), (dict(ACTIVE, stage='R0'), 'stage'),
                     (dict(ACTIVE, allow_feature_extraction=False), 'extraction'), (dict(ACTIVE, allowed_job_ids=[]), 'allowlist')):
        with pytest.raises(LineageError, match=msg):
            verify_lineage(ck, None, MANS, res, ROLES, TASK)
    with pytest.raises(LineageError, match='checkpoint SHA'):
        verify_lineage(dict(checkpoint_sha256='0' * 64), None, MANS, ACTIVE, ROLES, TASK)
    with pytest.raises(LineageError, match='manifest'):
        verify_lineage(ck, None, dict(MANS, **{'1': 'f' * 64}), ACTIVE, ROLES, TASK)
    for bad_rx in (29, 3, 99):                                                   # final, development, unknown
        t = dict(TASK, held_rx=[bad_rx])
        with pytest.raises(LineageError):
            verify_lineage(ck, None, MANS, ACTIVE, ROLES, t)
    with pytest.raises(LineageError, match='held RX inside'):
        verify_lineage(ck, None, MANS, ACTIVE, ROLES, dict(TASK, held_rx=[2]))


def test_cache_forgery_symlink_and_write_once(tmp_path):
    g = ArtifactGuard(tmp_path / 'cache')
    m = dict(checkpoint_sha256=TASK['checkpoint_sha256'], encoder_layer='x', scaler_sha256='none', source_asset_sha256='s',
             dataset='wisig', rx=['2', '5'], dates=[0, 1], positions=[0, 120], manifest_sha256=MANS, code_sha256='c',
             dtype='float64', role='source')
    g.write('t_source', m, z=np.zeros((2, 3)), ids=np.array(['a', 'b']))
    with pytest.raises(LineageError, match='write-once'):
        g.write('t_source', m, z=np.zeros((2, 3)))
    meta_, arr = g.open_features('t_source', TASK, ACTIVE, MANS, ROLES)
    assert arr['z'].dtype == np.float64
    p = tmp_path / 'cache/t_source.meta.json'
    forged = json.loads(p.read_text())
    forged['rx'] = ['2', '5', '1']                                               # add a held RX, keep old key
    p.write_text(json.dumps(forged))
    with pytest.raises(LineageError, match='key'):
        g.open_features('t_source', TASK, ACTIVE, MANS, ROLES)
    forged['key'] = cache_key(forged)                                            # recomputed key: role check fires
    p.write_text(json.dumps(forged))
    with pytest.raises(LineageError, match='outside'):
        g.open_features('t_source', TASK, ACTIVE, MANS, ROLES)
    (tmp_path / 'elsewhere').mkdir()
    (tmp_path / 'cache/esc.npz').symlink_to(tmp_path / 'elsewhere/x.npz')
    with pytest.raises(LineageError, match='outside the cache root'):
        g.path('esc.npz')
    with pytest.raises(LineageError):
        g.path('../x.npz')


# ---------------------------------------------------------------------------- configs & gates
def test_new_configs_are_inactive_and_consistent():
    r = REG['reservation']
    assert r['active'] is False
    assert not r['allow_final_numeric_access'] and not r['allow_development_numeric_access']
    assert len(PILOT['jobs']) == 6 and {j['arm'] for j in PILOT['jobs']} == {'C_MATCH', 'P_MIX', 'P_RX'}
    assert set(REG['requests_not_active']['R0']['allowed_job_ids']) == {j['job_id'] for j in PILOT['jobs']}
    assert {t['task'] for t in DIAG['tasks']} == set(REG['requests_not_active']['D1']['allowed_job_ids'])
    assert sum(len(t['held_rx']) for t in DIAG['tasks']) == 12
    for f in ('datasets.json', 'roles.json'):
        assert (ROOT / 'configs/rffi_post_a' / f).read_bytes() == (ROOT / 'configs/rffi_calibration' / f).read_bytes()
    stage_a = json.loads((ROOT / 'configs/rffi_calibration/registration.json').read_text())['reservation']
    assert stage_a['active'] is False                                            # Stage A stays closed


def test_run_pilot_reservation_check_refuses_everything_but_the_exact_request():
    rp = load_by_path('run_pilot_t', 'scripts/rffi_post_a/run_pilot.py')
    good = dict(REG['requests_not_active']['R0'], active=True)
    rp.check_reservation(good, PILOT)
    for bad in (dict(good, active=False), dict(good, stage='D1'), dict(good, max_fits=7),
                dict(good, max_reserved_gpu_hours=2.0), dict(good, allow_failed_fit_retry=True),
                dict(good, allowed_job_ids=good['allowed_job_ids'][:5]),
                dict(good, allowed_job_ids=good['allowed_job_ids'] + ['A_wisig_w0_T0_s20261013']),
                dict(good, allow_final_numeric_access=True)):
        with pytest.raises(SystemExit):
            rp.check_reservation(bad, PILOT)


@pytest.mark.parametrize('script,args', [('scripts/rffi_post_a/train_pilot.py', ['--job', 'R0_wisig_w0_P_RX_s20261024']),
                                         ('scripts/rffi_post_a/cache_features.py', []),
                                         ('scripts/rffi_post_a/diagnose.py', ['--out', 'reports/rffi_post_a/_never'])])
def test_real_entry_points_exit_before_reading_when_inactive(script, args, monkeypatch, tmp_path):
    mod = load_by_path(Path(script).stem + '_gate', script)
    before = set((ROOT / 'outputs/rffi_post_a').glob('**/*'))
    monkeypatch.setattr(sys, 'argv', [script] + args)
    with pytest.raises(SystemExit) as e:
        mod.main()
    assert 'nothing was read' in str(e.value)
    assert set((ROOT / 'outputs/rffi_post_a').glob('**/*')) == before


def test_train_pilot_smoke_runs_three_arms_with_identical_ids(monkeypatch):
    mod = load_by_path('train_pilot_smoke', 'scripts/rffi_post_a/train_pilot.py')
    seen = {}
    for arm in ('C_MATCH', 'P_MIX', 'P_RX'):
        monkeypatch.setattr(sys, 'argv', ['x', '--smoke', '--arm', arm, '--steps', '4'])
        mod.main()
        st = json.loads(Path(f'/tmp/rffi_post_a_smoke/smoke_{arm}/status.json').read_text())
        seen[arm] = (st['init_state_sha256'], st['all_ids_chain_sha256'])
    assert len(set(seen.values())) == 1


# ------------------------------------------------------------------- scheduler (JobQueue reuse)
def test_queue_exactly_once_and_no_retry(tmp_path):
    from rffi_calibration.queue import BudgetExceeded, JobQueue
    q = JobQueue(tmp_path / 'q.sqlite')
    for j in PILOT['jobs']:
        q.add(job_id=j['job_id'], stage='R0', predicted_gpu_seconds=j['predicted_gpu_seconds'])
    res = dict(REG['requests_not_active']['R0'], active=True)
    got = []
    while (c := q.claim(0, res)) is not None:
        got.append(c['job_id'])
        q.complete(c['job_id'], 1 if len(got) == 1 else 0, True, True, gpu_reserved_seconds=300)
    assert sorted(got) == sorted(j['job_id'] for j in PILOT['jobs'])
    assert q.claim(0, res) is None                                               # the failed one is not retried
    q2 = JobQueue(tmp_path / 'q2.sqlite')
    q2.add(job_id='x', stage='R0', predicted_gpu_seconds=6000)
    with pytest.raises(BudgetExceeded):
        q2.claim(0, dict(res, allowed_job_ids=['x']))


# ------------------------------------------------------------------- independent replay/decision
def _fake_r0(tmp_path, gain=0.04, mix_gap=0.02):
    vr = load_by_path('verify_t', 'scripts/rffi_post_a/verify_results.py')
    rng = np.random.default_rng(0)
    held = {'w0': ['1', '7', '16', '30'], 'w1': ['6', '26', '2', '5']}
    base = {'C_MATCH': .55, 'P_MIX': .55 + gain - mix_gap, 'P_RX': .55 + gain}
    for split, rxs in held.items():
        for arm, acc in base.items():
            rows, sc = [], {}
            for rx in rxs:
                y = np.repeat(np.arange(10), 20)
                sc[f'{rx}|y_query'] = y
                for head, off in (('H1e', 0.), ('H1', -.01), ('H2L', -.02), ('H5a', -.01), ('H5b', -.01), ('H5c[k0=1]', -.03), ('H0', -.2)):
                    for m in ((0,) if head == 'H0' else (1, 2, 5, 10)):
                        for k in range(1 if m == 0 else 3):
                            p = np.clip(acc + off + .01 * m, 0, 1)
                            correct = rng.random(len(y)) < p
                            s = rng.standard_normal((len(y), 10))
                            s[np.arange(len(y)), np.where(correct, y, (y + 1) % 10)] += 10
                            sc[f'{rx}|{head}|{m}|{k}'] = s
                            rows.append(dict(rx=rx, head=head, m=m, draw=k, acc=float((s.argmax(1) == y).mean()),
                                             arm=arm, split=split, checkpoint='last.pt', job_id=f'R0_{split}_{arm}'))
            d = tmp_path / 'evaluation' / f'R0_wisig_{split}_{arm}_last'
            d.mkdir(parents=True)
            with open(d / 'rows.csv', 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            np.savez(d / 'scores.npz', **sc)
    return vr


def test_independent_r0_replay_and_decision(tmp_path):
    vr = _fake_r0(tmp_path / 'a', gain=.06, mix_gap=.03)
    res = vr.verify_r0(tmp_path / 'a')
    assert res['all_ok'] and res['decision']['decision'] in ('GO_DESIGN_SHEN', 'EPISODIC_GAIN_RX_UNPROVEN')
    vr = _fake_r0(tmp_path / 'b', gain=0., mix_gap=0.)
    assert vr.verify_r0(tmp_path / 'b')['decision']['decision'] == 'NO_GAIN_IN_FIXED_PILOT'
    # tampering with a stored accuracy is caught by the score replay
    p = next((tmp_path / 'a/evaluation').glob('*/rows.csv'))
    rows = list(csv.DictReader(open(p)))
    rows[0]['acc'] = str(float(rows[0]['acc']) + .05)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    assert not vr.verify_r0(tmp_path / 'a')['all_ok']


def test_synthetic_d1_end_to_end_with_anchor_and_score_replay(tmp_path):
    dg = load_by_path('diagnose_t', 'scripts/rffi_post_a/diagnose.py')
    vr = load_by_path('verify_t2', 'scripts/rffi_post_a/verify_results.py')
    for task, src, tgt in dg.synthetic_tasks():
        rows, res = dg.diagnose_task(task, src, tgt, DIAG, 10, tmp_path, n_draws=2)
        for rx, t in res['targets'].items():
            assert all(v['ok'] for v in t['anchors'].values()), (task['task'], rx, t['anchors'])
        assert 'loro' in res['stability'] and res['panel']['d'] > 0
        assert ('date_decomposition' in res['stability']) == (task['dataset'] == 'wisig')
    v = vr.verify_d1(tmp_path)
    assert v['all_ok'] and len(v['score_replay']) == 3


# ------------------------------------------------------------------------------------- D0 helpers
def test_d0_replay_and_curve_audit_on_toy_inputs(tmp_path):
    from rffi_post_a import reporting as R
    rows = [dict(rx=str(r), head=h, m=1, draw=k, acc=a + .01 * r, macro_f1=.5, nll=1., brier=.5, ece=.1,
                 split='w0', recall_0=.5) for r in range(3) for k in range(2) for h, a in (('H4', .4), ('H1', .5))]
    ref = tmp_path / 'agg.csv'
    with open(ref, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['head', 'm', 'accuracy_pct', 'worst_rx_pct'])
        w.writerow(['H4', 1, 100 * .41, 100 * .40])
        w.writerow(['H1', 1, 100 * .51, 100 * .50])
    assert R.replay_aggregate(rows, ref)['ok']
    t = R.per_rx_table(rows, 'toy')
    fd = {(d['a'], d['b'], d['m']): d for d in R.fixed_differences(t, budgets=(1,))}
    assert abs(fd[('H4', 'H1', 1)]['mean_pp'] + 10) < 1e-9 and fd[('H4', 'H1', 1)]['n_rx'] == 3
    curve = [dict(step=s, val_ce=1. / s) for s in range(10, 110, 10)]
    st = dict(job=dict(job_id='j', steps=100, eval_every=10), best=dict(step=100, ce=.01))
    ca = R.curve_audit(curve, st)
    assert ca['best_is_final'] and ca['tail_slope_per_100_steps'] < 0
    R.svg_lines({'a': [(1, 50), (5, 60)]}, 't', tmp_path / 'f.svg')
    assert (tmp_path / 'f.svg').read_text().startswith('<svg')
