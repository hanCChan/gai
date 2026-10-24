#!/usr/bin/env python3
"""R0 single fit (C_MATCH / P_MIX / P_RX) on WiSig, plan §7. Gated by configs/rffi_post_a.

Real run: only when started by run_pilot.py (RFFI_POST_QUEUE env + a running queue claim) under an
ACTIVE R0 reservation listing the job. `--smoke` uses random tensors on CPU, never reads data and
writes only under /tmp.

Primary checkpoint last.pt (fixed horizon); secondary native_best.pt (min receiver-equal native CE on
fit-RX positions 120:160). Held RX are never loaded.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from rffi_calibration.registry import Registry, file_sha  # noqa: E402
from rffi_post_a.episode_controls import ARMS, ControlSchedule, grad_probe, pilot_loss  # noqa: E402

CONFIG = ROOT / 'configs/rffi_post_a'


def state_sha(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_job(job_id, config=None):
    matrix = json.loads(((config or CONFIG) / 'pilot_matrix.json').read_text())
    found = [j for j in matrix['jobs'] if j['job_id'] == job_id]
    if len(found) != 1:
        sys.exit(f'{job_id} is not registered in pilot_matrix.json; nothing was read')
    return dict(found[0], **matrix['recipe'], dataset=matrix['dataset'], num_classes=matrix['num_classes'])


def smoke_data(rng, T=10, R=10, n=24):
    tx, rx, date = [], [], []
    for t in range(T):
        for r in range(R):
            for d in (0, 1):
                tx += [t] * n
                rx += [r] * n
                date += [d] * n
    x = rng.standard_normal((len(tx), 1, 256, 2)).astype(np.float32)
    tx, rx, date = np.array(tx), np.array(rx), np.array(date)
    return (tx, rx, date, np.array([f'smoke|{i}' for i in range(len(tx))]), x), (tx[::6], rx[::6], x[::6])


def real_data(reg, job, report_root=None):
    from rffi_calibration.access import AccessGuard
    from rffi_calibration.datasets import wisig
    res = reg.reservation()
    guard = AccessGuard(reg, res, log_path=(report_root or ROOT / 'reports/rffi_post_a/R0') / 'access_log.jsonl',
                        stage=res['stage'])
    fold = reg.roles['wisig']['folds'][job['split']]
    fit = fold['fit']
    if set(fit) & set(fold['source_selection']):
        raise ValueError('fit RX overlap held RX')
    tr, x = wisig.load(reg, guard, fit, job['fit_dates'], tuple(job['fit_positions']), numeric=True, purpose=job['job_id'])
    va, xv = wisig.load(reg, guard, fit, job['fit_dates'], tuple(job['val_positions']), numeric=True, purpose=job['job_id'])
    return (tr.tx, tr.rx, tr.date, tr.ids, x), (va.tx, va.rx, xv)


def val_ce(wrapper, xv, yv, rxv, device, batch=2048):
    wrapper.eval()
    out = []
    with torch.no_grad():
        for r in np.unique(rxv):
            m = np.flatnonzero(rxv == r)
            lg = torch.cat([wrapper.native_logits(torch.as_tensor(xv[m[i:i + batch]], device=device)).float().cpu()
                            for i in range(0, len(m), batch)])
            out.append(float(F.cross_entropy(lg, torch.as_tensor(yv[m]))))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--arm', default='P_RX', choices=ARMS)
    ap.add_argument('--steps', type=int, default=6)
    ap.add_argument('--stage', choices=('R0', 'RV1'), default='R0')
    a = ap.parse_args()
    config, report_root, fit_root = CONFIG, ROOT / 'reports/rffi_post_a/R0', ROOT / 'outputs/rffi_post_a'
    if a.stage == 'RV1':
        from rffi_post_a.variance import CONFIG as variance_config, OUT, FITS, check_reservation
        config, report_root, fit_root = variance_config, OUT, FITS
    reg = Registry(config)
    if a.smoke:
        job = dict(job_id=f'smoke_{a.arm}', split='smoke', arm=a.arm, seed=0, lr=5e-4, eps=1e-7, clip_norm=5.,
                   budgets_train=[1, 5], n_ref_per_date=8, n_query=8, episode_weight=1., temperature=1.,
                   steps=a.steps, eval_every=3, grad_probe_every=2, num_classes=10)
        (tx, rx, date, ids, x), (vy, vrx, xv) = smoke_data(np.random.default_rng(0))
        device = torch.device('cpu')
        out = Path('/tmp/rffi_post_a_smoke') / job['job_id']
        out.mkdir(parents=True, exist_ok=True)
    else:
        job = load_job(a.job, config)
        res = reg.reservation()
        if a.stage == 'RV1':
            check_reservation(res, json.loads((config / 'pilot_matrix.json').read_text()))
        if not (res.get('active') and res.get('stage') == a.stage and job['job_id'] in res.get('allowed_job_ids', [])):
            sys.exit(f'{job["job_id"]} is not in an active {a.stage} reservation; nothing was read')
        if 'RFFI_POST_QUEUE' not in os.environ:
            sys.exit('R0 fits must be claimed by run_pilot.py; nothing was read')
        from rffi_calibration.queue import JobQueue
        claim = JobQueue(os.environ['RFFI_POST_QUEUE']).get(job['job_id'])
        if not claim or claim['status'] != 'running' or claim['stage'] != a.stage:
            sys.exit('no running queue claim; nothing was read')
        if a.stage == 'RV1' and not torch.cuda.is_available():
            sys.exit('RV1 requires the registered A100 device; nothing was read')
        (tx, rx, date, ids, x), (vy, vrx, xv) = real_data(reg, job, report_root)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        out = fit_root / job['job_id']
        out.mkdir(parents=True, exist_ok=False)
    T = job['num_classes']
    torch.manual_seed(job['seed'])
    from rffi_calibration.encoders import build_wisig_official
    wrapper = build_wisig_official(classes=T).to(device)
    init_sha = state_sha(wrapper.model)
    opt = torch.optim.Adam(wrapper.parameters(), lr=job['lr'], eps=job['eps'])
    sched = ControlSchedule(tx, rx, date, T, seed=job['seed'] + 1, budgets=job['budgets_train'],
                            n_query=job['n_query'], n_ref_per_date=job['n_ref_per_date'])
    xt = torch.as_tensor(x)
    ids_chain = hashlib.sha256()
    exposure = np.zeros(len(tx), np.int64)
    log, curve, clipped = [], [], 0
    best = dict(ce=float('inf'), step=-1)
    t0 = time.time()
    for step in range(job['steps']):
        ep = sched.episode(step)
        ids_chain.update(ep.ids_sha(ids).encode())
        np.add.at(exposure, ep.all_idx, 1)
        wrapper.train()
        opt.zero_grad(set_to_none=True)
        z, logits = wrapper.train_forward(xt[ep.all_idx].to(device))
        loss, parts = pilot_loss(job['arm'], z, logits, tx[ep.all_idx], ep, T, job['episode_weight'], job['temperature'])
        loss = loss + wrapper.regularization()
        probe = None
        if 'proto' in parts and (step + 1) % job['grad_probe_every'] == 0:
            probe = grad_probe(wrapper.parameters(), parts['native'], parts['proto'])
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(wrapper.parameters(), job['clip_norm'], error_if_nonfinite=True))
        clipped += gnorm > job['clip_norm']
        opt.step()
        if (step + 1) % job['eval_every'] == 0 or step + 1 == job['steps']:
            v = val_ce(wrapper, xv, vy, vrx, device)
            fz = z.detach().norm(dim=1)
            rec = dict(step=step + 1, val_ce=v, native=float(parts['native']), grad_norm_preclip=gnorm,
                       clip_fraction=clipped / (step + 1), feat_norm_q50=float(fz.median()),
                       feat_norm_q90=float(torch.quantile(fz, .9)), head_weight_norm=float(wrapper.model.head.weight.norm()),
                       pseudo_target=ep.pseudo_target, m=ep.m, seconds=time.time() - t0)
            if 'proto' in parts:
                rec.update(proto=float(parts['proto']), proto_acc=float(parts['proto_acc']))
            if probe:
                rec.update(probe)
            curve.append(rec)
            if v < best['ce']:
                best = dict(ce=v, step=step + 1)
                torch.save(dict(model=wrapper.model.state_dict(), job=job, step=step + 1, val_ce=v), out / 'native_best.pt')
        elif probe:
            log.append(dict(step=step + 1, **probe))
    torch.save(dict(model=wrapper.model.state_dict(), job=job, step=job['steps']), out / 'last.pt')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    status = dict(state='completed', job=job, init_state_sha256=init_sha, all_ids_chain_sha256=ids_chain.hexdigest(),
                  best_native=best, seconds=time.time() - t0, last_sha256=file_sha(out / 'last.pt'),
                  native_best_sha256=file_sha(out / 'native_best.pt'), registry_fingerprint=reg.fingerprint(),
                  records_seen=int((exposure > 0).sum()), record_exposure_max=int(exposure.max()),
                  peak_memory_mb=torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == 'cuda' else 0.,
                  device=str(device),
                  code_sha={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / 'src/rffi_post_a').glob('*.py'))})
    (out / 'curve.json').write_text(json.dumps(curve, indent=1))
    (out / 'probes.json').write_text(json.dumps(log, indent=1))
    cells = Counter()
    for r_, d_, t_, e_ in zip(np.asarray(rx).astype(str), date, tx, exposure):
        cells[f'{r_}|{int(d_)}|{int(t_)}'] += int(e_)
    (out / 'exposure.json').write_text(json.dumps(dict(sorted(cells.items()))))
    (out / 'status.json').write_text(json.dumps(status, indent=1, default=str))
    print(json.dumps(dict(job=job['job_id'], init=init_sha[:12], ids=status['all_ids_chain_sha256'][:12],
                          best=best, seconds=round(status['seconds'], 1))))


if __name__ == '__main__':
    main()
