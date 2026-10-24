#!/usr/bin/env python3
"""Encoder training for T0-T3 on the shared receiver-episode schedule (EXPERIMENT_PROTOCOL.md §6).

Gated: a real run requires an ACTIVE reservation in configs/rffi_calibration/registration.json that
lists the job id; numeric data are opened only through AccessGuard. `--smoke` runs 3 steps on random
tensors (no data access) to check the code path, and never writes a checkpoint into outputs/.

Recipe (inherited from the historical WiSig/Shen trainers): Adam, grad-norm clip 5, OfficialCNN L2
`regularization()` on WiSig, checkpoint = minimum receiver-equal CE on held POSITIONS of the FIT RX
(WiSig positions 120:160 of dates 0-1, Shen last 40 rows per TX); source_selection RX never enter.

    python scripts/rffi_calibration/train.py --job configs/rffi_calibration/jobs/<job>.json
    python scripts/rffi_calibration/train.py --smoke --dataset wisig --arm T3
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from rffi_calibration.access import AccessGuard  # noqa: E402
from rffi_calibration.encoders import build_shen_featurecnn, build_wisig_official  # noqa: E402
from rffi_calibration.episodes import EpisodeSchedule, episode_loss  # noqa: E402
from rffi_calibration.registry import Registry, file_sha  # noqa: E402
from rffi_calibration.statistics import StatsConfig  # noqa: E402


def ccalign_fn():
    import importlib.util
    spec = importlib.util.spec_from_file_location('p2_losses_direct', ROOT / 'scripts/receiver_p2/p2_losses.py')
    L = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(L)
    return L.ccalign


def load_data(reg, guard, job):
    ds = job['dataset']
    if ds == 'wisig':
        from rffi_calibration.datasets import wisig
        fit = reg.roles['wisig']['folds'][job['split']]['fit']
        tr, x = wisig.load(reg, guard, fit, [0, 1], (0, 120), numeric=True, purpose=job['job_id'])
        va, xv = wisig.load(reg, guard, fit, [0, 1], (120, 160), numeric=True, purpose=job['job_id'])
        return tr, x, va, xv, None
    if ds == 'shen':
        from rffi_calibration.datasets import shen
        fit = job['fit_rx']
        excluded = set(job['excluded_rx'])
        if excluded & set(fit):
            raise ValueError('fit RX overlap the excluded (selection/outer) RX')
        tr, x = shen.load(reg, guard, fit, 'fit', numeric=True, scaler=dict(mean=0., std=1.), purpose=job['job_id'])
        scaler = shen.fit_scaler(x)
        x = ((x - scaler['mean']) / scaler['std']).astype(np.float32)
        va, xv = shen.load(reg, guard, fit, 'val', numeric=True, scaler=scaler, purpose=job['job_id'])
        return tr, x, va, xv, scaler
    raise ValueError(f'training is defined for wisig/shen; {ds} uses frozen transfer only')


def smoke_data(dataset, rng):
    T, R = 10, 4
    shape = (1, 256, 2) if dataset == 'wisig' else (1, 52, 126)
    n = 40
    y = np.repeat(np.tile(np.arange(T), R), n)
    rx = np.repeat(np.repeat(np.arange(R), T), n)
    x = rng.standard_normal((len(y), *shape)).astype(np.float32)

    class Tab:
        pass
    tr = Tab()
    tr.tx, tr.rx = y, rx
    va = Tab()
    va.tx, va.rx = y[::10], rx[::10]
    return tr, x, va, x[::10], None


def val_ce(wrapper, xv, yv, rxv, device, batch=1024):
    wrapper.eval()
    ces = []
    with torch.no_grad():
        for r in np.unique(rxv):
            m = np.flatnonzero(rxv == r)
            lg = torch.cat([wrapper.native_logits(torch.as_tensor(xv[m[i:i + batch]], device=device)).float().cpu()
                            for i in range(0, len(m), batch)])
            ces.append(float(F.cross_entropy(lg, torch.as_tensor(yv[m]))))
    return float(np.mean(ces))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--dataset', default='wisig')
    ap.add_argument('--arm', default='T0')
    a = ap.parse_args()
    reg = Registry()
    device = torch.device('cuda:0' if torch.cuda.is_available() and not a.smoke else 'cpu')
    if a.smoke:
        job = dict(job_id='smoke', dataset=a.dataset, arm=a.arm, seed=0, steps=3, eval_every=2, lr=5e-4,
                   episode_weight=1.0, ccalign_lam=1.0, temperature=1.0, n_ref=4, n_query=2)
        tr, x, va, xv, scaler = smoke_data(a.dataset, np.random.default_rng(0))
        out = Path('/tmp/rffi_calibration_smoke') / f'{a.dataset}_{a.arm}'
        out.mkdir(parents=True, exist_ok=True)
    else:
        matrix = json.loads((ROOT / 'configs/rffi_calibration/formal_matrix.json').read_text())
        found = [j for j in matrix['jobs'] if j['job_id'] == a.job]
        if len(found) != 1:
            sys.exit(f'job {a.job} is not registered in formal_matrix.json')
        job = found[0]
        res = reg.reservation()
        if not res.get('active') or job['job_id'] not in res.get('allowed_job_ids', []):
            sys.exit(f'job {job["job_id"]} is not in an active reservation; nothing was run')
        guard = AccessGuard(reg, res, log_path=ROOT / 'reports/rffi_calibration/access_log.jsonl', stage=res['stage'])
        if 'RFFI_QUEUE' not in os.environ:
            sys.exit('real training must be claimed by run_stage_a.py; nothing was read')
        from rffi_calibration.queue import JobQueue
        claim = JobQueue(os.environ['RFFI_QUEUE']).get(job['job_id'])
        if not claim or claim['status'] != 'running':
            sys.exit('job has no running queue claim; nothing was read')
        tr, x, va, xv, scaler = load_data(reg, guard, job)
        out = ROOT / 'outputs/rffi_calibration' / job['job_id']
        out.mkdir(parents=True, exist_ok=False)
    T = reg.num_classes(job['dataset'])
    rx_codes = np.unique(tr.rx, return_inverse=True)[1]
    rxv_codes = np.unique(va.rx, return_inverse=True)[1]
    torch.manual_seed(job['seed'])
    np.random.seed(job['seed'])
    wrapper = (build_wisig_official(classes=T) if job['dataset'] == 'wisig' else build_shen_featurecnn()).to(device)
    opt = torch.optim.Adam(wrapper.parameters(), lr=job['lr'], eps=1e-7 if job['dataset'] == 'wisig' else 1e-8)
    sched = EpisodeSchedule(tr.tx, rx_codes, T, seed=job['seed'], n_ref=job.get('n_ref', 16),
                            n_query=job.get('n_query', 8))
    aux = ccalign_fn() if job['arm'] == 'T1' else None
    weight = job['ccalign_lam'] if job['arm'] == 'T1' else job.get('episode_weight', 1.0)
    xt = torch.as_tensor(x)
    best, curve, t0 = dict(ce=float('inf'), step=-1), [], time.time()
    for step in range(job['steps']):
        ep = sched.episode(step)
        idx = ep.all_idx()
        wrapper.train()
        opt.zero_grad(set_to_none=True)
        z, logits = wrapper.train_forward(xt[idx].to(device))
        loss, info = episode_loss(job['arm'], z.double(), tr.tx, ep, rx_codes, T, logits.float(),
                                  cfg=StatsConfig(), temperature=job.get('temperature', 1.0), weight=weight,
                                  aux_fn=aux)
        loss = loss + wrapper.regularization()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapper.parameters(), 5., error_if_nonfinite=True)
        opt.step()
        if (step + 1) % job['eval_every'] == 0 or step + 1 == job['steps']:
            v = val_ce(wrapper, xv, va.tx, rxv_codes, device)
            curve.append(dict(step=step + 1, val_ce=v, **info))
            print(json.dumps(dict(job=job['job_id'], step=step + 1, val_ce=v,
                                  seconds=round(time.time() - t0, 1))), flush=True)
            if v < best['ce']:
                best = dict(ce=v, step=step + 1)
                torch.save(dict(model=wrapper.model.state_dict(), scaler=scaler, job=job, step=step + 1, val_ce=v),
                           out / 'selected.pt')
    status = dict(state='completed', job=job, best=best, seconds=time.time() - t0,
                  checkpoint_sha256=file_sha(out / 'selected.pt'),
                  registry_fingerprint=reg.fingerprint(),
                  code_sha={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted((ROOT / 'src/rffi_calibration').glob('*.py'))})
    (out / 'curve.json').write_text(json.dumps(curve, indent=1))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    status['peak_memory_mb'] = torch.cuda.max_memory_allocated() / 2**20 if device.type == 'cuda' else 0.
    status['device'] = str(device)
    (out / 'status.json').write_text(json.dumps(status, indent=1, default=str))
    print(json.dumps(dict(job=job['job_id'], best=best, seconds=round(status['seconds'], 1))))


if __name__ == '__main__':
    main()
