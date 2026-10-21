#!/usr/bin/env python3
"""P1 trainer: R0/R1 Shen recipe verbatim + one auxiliary loss on a frozen pair set.

methods: ce | ls (label smoothing 0.1, generic-regulariser control) | drift (P0 V0, bitwise path)
         | var (A2 kappa*Var_r Delta) | margin (A3 RX-agnostic normalized margin risk)
         | tail (A4 worst-source-RX margin risk) | tail_drift (A5 = tail lam + drift lam2)
stages:  search  train inner_train RX (first160), select on inner_validation RX (all packets),
                 pair set Rq_inner, calib 'inner'  -> lambda selection only, never outer data
         formal  train all 5 source RX (first160), checkpoint by last-40 receiver-equal val CE,
                 pair set --set (Rq | rand), calib 'formal'
Recipe: torch.manual_seed(seed) -> FeatureCNN, Adam(3e-4, eps 1e-8), balanced_schedule(seed),
clip 5, 20 epochs, deterministic. Per-epoch diagnostics on the first batch (CE, aux, separate
gradient norms, mean ||w_t-w_u|| over P) are read-only and do not alter the trajectory.

    python3 train_p1.py --stage formal --fold 0 --method tail --lam 1 --seed 20261013 \
        --sets sets_p1.json --out OUT [--set Rq]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import p0_common as C   # noqa: E402
import p1_losses as L   # noqa: E402

C.add_root()


def main():
    import torch
    from torch.nn import functional as F
    from b1_baselines.model import FeatureCNN
    from receiver_r01.core import balanced_schedule, predict, metrics, regularization
    from receiver_r01.data import shen as load_shen, groups

    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['search', 'formal'], required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--method', choices=['ce', 'ls', 'drift', 'var', 'margin', 'tail', 'tail_drift'], required=True)
    ap.add_argument('--lam', type=float, default=0.)
    ap.add_argument('--lam2', type=float, default=0.)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--set', default='Rq', choices=['Rq', 'rand'])
    ap.add_argument('--sets', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--gamma-drift', type=float, default=0.5)
    ap.add_argument('--tau-r', type=float, default=0.5)
    a = ap.parse_args()

    tag = f'{a.method}{a.lam:g}' + (f'_{a.lam2:g}' if a.method == 'tail_drift' else '')
    run_id = f'shen_rank16_CV{a.fold}_{tag}_{a.set}_{a.stage}_seed{a.seed}'
    out = Path(a.out) / run_id
    assert not out.exists(), out
    out.mkdir(parents=True)
    reg = C.registration()
    fs = json.loads(Path(a.sets).read_text())['folds'][str(a.fold)]
    if a.stage == 'search':
        assert a.set == 'Rq'
        P = [tuple(p) for p in fs['Rq_inner']]
        cal = fs['calib']['inner']
    else:
        P = [tuple(p) for p in fs[a.set]]
        cal = fs['calib']['formal']
    gamma_m, tau_m = cal['gamma'], cal['tau_m']

    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    model = FeatureCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-8)

    fold = reg['shen_folds'][a.fold]
    if a.stage == 'search':
        train, stats = load_shen(reg, fold['inner_train'], None, 'train')
        val, _ = load_shen(reg, fold['inner_validation'], stats, 'all')
    else:
        keys = fold['inner_train'] + fold['inner_validation']
        train, stats = load_shen(reg, keys, None, 'train')
        val, _ = load_shen(reg, keys, stats, 'val')
    schedule = balanced_schedule(groups(train), a.epochs, a.seed)
    x = torch.as_tensor(train['x'], device=device)
    y = torch.as_tensor(train['y'], device=device)
    rx = torch.as_tensor(train['rx'], device=device)
    vx = torch.as_tensor(val['x'], device=device)
    vy = torch.as_tensor(val['y'], device=device)
    vrx = torch.as_tensor(val['rx'], device=device)

    def feat512(xb):
        h = F.relu(model.stem(F.pad(xb, (2, 3, 2, 3))))
        h = model.blocks(h).permute(0, 2, 3, 1).reshape(xb.shape[0], -1)
        return F.normalize(model.feature(h), p=2, dim=1, eps=1e-12)

    def losses(xb, yb, rb):
        """(ce, aux) with aux already weighted; ce/ls/drift keep the exact P0 op order."""
        W = model.classifier.weight
        if a.method == 'ce':
            return F.cross_entropy(model(xb), yb, reduction='none').mean() + regularization(model), None
        if a.method == 'ls':
            return F.cross_entropy(model(xb), yb, label_smoothing=0.1), None
        if a.method == 'drift':          # P0 path: logits from model(xb), second feature pass
            logits = model(xb)
            pre = F.relu(model.tx_input(feat512(xb)))
            return F.cross_entropy(logits, yb), a.lam * L.drift_v0(pre, W, yb, rb, P, a.gamma_drift)
        pre = F.relu(model.tx_input(feat512(xb)))
        logits = model.classifier(pre)
        ce = F.cross_entropy(logits, yb)
        if a.method == 'var':
            aux = a.lam * L.proj_var(pre, W, yb, rb, P, a.gamma_drift)
        elif a.method == 'margin':
            aux = a.lam * L.margin_risk(logits, W, yb, rb, P, gamma_m, tau_m, None)
        elif a.method == 'tail':
            aux = a.lam * L.margin_risk(logits, W, yb, rb, P, gamma_m, tau_m, a.tau_r)
        else:
            aux = (a.lam * L.margin_risk(logits, W, yb, rb, P, gamma_m, tau_m, a.tau_r)
                   + a.lam2 * L.drift_v0(pre, W, yb, rb, P, a.gamma_drift))
        return ce, aux

    params = list(model.parameters())

    def gnorm(v):
        g = torch.autograd.grad(v, params, retain_graph=True, allow_unused=True)
        return float(torch.sqrt(sum((q.float() ** 2).sum() for q in g if q is not None)))

    history = []
    best, best_state, best_epoch = float('inf'), None, 0
    t0 = time.perf_counter()
    for epoch in range(a.epochs):
        for bi, ids in enumerate(schedule[epoch]):
            ix = torch.tensor(ids, device=device)
            xb, yb, rb = x[ix], y[ix], rx[ix]
            model.train()
            opt.zero_grad(set_to_none=True)
            ce, aux = losses(xb, yb, rb)
            loss = ce if aux is None else ce + aux
            assert torch.isfinite(loss)
            if bi == 0:
                W = model.classifier.weight.detach()
                diag = dict(epoch=epoch + 1, ce=float(ce), aux=float(aux) if aux is not None else 0.,
                            g_ce=gnorm(ce), g_aux=gnorm(aux) if aux is not None and aux.requires_grad else 0.,
                            w_tu=float(np.mean([float((W[t] - W[u]).norm()) for t, u in P])))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.step()
        model.eval()
        ms = []
        with torch.no_grad():
            for r in torch.unique(vrx):
                m = vrx == r
                ms.append(metrics(predict(model, vx[m]).cpu().numpy(), vy[m].cpu().numpy()))
        vce = float(np.mean([q['CE'] for q in ms]))
        vacc = float(np.mean([q['accuracy'] for q in ms]))
        if vce < best:
            best, best_epoch, best_acc = vce, epoch + 1, vacc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(dict(diag, val_CE=vce, val_acc=vacc))
        C.dump_json(out / 'history.json', history)
    model.load_state_dict(best_state)
    job = dict(vars(a), run_id=run_id, P=[list(p) for p in P], gamma_m=gamma_m, tau_m=tau_m)
    torch.save(dict(model=model.state_dict(), job=job, scaler=stats, selection=dict(epoch=best_epoch)), out / 'selected.pt')
    C.dump_json(out / 'status.json', dict(state='completed', run_id=run_id, seconds=time.perf_counter() - t0,
                                          best_epoch=best_epoch, best_val_CE=best, best_val_acc=best_acc,
                                          checkpoint_sha256=C.sha(out / 'selected.pt')))
    print(json.dumps(dict(run_id=run_id, best_epoch=best_epoch, val_CE=round(best, 4), val_acc=round(best_acc, 4),
                          sec=round(time.perf_counter() - t0))), flush=True)


if __name__ == '__main__':
    main()
