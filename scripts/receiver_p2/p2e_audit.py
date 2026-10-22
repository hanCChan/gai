#!/usr/bin/env python3
"""P2E-1: error composition and receiver-shift learnability on unseen RX. Inference + oracle
linear-head refits only; no backbone training, no sealed data (Shen 10 registered RX only).

Checkpoints: search (trained on fold inner_train = 4 RX; inner_validation RX is truly unseen = "held")
and formal (trained on all 5 source RX). Targets: held RX (search only) and the 5 outer RX.

Per (checkpoint, target RX q), on the 128-d pre-head features h and the frozen linear head, accuracy
after successive corrections of q's features (statistics from all q packets, accuracy on the odd
half of each class so the oracle head has an untouched test half):
  raw      frozen model
  ca       label-free RX centering  h - mean_q(h) + mean_S(h)          (deployable with unlabeled q)
  cc       oracle class-conditional mean fix  h - mu_{y,q} + mu_y^S    (first moment removed exactly)
  ccproj   cc restricted to the source shift span: h - P_{U_y} D_{y,q}, U_t = span{D_{t,r}: r in S}
  ccrand   same, random subspace of equal dim inside the source feature PCA span (control)
  ccw      cc + class-conditional covariance whiten-recolor in the source PCA subspace (2nd moment)
  oracle   linear head refit on q's own even half (linear separability left in the features)
Shift learnability: cov = ||P_U D_{t,q}||^2/||D_{t,q}||^2 (feature and centred-logit space) vs random,
and mag = ||D_{t,q}|| / rms_r ||D_{t,r}|| (extrapolation factor).
Source-only predictors per checkpoint: in-domain val acc, per-RX val CE variance (REx quantity),
worst source-RX val acc, kappa, closest-competitor q10, within-source LORO coverage.

    python3 p2e_audit.py --out reports/receiver_p2e/audit [--shard 0/4]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'receiver_p0')]
import p0_common as C   # noqa: E402
from audit_p2 import forward   # noqa: E402

C.add_root()
GAI = Path('/workspace/rl1/gai')
PHASES = ('receiver_p1', 'receiver_p2', 'receiver_p2b', 'receiver_p2d')


def model_list(phases=PHASES, anchors=True):
    import torch
    runs = {}
    for j in (C.jobs('shen') if anchors else []):
        if j['method'] == 'ce':
            C.load_model(j['ckpt'])   # hash guard
            runs[('formal', j['fold'], 'ce', 0., j['seed'])] = Path(j['ckpt']) / 'selected.pt'
    for ph in phases:
        for d in sorted((GAI / 'outputs' / ph).glob('*/shen_rank16_CV*_seed*')):
            if not (d / 'status.json').exists():
                continue
            j = torch.load(d / 'selected.pt', map_location='cpu', weights_only=False)['job']
            if j['stage'] not in ('search', 'formal') or j.get('epochs', 20) != 20:
                continue   # P2E loro fits have a different fitting set; 40-epoch runs audited separately
            arm = j['method'] + ('' if j.get('set', 'Rq') == 'Rq' else '_' + j['set'])
            runs.setdefault((j['stage'], j['fold'], arm, float(j['lam']), j['seed']), d / 'selected.pt')
    return [dict(stage=k[0], fold=k[1], arm=k[2], lam=k[3], seed=k[4], path=v) for k, v in sorted(runs.items())]


def cmeans(h, y, rx, rs):
    return np.stack([np.stack([h[(y == t) & (rx == r)].mean(0) for r in rs]) for t in range(10)])


def basis(M, tol=1e-6):
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[S > tol * max(S.max(), 1e-30)].T


def msqrt(S, inv=False):
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-10 * w.max(), None)
    return (V * (w ** (-.5 if inv else .5))) @ V.T


def shrink(S, a=0.1):
    k = S.shape[0]
    return (1 - a) * S + a * np.trace(S) / k * np.eye(k)


def acc(z, y):
    return float((z.argmax(1) == y).mean())


def target_metrics(ht, yt, hs, ys, rxs, W, b, V, rng, device):
    """All corrections for one target RX. hs/ys/rxs = source train features of the fitting RX."""
    import torch
    rs = np.unique(rxs)
    mu_sr = cmeans(hs, ys, rxs, rs)                    # (10, R, d)
    muS = mu_sr.mean(1)                                # equal-RX class mean
    Dsrc = mu_sr - muS[:, None]
    muq = np.stack([ht[yt == t].mean(0) for t in range(10)])
    Dq = muq - muS
    odd = np.zeros(len(yt), bool)
    for t in range(10):
        ix = np.flatnonzero(yt == t)
        odd[ix[1::2]] = True
    lin = lambda h: h @ W.T + b
    out = {}
    out['acc_raw'] = acc(lin(ht[odd]), yt[odd])
    out['acc_ca'] = acc(lin(ht - ht.mean(0) + hs.mean(0))[odd], yt[odd])
    out['acc_cc'] = acc(lin(ht - Dq[yt])[odd], yt[odd])
    A = W - W.mean(0, keepdims=True)
    hp, hr = ht.copy(), ht.copy()
    cf, cr, cz, mg = [], [], [], []
    for t in range(10):
        U = basis(Dsrc[t])
        R = np.linalg.qr(V @ rng.standard_normal((V.shape[1], U.shape[1])))[0]
        m = yt == t
        hp[m] -= Dq[t] @ U @ U.T
        hr[m] -= Dq[t] @ R @ R.T
        n2 = (Dq[t] ** 2).sum() + 1e-12
        cf.append(((Dq[t] @ U) ** 2).sum() / n2)
        cr.append(((Dq[t] @ R) ** 2).sum() / n2)
        Uz = basis(Dsrc[t] @ A.T)
        dz = Dq[t] @ A.T
        cz.append(((dz @ Uz) ** 2).sum() / ((dz ** 2).sum() + 1e-12))
        mg.append(np.sqrt(n2 / ((Dsrc[t] ** 2).sum(1).mean() + 1e-12)))
    out['acc_ccproj'] = acc(lin(hp)[odd], yt[odd])
    out['acc_ccrand'] = acc(lin(hr)[odd], yt[odd])
    out.update(cov_feat=float(np.mean(cf)), cov_rand=float(np.mean(cr)), cov_logit=float(np.mean(cz)),
               mag=float(np.mean(mg)), mag_max=float(np.max(mg)), dim_U=int(basis(Dsrc[0]).shape[1]))
    # second moment: whiten-recolor to the pooled source class-t distribution inside span(V)
    hw = ht.copy()
    for t in range(10):
        m, ms = yt == t, ys == t
        xt, xs = ht[m] @ V, hs[ms] @ V
        B = msqrt(shrink(np.atleast_2d(np.cov(xs.T)))) @ msqrt(shrink(np.atleast_2d(np.cov(xt.T))), inv=True)
        inner = (xt - xt.mean(0)) @ B.T + xs.mean(0)
        comp_t = ht[m] - (ht[m] @ V) @ V.T
        comp_s = hs[ms] - (hs[ms] @ V) @ V.T
        hw[m] = inner @ V.T + comp_t - comp_t.mean(0) + comp_s.mean(0)
    out['acc_ccw'] = acc(lin(hw)[odd], yt[odd])
    head, _ = C.refit_head(lambda: torch.nn.Linear(ht.shape[1], 10), ht[~odd], yt[~odd], device)
    with torch.no_grad():
        zo = head(torch.as_tensor(ht[odd], dtype=torch.float32, device=device)).cpu().numpy()
    out['acc_oracle'] = acc(zo, yt[odd])
    return out


def source_metrics(hs, zs, ys, rxs, hv, zv, yv, rxv):
    rs = np.unique(rxs)
    zc = zs - zs.mean(1, keepdims=True)
    mz = cmeans(zc, ys, rxs, rs)
    AD = mz - mz.mean(1, keepdims=True)
    AC = mz.mean(1) - mz.mean(1).mean(0)
    kappa = float((AD ** 2).sum(-1).mean() / (AC ** 2).sum(-1).mean())
    lsm = zv - zv.max(1, keepdims=True)
    lsm = lsm - np.log(np.exp(lsm).sum(1, keepdims=True))
    ce = -lsm[np.arange(len(yv)), yv]
    per_ce = [ce[rxv == r].mean() for r in rs]
    per_acc = [acc(zv[rxv == r], yv[rxv == r]) for r in rs]
    q10 = []
    for t in range(10):
        m = yv == t
        q10.append(min(np.quantile(zv[m, t] - zv[m, u], .1) for u in range(10) if u != t))
    mu = cmeans(hs, ys, rxs, rs)
    loro = []
    for i in range(len(rs)):
        keep = [j for j in range(len(rs)) if j != i]
        for t in range(10):
            ref = mu[t, keep].mean(0)
            U = basis(mu[t, keep] - ref)
            d = mu[t, i] - ref
            loro.append(((d @ U) ** 2).sum() / ((d ** 2).sum() + 1e-12))
    return dict(src_val_acc=float(np.mean(per_acc)), src_val_worst=float(np.min(per_acc)),
                src_ce_var=float(np.var(per_ce)), src_val_ce=float(np.mean(per_ce)), kappa=kappa,
                src_q10=float(np.mean(q10)), loro_cov=float(np.mean(loro)))


def main():
    import torch
    from b1_baselines.model import FeatureCNN
    from receiver_r01.data import shen as load_shen
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--phases', default=','.join(PHASES))
    ap.add_argument('--no-anchors', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    si, sn = map(int, a.shard.split('/'))
    reg = C.registration()
    device = torch.device('cuda:0')
    runs = model_list(tuple(a.phases.split(',')), not a.no_anchors)[si::sn]
    rows, mods, cache = [], [], {}
    for k, ru in enumerate(runs):
        obj = torch.load(ru['path'], map_location='cpu', weights_only=False)
        sc = obj['scaler']
        f = reg['shen_folds'][ru['fold']]
        fit = f['inner_train'] if ru['stage'] == 'search' else f['inner_train'] + f['inner_validation']
        key = (ru['fold'], ru['stage'], round(sc['mean'], 12), round(sc['std'], 12))
        if key not in cache:
            tg = {'held': f['inner_validation']} if ru['stage'] == 'search' else {}
            tg.update({'outer': f['outer_evaluation']})
            cache = {key: dict(src=load_shen(reg, fit, sc, 'train')[0], val=load_shen(reg, fit, sc, 'val')[0],
                               tg={kind: load_shen(reg, keys, sc, 'all')[0] for kind, keys in tg.items()})}
        D = cache[key]
        model = FeatureCNN()
        model.load_state_dict(obj['model'])
        model = model.to(device).eval()
        W = model.classifier.weight.detach().double().cpu().numpy()
        b = model.classifier.bias.detach().double().cpu().numpy()
        hs, zs = forward(model, D['src']['x'], device)
        hv, zv = forward(model, D['val']['x'], device)
        ys, rxs = D['src']['y'], D['src']['rx']
        ev = np.linalg.svd(hs - hs.mean(0), compute_uv=False) ** 2
        kv = max(2, min(32, int(np.searchsorted(np.cumsum(ev) / ev.sum(), .995)) + 1))
        V = np.linalg.svd(hs - hs.mean(0), full_matrices=False)[2][:kv].T
        rng = np.random.default_rng(20261022 + ru['fold'])
        base = dict(stage=ru['stage'], fold=ru['fold'], arm=ru['arm'], lam=ru['lam'], seed=ru['seed'])
        sm = source_metrics(hs, zs, ys, rxs, hv, zv, D['val']['y'], D['val']['rx'])
        mrow = dict(base, pca_k=kv, **sm)
        for kind, td in D['tg'].items():
            ht, _ = forward(model, td['x'], device)
            accs = []
            for r in np.unique(td['rx']):
                m = td['rx'] == r
                tm = target_metrics(ht[m], td['y'][m], hs, ys, rxs, W, b, V, rng, device)
                rows.append(dict(base, kind=kind, rx=reg['shen_allowed'][int(r)], **tm))
                accs.append(tm['acc_raw'])
            mrow[f'{kind}_acc'] = float(np.mean(accs))
        mods.append(mrow)
        print(k + 1, len(runs), ru['stage'], ru['fold'], ru['arm'], ru['lam'], ru['seed'],
              'outer=%.4f' % mrow['outer_acc'], flush=True)
        tag = '' if sn == 1 else f'_{si}of{sn}'
        pd.DataFrame(rows).to_csv(out / f'targets{tag}.csv', index=False)
        pd.DataFrame(mods).to_csv(out / f'models{tag}.csv', index=False)


if __name__ == '__main__':
    main()
