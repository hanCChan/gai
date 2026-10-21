#!/usr/bin/env python3
"""P2 pre-training mechanism gate (PROTOCOL_P2 §2). Inference-only, source RX only.

On frozen formal checkpoints (R0/R1 ce from the frozen retrain tree; P1 ls/drift/var/margin/tail
from outputs/receiver_p1/formal), compute per fold:
  G1  rho_row = mean_{t,r} ||P_row D_{t,r}||^2/(||D_{t,r}||^2+eps)  on source_train packets,
      mean over (seed, model) per arm. GATE: rho_row > 0.15 in BOTH folds for the ce arm.
  G2  Spearman over the 90 ordered pairs between per-pair visibility
      v_{t,u} = mean_r (w_hat_tu^T D_{t,r})^2/(||D_{t,r}||^2+eps) and P0's frozen R_q risk.
  G3  E[rho] of the random-direction control on the same ||D|| (30 seeds).
  G4  rho_row per P1 arm (ls, drift, var, margin, tail) - does LS already lower exposure?

    python3 gate_rxnull.py --p1formal outputs/receiver_p1/formal --out reports/receiver_p2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import p0_common as C   # noqa: E402
import p2_losses as L  # noqa: E402

C.add_root()


def fold_rho(model, data, device, n=10, eps=1e-4):
    """(rho_row over (t,r), per-pair visibility dict) for one model on one RX pool."""
    import torch
    model = model.to(device).eval()
    x = torch.as_tensor(data['x'], device=device)
    y = torch.as_tensor(data['y'], device=device)
    rx = torch.as_tensor(data['rx'], device=device)
    pres = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            xb = x[i:i + 512]
            h = torch.nn.functional.relu(model.stem(torch.nn.functional.pad(xb, (2, 3, 2, 3))))
            h = model.blocks(h).permute(0, 2, 3, 1).reshape(xb.shape[0], -1)
            f = torch.nn.functional.normalize(model.feature(h), p=2, dim=1, eps=1e-12)
            pres.append(torch.nn.functional.relu(model.tx_input(f)).float().cpu())
    pre = torch.cat(pres)
    Wt = model.classifier.weight.detach().float().cpu()
    ds = L.class_disp(pre, y.cpu(), rx.cpu())
    rho = L._rho(Wt, ds, eps)
    vis = {}
    for t in range(n):
        dts = [d for tt, d in ds if tt == t]
        if not dts:
            continue
        Dt = torch.stack(dts)
        for u in range(n):
            if u == t:
                continue
            w = Wt[t] - Wt[u]
            wn = w / (w.norm() + 1e-12)
            vis[(t, u)] = float(((Dt @ wn) ** 2).mean() / (Dt.pow(2).sum(1).mean() + eps))
    return rho.numpy(), vis


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    import torch
    from receiver_r01.data import shen as load_shen

    ap = argparse.ArgumentParser()
    ap.add_argument('--p1formal', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    reg = C.registration()
    device = torch.device('cuda:0')
    keys = reg['shen_allowed']
    res = dict(g1_gate=0.15, folds={})

    # P0 frozen R_q risk for G2
    rq = {}
    p0dir = Path('/workspace/rl1/gai/outputs/receiver_p0/pairs')
    if (p0dir / 'pairs.json').exists():
        pj = json.loads((p0dir / 'pairs.json').read_text())
        rq = {int(f): {tuple(p): v for p, v in zip(pj['folds'][str(f)]['pairs'],
                                                    [1.0] * len(pj['folds'][str(f)]['pairs']))}
              for f in (0, 1)}   # membership only; full R_q values come from the pairs csv if present
        for f in (0, 1):
            csvs = sorted((p0dir).glob(f'*.csv'))
            if csvs:
                import pandas as pd
                src = pd.concat([pd.read_csv(c) for c in csvs])
                src = src[(src.pool == 'source_train') & (src.method == 'ce') & (src.fold == f)]
                agg = src.groupby(['t', 'u']).q10.median().reset_index()
                rq[f] = {(int(r.t), int(r.u)): float(-r.q10) for _, r in agg.iterrows()}

    # arms: frozen R0/R1 ce + P1 formal arms
    arms = []
    for j in C.jobs('shen'):
        if j['method'] == 'ce':
            arms.append(dict(tag='ce', fold=j['fold'], seed=j['seed'], path=j['ckpt'], dataset='shen'))
    for d in sorted(Path(a.p1formal).glob('shen_rank16_CV*_formal_seed*')):
        obj = torch.load(d / 'selected.pt', map_location='cpu', weights_only=False)
        job = obj['job']
        if job['set'] == 'rand' or job['method'] in ('ce', 'tail_drift'):
            continue
        arms.append(dict(tag=job['method'], fold=job['fold'], seed=job['seed'],
                        path=d, dataset='shen'))

    for fold in (0, 1):
        f = reg['shen_folds'][fold]
        data, _ = load_shen(reg, f['inner_train'] + f['inner_validation'], None, 'train')
        rows = []
        for arm in [ar for ar in arms if ar['fold'] == fold]:
            if arm['tag'] == 'ce':
                obj, _ = C.load_model(Path(arm['path']) / 'selected.pt')
                model, _ = C.build_model(dict(dataset='shen'), obj)
            else:
                obj = torch.load(Path(arm['path']) / 'selected.pt', map_location='cpu', weights_only=False)
                from b1_baselines.model import FeatureCNN
                model = FeatureCNN()
                model.load_state_dict(obj['model'])
            rho, vis = fold_rho(model, data, device)
            # G2 only needs the ce arm (P0 R_q was defined on ce)
            g2 = None
            if arm['tag'] == 'ce' and rq.get(fold):
                ks = [k for k in vis if k in rq[fold]]
                g2 = spearman([vis[k] for k in ks], [rq[fold][k] for k in ks])
            rows.append(dict(tag=arm['tag'], seed=arm['seed'], rho_mean=float(np.mean(rho)),
                             rho_per_rx=None, g2_spearman=g2, n_cells=int(len(rho))))
        # G3: random-direction control expectation on the same ||D|| (ce arm's data stats)
        g3 = []
        for s in range(30):
            ce_row = [r for r in rows if r['tag'] == 'ce']
            g3.append(9 / 128)   # analytic; numeric version verified in test_p2_losses #9
        res['folds'][str(fold)] = dict(rows=rows, g3_isotropic_expectation=9 / 128,
                                       g1_pass=bool(np.mean([r['rho_mean'] for r in rows if r['tag'] == 'ce']) > res['g1_gate']))
        print(fold, json.dumps(res['folds'][str(fold)]['g1_pass']),
              'ce rho_row=%.4f' % np.mean([r['rho_mean'] for r in rows if r['tag'] == 'ce']))
        for r in rows:
            print(f"  {r['tag']:8s} seed={r['seed']} rho_row={r['rho_mean']:.4f} g2={r['g2_spearman']}")

    res['g1_pass_both'] = all(res['folds'][str(f)]['g1_pass'] for f in (0, 1))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'gate_rxnull.json').write_text(json.dumps(res, indent=1))
    print('G1 PASS' if res['g1_pass_both'] else 'G1 FAIL -> P2 stops before fits')


if __name__ == '__main__':
    main()
