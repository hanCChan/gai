#!/usr/bin/env python3
"""P2E-FS: few-shot labelled receiver calibration with an eigen-receiver prior. Inference only.

After p2e_eigenrx.py showed that UNLABELED adaptation is not identifiable on WiSig (EM from 56%-correct
posteriors cannot attribute class-dependent shifts), test the minimal-supervision setting: the new
receiver records m labelled packets per transmitter (re-enrolment), m in {1, 2, 5, 10}.

Same Gaussian model as EigenRX (source PCA space, pooled within-(class,RX) covariance):
  lda       source means (no calibration)
  proto     m-shot class means shrunk to the source mean:  mu_t + m/(m+2) (gbar_t - mu_t)
  span      per-class source-span prior: in-span part of the shift kept (m/(m+0.5)), out-of-span part
            shrunk (m/(m+8)); constants fixed before any run
  eig       eigen-receiver: a = argmin sum_i ||g_i - mu_t - E_t a||_Sigma^-1 + ||a||^2 / tau  (J numbers)
  eig+res   eig, then the per-class residual shrunk like proto  (prior first, data second)
  headft    frozen features, linear head re-fitted on source packets + the m-shot packets (x20 weight)
Accuracy on the target packets NOT used for calibration; 5 draws per m.

    python3 p2e_fewshot.py --dataset shen|wisig --out reports/receiver_p2e/fewshot
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'receiver_p0')]
import p0_common as C   # noqa: E402
from p2e_eigenrx import EigenRX, TAU   # noqa: E402

C.add_root()
GAI = Path('/workspace/rl1/gai')
SHOTS = (1, 2, 5, 10)


def calibrate(er, g, y, cal, mode, m):
    K, J = er.K, er.E.shape[2]
    gbar = np.stack([g[cal & (y == t)].mean(0) for t in range(K)])
    if mode == 'proto':
        return er.mu + m / (m + 2.) * (gbar - er.mu)
    if mode == 'span':
        # per-class source-span prior (REPORT §8, G2/G3): trust the in-span part of the m-shot shift,
        # shrink the out-of-span part hard; kappas fixed a priori (0.5 / 8), never tuned on targets
        means = er.mu.copy()
        for t in range(K):
            Q = np.linalg.qr(er.E[t])[0]
            d = gbar[t] - er.mu[t]
            din = d @ Q @ Q.T
            means[t] = er.mu[t] + m / (m + .5) * din + m / (m + 8.) * (d - din)
        return means
    A = sum(m * er.E[t].T @ er.Si @ er.E[t] for t in range(K)) + np.eye(J) / TAU
    rhs = sum(m * er.E[t].T @ er.Si @ (gbar[t] - er.mu[t]) for t in range(K))
    means = er.mu + np.einsum('tkj,j->tk', er.E, np.linalg.solve(A, rhs))
    if mode == 'eig+res':
        means = means + m / (m + 2.) * (gbar - means)
    return means


def head_ft(hs, ys, ht, yt, cal, K, device):
    import torch
    X = np.concatenate([hs] + [ht[cal]] * 20)
    Y = np.concatenate([ys] + [yt[cal]] * 20)
    head, _ = C.refit_head(lambda: torch.nn.Linear(hs.shape[1], K), X, Y, device)
    with torch.no_grad():
        return head(torch.as_tensor(ht, dtype=torch.float32, device=device)).cpu().numpy().argmax(1)


def evaluate_target(er, hs, ys, ht, yt, W, b, rng, device):
    K = W.shape[0]
    if any((yt == t).sum() < max(SHOTS) + 4 for t in range(K)):
        return []
    g = er.proj(ht)
    raw = ht @ W.T + b
    rows = []
    for m in SHOTS:
        res = {k: [] for k in ('raw', 'lda', 'proto', 'span', 'eig', 'eig+res', 'headft')}
        for _ in range(5):
            cal = np.zeros(len(yt), bool)
            for t in range(K):
                cal[rng.choice(np.flatnonzero(yt == t), m, replace=False)] = True
            te = ~cal
            acc = lambda p: float((p[te] == yt[te]).mean())
            res['raw'].append(acc(raw.argmax(1)))
            res['lda'].append(acc(er.loglik(g, er.mu, np.zeros(K)).argmax(1)))
            for mode in ('proto', 'span', 'eig', 'eig+res'):
                res[mode].append(acc(er.loglik(g, calibrate(er, g, yt, cal, mode, m), np.zeros(K)).argmax(1)))
            if m in (1, 5):
                res['headft'].append(acc(head_ft(hs, ys, ht, yt, cal, K, device)))
        rows.append(dict(shots=m, **{f'acc_{k}': float(np.mean(v)) if v else np.nan for k, v in res.items()}))
    return rows


def run(dataset, out, device):
    import torch
    reg = C.registration()
    rows = []
    if dataset == 'shen':
        from b1_baselines.model import FeatureCNN
        from receiver_r01.data import shen as load_shen
        from audit_p2 import forward
        cache = {}
        for d in sorted((GAI / 'outputs/receiver_p2e/formal').glob('shen_rank16_CV*_formal_seed*')):
            obj = torch.load(d / 'selected.pt', map_location='cpu', weights_only=False)
            j = obj['job']
            if j['arm'] not in ('ce', 'ccalign') or j.get('epochs', 20) != 20:
                continue
            f = reg['shen_folds'][j['fold']]
            key = (j['fold'], round(obj['scaler']['mean'], 12))
            if key not in cache:
                cache = {key: (load_shen(reg, f['inner_train'] + f['inner_validation'], obj['scaler'], 'train')[0],
                               load_shen(reg, f['outer_evaluation'], obj['scaler'], 'all')[0])}
            src, tg = cache[key]
            model = FeatureCNN()
            model.load_state_dict(obj['model'])
            model = model.to(device).eval()
            W = model.classifier.weight.detach().double().cpu().numpy()
            b = model.classifier.bias.detach().double().cpu().numpy()
            hs, _ = forward(model, src['x'], device)
            ht, _ = forward(model, tg['x'], device)
            er = EigenRX(hs, src['y'], src['rx'], 10)
            rng = np.random.default_rng(j['seed'])
            for r in np.unique(tg['rx']):
                mm = tg['rx'] == r
                for row in evaluate_target(er, hs, src['y'], ht[mm], tg['y'][mm], W, b, rng, device):
                    rows.append(dict(dataset='shen', fold=j['fold'], arm=j['arm'], seed=j['seed'],
                                     rx=reg['shen_allowed'][int(r)], J=er.E.shape[2], **row))
            print(d.name, flush=True)
            pd.DataFrame(rows).to_csv(out / 'shen_fewshot.csv', index=False)
    else:
        from receiver_r01.data import wisig as load_wisig
        from p2e_generality import wisig_features, head_of
        names = reg['wisig']['rx_names']
        roles = reg['wisig']['roles']
        src = load_wisig(reg, roles['source'], [0, 1], [0, 120])
        tg = load_wisig(reg, roles['internal_rx'], [0, 1], [0, 200])
        for j in C.jobs('wisig'):
            if j['method'] not in ('ce', 'swad', 'mldg_rx'):
                continue
            obj, _ = C.load_model(j['ckpt'])
            model, _ = C.build_model(dict(dataset='wisig', backbone=j['backbone']), obj)
            model = model.to(device).eval()
            W, b = head_of(model, j['backbone'])
            hs, _ = wisig_features(model, src['x'], device, j['backbone'])
            ht, _ = wisig_features(model, tg['x'], device, j['backbone'])
            er = EigenRX(hs, src['y'], src['rx'], 10)
            rng = np.random.default_rng(j['seed'])
            for r in np.unique(tg['rx']):
                mm = tg['rx'] == r
                for row in evaluate_target(er, hs, src['y'], ht[mm], tg['y'][mm], W, b, rng, device):
                    rows.append(dict(dataset='wisig', run_id=j['run_id'], backbone=j['backbone'], arm=j['method'],
                                     seed=j['seed'], rx=names[int(r)], J=er.E.shape[2], **row))
            print(j['run_id'], flush=True)
            pd.DataFrame(rows).to_csv(out / 'wisig_fewshot.csv', index=False)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['shen', 'wisig'], required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run(a.dataset, out, torch.device('cuda:0'))


if __name__ == '__main__':
    main()
