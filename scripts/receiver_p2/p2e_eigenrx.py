#!/usr/bin/env python3
"""P2E-ER: eigen-receiver adaptation (unsupervised receiver calibration). Inference only.

Setting: a new receiver supplies N UNLABELED packets before identification. Motivation (REPORT §8): on
both Shen and WiSig the recoverable unseen-RX error is a class-conditional mean shift that is mostly
inside the span of source-receiver shifts, and class-agnostic centring recovers little of it.

Model (eigenvoice-style, Kuhn et al. 2000): in the source-feature PCA space, packets of class t on
receiver q ~ N(mu_t + E_t a_q, Sigma), with mu_t the source class means, Sigma the pooled within-(class,RX)
covariance, and E = [E_1; ...; E_K] the basis of source receivers' class-mean-shift SUPERVECTORS
s_r = vec(mu_{t,r} - mu_t) (rank <= R-1), scaled so that source coefficients are white. A new receiver
is J = R-1 numbers a_q, estimated by EM from unlabeled packets with prior a ~ N(0, tau I) (tau fixed = 4,
i.e. up to 2x the source spread), optionally re-estimating class priors (Saerens et al. 2002).

Methods per target RX: raw (frozen head) | lda (a = 0) | ca (label-free centring) | em_free (class
means re-estimated freely, shrunk) | eig (eigen-receiver EM, uniform prior) | eig_pi (+ prior EM)
Calibration sets: all target packets (transductive), N = 50 / 200 balanced-random, N = 200 Dirichlet(1)
class prior; adapted model then scores all target packets. Oracle references: cc and oracle head
come from p2e_audit / p2e_generality.

    python3 p2e_eigenrx.py --dataset shen|wisig --out reports/receiver_p2e/eigenrx
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'receiver_p0')]
import p0_common as C   # noqa: E402

C.add_root()
GAI = Path('/workspace/rl1/gai')
TAU = 4.0


class EigenRX:
    def __init__(self, hs, ys, rxs, K, var=.995, cap=32, shrink=.1):
        self.m = hs.mean(0)
        ev = np.linalg.svd(hs - self.m, compute_uv=False) ** 2
        k = max(2, min(cap, int(np.searchsorted(np.cumsum(ev) / ev.sum(), var)) + 1))
        self.V = np.linalg.svd(hs - self.m, full_matrices=False)[2][:k].T
        g = self.proj(hs)
        rs = np.unique(rxs)
        self.K = K
        mu_tr = np.stack([np.stack([g[(ys == t) & (rxs == r)].mean(0) for r in rs]) for t in range(K)])
        self.mu = mu_tr.mean(1)                                          # (K, k)
        res = np.concatenate([g[(ys == t) & (rxs == r)] - mu_tr[t, i] for t in range(K) for i, r in enumerate(rs)])
        S = res.T @ res / len(res)
        S = (1 - shrink) * S + shrink * np.trace(S) / k * np.eye(k)
        self.Si = np.linalg.inv(S)
        sup = (mu_tr - self.mu[:, None]).transpose(1, 0, 2).reshape(len(rs), -1)   # (R, K*k)
        U, sv, Vt = np.linalg.svd(sup, full_matrices=False)
        keep = sv > 1e-6 * sv.max()
        self.E = (Vt[keep].T * (sv[keep] / np.sqrt(len(rs)))).reshape(K, k, -1)   # (K, k, J), white coefs

    def proj(self, h):
        return (h - self.m) @ self.V

    def loglik(self, g, means, logpi):
        d = g[:, None, :] - means[None]                                   # (n, K, k)
        return -.5 * np.einsum('nkd,de,nke->nk', d, self.Si, d) + logpi

    def fit_predict(self, g_cal, g_test, mode, init_post=None, iters=20):
        K, J = self.K, self.E.shape[2]
        logpi = np.full(K, -np.log(K))
        a = np.zeros(J)
        Dfree = np.zeros_like(self.mu)
        means = self.mu.copy()
        if mode == 'lda':
            return self.loglik(g_test, means, logpi).argmax(1)
        post = init_post if init_post is not None else softmax(self.loglik(g_cal, means, logpi))
        for _ in range(iters):
            n = post.sum(0) + 1e-9
            gbar = post.T @ g_cal / n[:, None]                            # (K, k) posterior class means
            if mode == 'em_free':
                Dfree = (n / (n + 20.))[:, None] * (gbar - self.mu)
                means = self.mu + Dfree
            else:
                A = sum(n[t] * self.E[t].T @ self.Si @ self.E[t] for t in range(K)) + np.eye(J) / TAU
                rhs = sum(n[t] * self.E[t].T @ self.Si @ (gbar[t] - self.mu[t]) for t in range(K))
                a = np.linalg.solve(A, rhs)
                means = self.mu + np.einsum('tkj,j->tk', self.E, a)
            if mode == 'eig_pi':
                logpi = np.log(np.clip(post.mean(0), 1e-4, None))
            post = softmax(self.loglik(g_cal, means, logpi))
        return self.loglik(g_test, means, logpi).argmax(1)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def evaluate_target(er, hs_mean, ht, yt, W, b, rng):
    """All methods x calibration regimes for one target RX. Returns list of rows."""
    rows = []
    K = W.shape[0]
    cls = [t for t in range(K) if (yt == t).sum() >= 4]
    keep = np.isin(yt, cls)
    ht, yt = ht[keep], yt[keep]
    g = er.proj(ht)
    z = ht @ W.T + b
    acc = lambda p: float((p == yt).mean())
    base = dict(acc_raw=acc(z.argmax(1)), acc_lda=acc(er.fit_predict(None, g, 'lda')))
    regimes = [('all', None)]
    for n in (50, 200):
        regimes += [(f'bal{n}', ('bal', n))]
    regimes += [('dir200', ('dir', 200))]
    for name, spec in regimes:
        draws = 1 if spec is None else 5
        res = {m: [] for m in ('ca', 'em_free', 'eig', 'eig_pi')}
        for _ in range(draws):
            if spec is None:
                ix = np.arange(len(yt))
            elif spec[0] == 'bal':
                ix = rng.choice(len(yt), min(spec[1], len(yt)), replace=False)
            else:
                p = rng.dirichlet(np.ones(len(cls)))
                pick = rng.choice(len(cls), spec[1], p=p)
                ix = np.array([rng.choice(np.flatnonzero(yt == cls[c])) for c in pick])
            post0 = softmax(z[ix])
            res['ca'].append(acc(((ht - ht[ix].mean(0) + hs_mean) @ W.T + b).argmax(1)))
            for m in ('em_free', 'eig', 'eig_pi'):
                res[m].append(acc(er.fit_predict(g[ix], g, m, init_post=post0)))
        rows.append(dict(calib=name, **base, **{f'acc_{m}': float(np.mean(v)) for m, v in res.items()}))
    return rows


def run_shen(out, device):
    import torch
    from b1_baselines.model import FeatureCNN
    from receiver_r01.data import shen as load_shen
    from audit_p2 import forward
    reg = C.registration()
    rows, cache = [], {}
    for d in sorted((GAI / 'outputs/receiver_p2e/formal').glob('shen_rank16_CV*_formal_seed*')):
        obj = torch.load(d / 'selected.pt', map_location='cpu', weights_only=False)
        j = obj['job']
        if j['arm'] not in ('ce', 'ls', 'ccalign', 'noise', 'ccanoise') or j.get('epochs', 20) != 20:
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
            m = tg['rx'] == r
            for row in evaluate_target(er, hs.mean(0), ht[m], tg['y'][m], W, b, rng):
                rows.append(dict(dataset='shen', fold=j['fold'], arm=j['arm'], seed=j['seed'],
                                 rx=reg['shen_allowed'][int(r)], J=er.E.shape[2], **row))
        print(d.name, flush=True)
        pd.DataFrame(rows).to_csv(out / 'shen_eigenrx.csv', index=False)


def run_wisig(out, device):
    from receiver_r01.data import wisig as load_wisig
    from p2e_generality import wisig_features, head_of
    reg = C.registration()
    names = reg['wisig']['rx_names']
    roles = reg['wisig']['roles']
    rows = []
    pools = {'formal': (roles['source'], roles['internal_rx']), 'search': (reg['wisig_inner_train'], reg['wisig_inner_held'])}
    for stage, (fit, tgk) in pools.items():
        src = load_wisig(reg, fit, [0, 1], [0, 120])
        tg = load_wisig(reg, tgk, [0, 1], [0, 200])
        if stage == 'formal':
            jobs = [(j['run_id'], j['backbone'], j['method'], j['seed'], Path(j['ckpt'])) for j in C.jobs('wisig')]
        else:
            jobs = [(d.name, d.name.split('_')[1], '_'.join(d.name.split('_')[3:-2]), -1, d)
                    for d in sorted((C.ROOT / 'outputs/receiver_r01/search').glob('wisig_*_search_*'))
                    if (d / 'status.json').exists()]
        for rid, bb, meth, seed, ck in jobs:
            if meth not in ('ce', 'swad', 'mldg_rx'):
                continue
            obj, _ = C.load_model(ck)
            model, _ = C.build_model(dict(dataset='wisig', backbone=bb), obj)
            model = model.to(device).eval()
            W, b = head_of(model, bb)
            hs, _ = wisig_features(model, src['x'], device, bb)
            ht, _ = wisig_features(model, tg['x'], device, bb)
            er = EigenRX(hs, src['y'], src['rx'], 10)
            rng = np.random.default_rng(20261022)
            for r in np.unique(tg['rx']):
                m = tg['rx'] == r
                for row in evaluate_target(er, hs.mean(0), ht[m], tg['y'][m], W, b, rng):
                    rows.append(dict(dataset='wisig', stage=stage, run_id=rid, backbone=bb, arm=meth, seed=seed,
                                     rx=names[int(r)], J=er.E.shape[2], **row))
            print(stage, rid, flush=True)
            pd.DataFrame(rows).to_csv(out / 'wisig_eigenrx.csv', index=False)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['shen', 'wisig'], required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (run_shen if a.dataset == 'shen' else run_wisig)(out, torch.device('cuda:0'))


if __name__ == '__main__':
    main()
