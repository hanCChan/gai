#!/usr/bin/env python3
"""P2E-G: is the Shen error regularity general? Dataset-agnostic re-implementation of the oracle
decomposition (independent of p2e_audit.py, used as a cross-check on Shen) applied to Shen and WiSig.

Regularities tested (REPORT_P2E §8):
  R1  across receivers, between-class input variance is ~constant while within-class variance varies
  R2  unseen-RX loss = recoverable class-mean shift + oracle-bounded loss, and the oracle-bounded part
      grows with the receiver's within-class input variance
  R3  unseen-RX class-mean shifts lie in the span of source-RX shifts (vs random subspaces)

Per checkpoint x unseen RX q (features h = penultimate layer, frozen linear head W, b; statistics from all
q packets; accuracy on the odd half of each class present on q):
  raw, ca (label-free centring), cc (class-mean fix), ccproj (fix restricted to the source shift span),
  ccrand (random span of equal dim inside the source PCA span), oracle (linear head refit on q's even half)

WiSig pools: formal ckpts (trained on the 14 source RX) -> the 10 internal RX, dates 0-1 (development
data; diagnosis only, as in P0/D); search ckpts (10 inner-train RX) -> the 4 held source RX.
Never read: final 8 RX, date 2/3, equalised data.

    python3 p2e_generality.py --dataset wisig --out reports/receiver_p2e/generality
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


def basis(M, tol=1e-6):
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[S > tol * max(S.max(), 1e-30)].T


def decompose(hs, ys, rxs, ht, yt, W, b, V, rng, device, refit):
    """One unseen RX. hs/ys/rxs: source fitting features; ht/yt: target features. Returns dict."""
    K = W.shape[0]
    cls = [t for t in range(K) if (yt == t).sum() >= 4]
    rs = np.unique(rxs)
    mu_sr = np.stack([np.stack([hs[(ys == t) & (rxs == r)].mean(0) for r in rs]) for t in range(K)])
    muS = mu_sr.mean(1)
    keep = np.isin(yt, cls)
    ht, yt = ht[keep], yt[keep]
    odd = np.zeros(len(yt), bool)
    for t in cls:
        ix = np.flatnonzero(yt == t)
        odd[ix[1::2]] = True
    lin = lambda h: h @ W.T + b
    acc = lambda h: float((lin(h)[odd].argmax(1) == yt[odd]).mean())
    # class shifts estimated on the EVEN half only (the odd half is the test half)
    Dq = {t: ht[(yt == t) & ~odd].mean(0) - muS[t] for t in cls}
    out = dict(n_cls=len(cls), acc_raw=acc(ht), acc_ca=acc(ht - ht.mean(0) + hs.mean(0)))

    def lf(shift):
        """Label-free decode: candidate class u is scored at h - shift[u] (= per-class logit bias
        -w_u . shift[u] for a linear head); the test label is never used. (Audit A1: the earlier
        h - D[y_true] decode puts the label into the features and is kept only as *_leak.)"""
        z = lin(ht[odd])
        for u in cls:
            z[:, u] -= W[u] @ shift[u]
        bad = [u for u in range(K) if u not in cls]
        z[:, bad] = -np.inf
        return float((z.argmax(1) == yt[odd]).mean())

    hc, hp, hr = ht.copy(), ht.copy(), ht.copy()
    sp, sr = {}, {}
    cf, cr, mg = [], [], []
    for t in cls:
        m = yt == t
        U = basis(mu_sr[t] - muS[t])
        R = np.linalg.qr(V @ rng.standard_normal((V.shape[1], min(U.shape[1], V.shape[1]))))[0]
        sp[t], sr[t] = Dq[t] @ U @ U.T, Dq[t] @ R @ R.T
        hc[m] -= Dq[t]
        hp[m] -= sp[t]
        hr[m] -= sr[t]
        n2 = (Dq[t] ** 2).sum() + 1e-12
        cf.append(((Dq[t] @ U) ** 2).sum() / n2)
        cr.append(((Dq[t] @ R) ** 2).sum() / n2)
        mg.append(np.sqrt(n2 / (((mu_sr[t] - muS[t]) ** 2).sum(1).mean() + 1e-12)))
    out.update(acc_cclf=lf(Dq), acc_ccprojlf=lf(sp), acc_ccrandlf=lf(sr),
               acc_cc_leak=acc(hc), acc_ccproj_leak=acc(hp), acc_ccrand_leak=acc(hr),
               cov_feat=float(np.mean(cf)), cov_rand=float(np.mean(cr)), mag=float(np.mean(mg)))
    out.update(lda_metrics(hs, ys, rxs, ht, yt, odd, cls, K))
    out['acc_oracle'] = refit(ht[~odd], yt[~odd], ht[odd], yt[odd], K, device)
    return out


def lda_metrics(hs, ys, rxs, ht, yt, odd, cls, K):
    """Label-free-at-test class-mean decomposition in a Gaussian (LDA) head: source PCA space, pooled
    within-(class,RX) covariance; class means from the EVEN half; scored on the odd half.
      lda_src     source class means                      (no target information)
      lda_cc      target class means (oracle first moment; decision directions move with the means)
      lda_ccproj  target shift projected on the source shift span of that class
      lda_eig     eigen-receiver: J coefficients fitted to the even half (source-span supervector prior)"""
    from p2e_eigenrx import EigenRX, TAU
    er = EigenRX(hs, ys, rxs, K)
    g = er.proj(ht)
    ev = ~odd
    bad = [u for u in range(K) if u not in cls]

    def score(means):
        z = er.loglik(g[odd], means, np.zeros(K))
        z[:, bad] = -np.inf
        return float((z.argmax(1) == yt[odd]).mean())

    mcc = er.mu.copy()
    for t in cls:
        mcc[t] = g[ev & (yt == t)].mean(0)
    mproj, mrand = er.mu.copy(), er.mu.copy()
    rng = np.random.default_rng(20261022)
    dims = []
    for t in cls:
        Q = np.linalg.qr(er.E[t])[0]
        Rq = np.linalg.qr(rng.standard_normal((er.mu.shape[1], Q.shape[1])))[0]   # same dim, random, in PCA space
        mproj[t] = er.mu[t] + (mcc[t] - er.mu[t]) @ Q @ Q.T
        mrand[t] = er.mu[t] + (mcc[t] - er.mu[t]) @ Rq @ Rq.T
        dims.append(Q.shape[1])
    n = np.array([(ev & (yt == t)).sum() for t in range(K)], float)
    J = er.E.shape[2]
    A = sum(n[t] * er.E[t].T @ er.Si @ er.E[t] for t in cls) + np.eye(J) / TAU
    rhs = sum(n[t] * er.E[t].T @ er.Si @ (mcc[t] - er.mu[t]) for t in cls)
    meig = er.mu + np.einsum('tkj,j->tk', er.E, np.linalg.solve(A, rhs))
    return dict(acc_lda_src=score(er.mu), acc_lda_cc=score(mcc), acc_lda_ccproj=score(mproj),
                acc_lda_ccrand=score(mrand), acc_lda_eig=score(meig), J=J, pca_k=er.mu.shape[1])


def refit_linear(xtr, ytr, xte, yte, K, device):
    import torch
    head, _ = C.refit_head(lambda: torch.nn.Linear(xtr.shape[1], K), xtr, ytr, device)
    with torch.no_grad():
        z = head(torch.as_tensor(xte, dtype=torch.float32, device=device)).cpu().numpy()
    return float((z.argmax(1) == yte).mean())


def pca_basis(hs, var=.995, cap=32):
    ev = np.linalg.svd(hs - hs.mean(0), compute_uv=False) ** 2
    k = max(2, min(cap, int(np.searchsorted(np.cumsum(ev) / ev.sum(), var)) + 1))
    return np.linalg.svd(hs - hs.mean(0), full_matrices=False)[2][:k].T


def input_stats(x, y, rx, names):
    rows = []
    for r in np.unique(rx):
        m = rx == r
        xs, ys = x[m].reshape(m.sum(), -1).astype(np.float64), y[m]
        cl = [t for t in np.unique(ys) if (ys == t).sum() >= 4]
        mu = np.stack([xs[ys == t].mean(0) for t in cl])
        rows.append(dict(rx=names[int(r)], within=float(np.mean([xs[ys == t].var(0).mean() for t in cl])),
                         between=float(mu.var(0).mean()), n_cls=len(cl)))
    D = pd.DataFrame(rows)
    D['fisher'] = D.between / D.within
    return D


def wisig_features(model, x, device, backbone):
    import torch
    import torch.nn.functional as F
    hs, zs = [], []
    with torch.no_grad():
        for i in range(0, len(x), 1024):
            xb = torch.as_tensor(np.ascontiguousarray(x[i:i + 1024]), device=device)
            if backbone == 'official':
                h = model.encode(xb)
                z = model.head(h)
            else:
                h = F.relu(model.head[1](model.head[0](model.body(xb))))
                z = model.head[4](h)
            hs.append(h.double().cpu())
            zs.append(z.double().cpu())
    return torch.cat(hs).numpy(), torch.cat(zs).numpy()


def head_of(model, backbone):
    lin = model.head if backbone == 'official' else model.head[4]
    return lin.weight.detach().double().cpu().numpy(), lin.bias.detach().double().cpu().numpy()


def run_wisig(a, out, device):
    import torch
    from receiver_r01.data import wisig as load_wisig
    reg = C.registration()
    names = reg['wisig']['rx_names']
    roles = reg['wisig']['roles']
    pools = {
        'formal': dict(fit=roles['source'], tg=roles['internal_rx']),
        # search ckpts: the 4 inner-held RX chose lr and epoch (audit A3) -> "selection RX", not unseen
        'selection_rx': dict(fit=reg['wisig_inner_train'], tg=reg['wisig_inner_held']),
    }
    # R1 input statistics on every non-final RX, dates 0-1 (raw IQ and phase-free log power spectrum)
    allrx = roles['source'] + roles['internal_rx']
    X = load_wisig(reg, allrx, [0, 1], [0, 200])
    z = X['x'][:, 0, :, 0] + 1j * X['x'][:, 0, :, 1]
    lps = np.log10(np.abs(np.fft.fft(z, axis=-1)) ** 2 + 1e-12)
    for tag, rep in (('rawiq', X['x']), ('logpsd', lps)):
        S = input_stats(rep, X['y'], X['rx'], names)
        S['role'] = ['source' if names.index(k) in roles['source'] else 'internal' for k in S.rx]
        S.to_csv(out / f'wisig_input_{tag}.csv', index=False)
        print(tag, '\n', S.round(5).to_string(index=False), flush=True)
    del X, z, lps
    rows = []
    for stage, pool in pools.items():
        src = load_wisig(reg, pool['fit'], [0, 1], [0, 120])
        tg = load_wisig(reg, pool['tg'], [0, 1], [0, 200])
        if stage == 'formal':
            jobs = [(j['run_id'], j['backbone'], j['method'], j['seed'], Path(j['ckpt'])) for j in C.jobs('wisig')]
        else:
            jobs = []
            for d in sorted((C.ROOT / 'outputs/receiver_r01/search').glob('wisig_*_search_*')):
                if (d / 'status.json').exists():
                    parts = d.name.split('_')
                    jobs.append((d.name, parts[1], '_'.join(parts[3:-2]), -1, d))
        for rid, bb, meth, seed, ck in jobs:
            obj, _ = C.load_model(ck)
            job = dict(dataset='wisig', backbone=bb)
            model, _ = C.build_model(job, obj)
            model = model.to(device).eval()
            W, b = head_of(model, bb)
            hs, _ = wisig_features(model, src['x'], device, bb)
            ht, _ = wisig_features(model, tg['x'], device, bb)
            V = pca_basis(hs)
            rng = np.random.default_rng(20261022)
            for r in np.unique(tg['rx']):
                m = tg['rx'] == r
                d = decompose(hs, src['y'], src['rx'], ht[m], tg['y'][m], W, b, V, rng, device, refit_linear)
                rows.append(dict(dataset='wisig', stage=stage, run_id=rid, backbone=bb, arm=meth, seed=seed,
                                 rx=names[int(r)], **d))
            print(stage, rid, 'raw=%.3f' % np.mean([q['acc_raw'] for q in rows if q['run_id'] == rid]), flush=True)
            pd.DataFrame(rows).to_csv(out / 'wisig_targets.csv', index=False)


def run_shen_crosscheck(a, out, device):
    """Same decomposition on Shen: P2E formal checkpoints (one trainer, 5 seeds), outer RX."""
    import torch
    from b1_baselines.model import FeatureCNN
    from receiver_r01.data import shen as load_shen
    from audit_p2 import forward
    reg = C.registration()
    rows, cache = [], {}
    X = {k: load_shen(reg, [k], dict(mean=0., std=1.), 'all')[0] for k in reg['shen_allowed']}
    S = input_stats(np.concatenate([X[k]['x'] for k in X]), np.concatenate([X[k]['y'] for k in X]),
                    np.concatenate([X[k]['rx'] for k in X]), reg['shen_allowed'])
    S.to_csv(out / 'shen_input_cis.csv', index=False)
    del X
    for d in sorted((GAI / 'outputs/receiver_p2e/formal').glob('shen_rank16_CV*_formal_seed*')):
        obj = torch.load(d / 'selected.pt', map_location='cpu', weights_only=False)
        j = obj['job']
        if j['arm'] not in ('ce', 'ls', 'ccalign', 'noise', 'ccanoise', 'ncons') or j.get('epochs', 20) != 20:
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
        V = pca_basis(hs)
        rng = np.random.default_rng(20261022)
        for r in np.unique(tg['rx']):
            m = tg['rx'] == r
            dd = decompose(hs, src['y'], src['rx'], ht[m], tg['y'][m], W, b, V, rng, device, refit_linear)
            rows.append(dict(dataset='shen', fold=j['fold'], arm=j['arm'], seed=j['seed'],
                             rx=reg['shen_allowed'][int(r)], **dd))
        print(d.name, flush=True)
        pd.DataFrame(rows).to_csv(out / 'shen_targets.csv', index=False)


def selftest():
    """Audit A1 regression test: on features with no class information the label-free decode must be at
    chance while the leaky decode is not."""
    rng = np.random.default_rng(0)
    K, d = 10, 16
    ys = np.repeat(np.arange(K), 200)
    proto = 3 * rng.standard_normal((K, d))
    hs = proto[ys] + rng.standard_normal((2000, d))          # separable source
    rxs = np.tile(np.arange(4), 500)
    ht = rng.standard_normal((2000, d))                       # target: no class information at all
    yt = np.repeat(np.arange(K), 200)
    W, b = proto, -.5 * (proto ** 2).sum(1)
    out = decompose(hs, ys, rxs, ht, yt, W, b, pca_basis(hs), rng, None, lambda *a: np.nan)
    assert out['acc_cclf'] < 0.2, out
    assert out['acc_lda_cc'] < 0.2, out
    assert out['acc_cc_leak'] > 0.5, out
    print('selftest ok: label-free %.3f, lda_cc %.3f, leaky %.3f' % (out['acc_cclf'], out['acc_lda_cc'], out['acc_cc_leak']))


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['wisig', 'shen', 'selftest'], required=True)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    if a.dataset == 'selftest':
        return selftest()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0')
    (run_wisig if a.dataset == 'wisig' else run_shen_crosscheck)(a, out, device)


if __name__ == '__main__':
    main()
