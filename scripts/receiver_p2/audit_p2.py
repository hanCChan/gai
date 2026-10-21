#!/usr/bin/env python3
"""P2A mechanism / implementation audit (docs/receiver_p2/PROTOCOL_P2A_AUDIT.md). Inference only.

Per checkpoint (ce R0/R1, all P1 formal, all P2 formal): rho, row/null displacement energy, the
function-level kappa (from centred logits), margin dispersion MD, within-cell scatter S_w, head
spectrum, source-val identity retention; outer-RX realized displacement kappa_out, rho_out, harmful
margin shrinkage H (diagnostic only). Then the pre-declared verdict G1-G5, the reparameterisation
test, and within-fold predictiveness of each source quantity for outer accuracy.

    python3 audit_p2.py --out reports/receiver_p2a
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'receiver_p0')]
import p0_common as C   # noqa: E402

C.add_root()
GAI = Path('/workspace/rl1/gai')
EPS = 1e-4


def model_list():
    import torch
    runs = []
    for j in C.jobs('shen'):
        if j['method'] == 'ce':
            C.load_model(j['ckpt'])   # hash guard
            runs.append(dict(arm='ce', fold=j['fold'], seed=j['seed'], path=Path(j['ckpt']) / 'selected.pt', src='R0R1'))
    for phase in ('receiver_p1', 'receiver_p2'):
        for d in sorted((GAI / 'outputs' / phase / 'formal').glob('shen_rank16_CV*_formal_seed*')):
            if not (d / 'status.json').exists():
                continue
            j = torch.load(d / 'selected.pt', map_location='cpu', weights_only=False)['job']
            arm = j['method'] + ('_rand' if j.get('set') == 'rand' else '')
            runs.append(dict(arm=arm, fold=j['fold'], seed=j['seed'], path=d / 'selected.pt', src=phase))
    return runs


def forward(model, x, device):
    import torch
    import torch.nn.functional as F
    hs, zs = [], []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            xb = torch.as_tensor(np.ascontiguousarray(x[i:i + 512]), device=device)
            h = F.relu(model.stem(F.pad(xb, (2, 3, 2, 3))))
            h = model.blocks(h).permute(0, 2, 3, 1).reshape(xb.shape[0], -1)
            h = F.relu(model.tx_input(F.normalize(model.feature(h), p=2, dim=1, eps=1e-12)))
            hs.append(h.double().cpu())
            zs.append(model.classifier(h).double().cpu())
    return torch.cat(hs).numpy(), torch.cat(zs).numpy()


def cell_means(v, y, rx):
    rs = np.unique(rx)
    return np.stack([np.stack([v[(y == t) & (rx == r)].mean(0) for r in rs]) for t in range(10)]), rs


def projectors(W):
    A = W - W.mean(0, keepdims=True)
    P_row = np.linalg.pinv(A, rcond=1e-10) @ A
    return A, P_row, np.eye(W.shape[1]) - P_row


def rho_of(Dflat, P_row):
    num = ((Dflat @ P_row) ** 2).sum(1)   # P_row symmetric
    return float(np.mean(num / ((Dflat ** 2).sum(1) + EPS)))


def kappa_from_logits(zc_cells, zc_ref_t):
    """zc_cells: (10, R, 10) centred-logit cell means; zc_ref_t: (10, 10) per-class reference."""
    AD = zc_cells - zc_ref_t[:, None, :]
    AC = zc_ref_t - zc_ref_t.mean(0, keepdims=True)
    return float((AD ** 2).sum(-1).mean() / (AC ** 2).sum(-1).mean()), AD, AC


def audit_model(h, z, y, rx, W, b, hv, zv, yv, ho, zo, yo, rxo):
    A, P_row, P_null = projectors(W)
    zc = z - z.mean(1, keepdims=True)
    mu, _ = cell_means(h, y, rx)                    # (10, R, 128)
    mu_t = mu.mean(1)
    D = (mu - mu_t[:, None]).reshape(-1, h.shape[1])
    Cc = mu_t - mu_t.mean(0)
    mz, _ = cell_means(zc, y, rx)                   # centred logits
    kap, AD, AC = kappa_from_logits(mz, mz.mean(1))
    # cross-check: A D from features equals logit-space displacement
    AD_feat = (D @ A.T).reshape(AD.shape)
    xcheck = float(np.abs(AD_feat - AD).max() / (np.abs(AD).max() + 1e-12))
    E_row = float(((D @ P_row) ** 2).sum(1).mean())
    E_null = float(((D @ P_null) ** 2).sum(1).mean())
    E_D = float((D ** 2).sum(1).mean())
    B_row = float(((Cc @ P_row) ** 2).sum(1).mean())
    mraw, _ = cell_means(z, y, rx)                  # raw logits for margins
    tt, uu = np.where(~np.eye(10, dtype=bool))
    m = mraw[tt, :, tt] - mraw[tt, :, uu]           # (90, R)
    MD = float(m.var(1).mean() / (m.mean(1) ** 2).mean())
    cell_z = mz[y, np.searchsorted(np.unique(rx), rx)]
    S_w = float(((zc - cell_z) ** 2).sum(1).mean() / (AC ** 2).sum(-1).mean())
    sv = np.linalg.svd(A, compute_uv=False)[:9]
    pv = zv.argmax(1)
    rec = [float((pv[yv == t] == t).mean()) for t in range(10)]
    # outer (diagnostic)
    zoc = zo - zo.mean(1, keepdims=True)
    muo, ros = cell_means(ho, yo, rxo)
    Dout = (muo - mu_t[:, None]).reshape(-1, h.shape[1])
    mzo, _ = cell_means(zoc, yo, rxo)
    ADo = mzo - mz.mean(1)[:, None, :]
    kap_out = float((ADo ** 2).sum(-1).mean() / (AC ** 2).sum(-1).mean())
    mro, _ = cell_means(zo, yo, rxo)
    m_src = mraw.mean(1)[tt, tt] - mraw.mean(1)[tt, uu]               # (90,)
    m_out = mro[tt, :, tt] - mro[tt, :, uu]                           # (90, R')
    dm = m_out - m_src[:, None]
    po = zo.argmax(1)
    acc_out = float(np.mean([(po[rxo == r] == yo[rxo == r]).mean() for r in ros]))
    return dict(rho=rho_of(D, P_row), E_row=E_row, E_null=E_null, E_D=E_D, B_row=B_row,
                n_row=E_row / B_row, n_null=E_null / B_row, kappa=kap, kappa_xcheck=xcheck, MD=MD, S_w=S_w,
                sv1=float(sv[0]), sv9=float(sv[-1]), A_fro=float(np.sqrt((sv ** 2).sum())),
                eff_rank=float((sv ** 2).sum() ** 2 / (sv ** 4).sum()), cond=float(sv[0] / sv[-1]),
                src_val_acc=float((pv == yv).mean()), src_min_recall=float(min(rec)),
                rho_out=rho_of(Dout, P_row), kappa_out=kap_out,
                H_out=float(np.maximum(0, -dm).mean() / m_src.mean()), frac_neg_out=float((dm < 0).mean()),
                acc_out=acc_out)


def reparam_test(h, z, y, rx, W, b):
    A, P_row, P_null = projectors(W)
    zc = z - z.mean(1, keepdims=True)
    mz, _ = cell_means(zc, y, rx)
    k0, _, _ = kappa_from_logits(mz, mz.mean(1))
    mu, _ = cell_means(h, y, rx)
    r0 = rho_of((mu - mu.mean(1, keepdims=True)).reshape(-1, h.shape[1]), P_row)
    sm = lambda q: np.exp(q - q.max(1, keepdims=True)) / np.exp(q - q.max(1, keepdims=True)).sum(1, keepdims=True)
    p0 = sm(h @ W.T + b)
    out = []
    for s in (0.25, 4.0, 16.0):
        h2 = h @ P_row + s * (h @ P_null)
        z2 = h2 @ W.T + b
        z2c = z2 - z2.mean(1, keepdims=True)
        m2, _ = cell_means(z2c, y, rx)
        k2, _, _ = kappa_from_logits(m2, m2.mean(1))
        mu2, _ = cell_means(h2, y, rx)
        r2 = rho_of((mu2 - mu2.mean(1, keepdims=True)).reshape(-1, h.shape[1]), P_row)
        out.append(dict(s=s, max_dsoftmax=float(np.abs(sm(z2) - p0).max()),
                        argmax_equal=bool((z2.argmax(1) == (h @ W.T + b).argmax(1)).all()),
                        kappa_rel_change=abs(k2 - k0) / k0, rho0=r0, rho_s=r2))
    return out


def spearman(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spearman(x, y, z):
    rx_, ry, rz = (pd.Series(v).rank().values for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(np.corrcoef(res(rx_), res(ry))[0, 1])


def main():
    import torch
    from b1_baselines.model import FeatureCNN
    from receiver_r01.data import shen as load_shen
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # static check: no outer RX access in the P2 training / selection code
    for f in ('train_p2.py', 'select_lambda_p2.py', 'queue_p2.py'):
        assert 'outer_evaluation' not in (HERE / f).read_text(), f
    reg = C.registration()
    device = torch.device('cuda:0')
    runs = model_list()
    rows, reparam = [], {}
    for fold in (0, 1):
        f = reg['shen_folds'][fold]
        cache = None
        for ru in [r for r in runs if r['fold'] == fold]:
            obj = torch.load(ru['path'], map_location='cpu', weights_only=False)
            sc = obj['scaler']
            if cache is None or abs(cache['sc']['mean'] - sc['mean']) > 1e-12 or abs(cache['sc']['std'] - sc['std']) > 1e-12:
                keys = f['inner_train'] + f['inner_validation']
                cache = dict(sc=sc, src=load_shen(reg, keys, sc, 'train')[0], val=load_shen(reg, keys, sc, 'val')[0],
                             out=load_shen(reg, f['outer_evaluation'], sc, 'all')[0])
            model = FeatureCNN()
            model.load_state_dict(obj['model'])
            model = model.to(device).eval()
            h, z = forward(model, cache['src']['x'], device)
            hv, zv = forward(model, cache['val']['x'], device)
            ho, zo = forward(model, cache['out']['x'], device)
            W = model.classifier.weight.detach().double().cpu().numpy()
            b = model.classifier.bias.detach().double().cpu().numpy()
            m = audit_model(h, z, cache['src']['y'], cache['src']['rx'], W, b, hv, zv, cache['val']['y'],
                            ho, zo, cache['out']['y'], cache['out']['rx'])
            rows.append(dict(fold=fold, arm=ru['arm'], seed=ru['seed'], src=ru['src'], **m))
            if ru['arm'] in ('ce', 'rxnull') and ru['seed'] == 20261013:
                reparam[f'{fold}_{ru["arm"]}'] = reparam_test(h, z, cache['src']['y'], cache['src']['rx'], W, b)
            print(fold, ru['arm'], ru['seed'], 'kappa=%.4f rho=%.3f n_row=%.4f acc_out=%.4f' %
                  (m['kappa'], m['rho'], m['n_row'], m['acc_out']), flush=True)
    M = pd.DataFrame(rows)
    M.to_csv(out / 'audit_models.csv', index=False)
    G = M.groupby(['fold', 'arm']).mean(numeric_only=True).drop(columns='seed')
    G.to_csv(out / 'audit_arms.csv')
    (out / 'reparam.json').write_text(json.dumps(reparam, indent=1))

    # predictiveness within fold
    pr = []
    for fold in (0, 1):
        d = M[M.fold == fold]
        for q in ('kappa', 'rho', 'n_row', 'MD', 'S_w', 'src_val_acc', 'kappa_out', 'H_out'):
            pr.append(dict(fold=fold, metric=q, n=len(d), spearman_acc_out=spearman(d[q], d.acc_out),
                           partial_given_srcacc=partial_spearman(d[q], d.acc_out, d.src_val_acc) if q != 'src_val_acc' else np.nan))
        pr.append(dict(fold=fold, metric='kappa->kappa_out', n=len(d), spearman_acc_out=spearman(d.kappa, d.kappa_out),
                       partial_given_srcacc=np.nan))
    P = pd.DataFrame(pr)
    P.to_csv(out / 'predictiveness.csv', index=False)

    # verdict
    g = lambda fold, arm, q: float(G.loc[(fold, arm), q])
    G1 = all(g(f_, 'rxnull', 'kappa') < g(f_, 'ce', 'kappa') and g(f_, 'rxnull', 'kappa') < g(f_, 'ls', 'kappa') for f_ in (0, 1))
    G2 = all(g(f_, 'rxnull', 'n_row') < g(f_, 'ce', 'n_row') for f_ in (0, 1))
    share = {f_: (np.log(g(f_, 'rxnull', 'n_row')) - np.log(g(f_, 'ce', 'n_row'))) /
                 (np.log(g(f_, 'rxnull', 'rho')) - np.log(g(f_, 'ce', 'rho'))) for f_ in (0, 1)}
    G3 = all(share[f_] >= 0.5 for f_ in (0, 1))
    G4 = all(g(f_, 'rxnull', 'eff_rank') >= 0.8 * g(f_, 'ce', 'eff_rank') and
             g(f_, 'rxnull', 'src_val_acc') >= g(f_, 'ce', 'src_val_acc') - 0.01 for f_ in (0, 1))
    G5 = all(all(t['max_dsoftmax'] < 1e-6 and t['argmax_equal'] and t['kappa_rel_change'] < 1e-6 for t in v)
             for v in reparam.values())
    xmax = float(M.kappa_xcheck.max())
    verdict = dict(G1_kappa_down=G1, G2_nrow_down=G2, G3_share_row=share, G3=G3, G4_rank_identity=G4,
                   G5_reparam_correct=G5, kappa_xcheck_max=xmax,
                   decision='AUDIT_BUG' if not G5 or xmax > 1e-6 else
                   ('MATHEMATICAL_GUARD_PASSED' if (G1 and G2 and G3 and G4) else 'MODIFY_OBJECTIVE'))
    (out / 'verdict.json').write_text(json.dumps(verdict, indent=1))
    cols = ['kappa', 'rho', 'n_row', 'n_null', 'E_D', 'MD', 'S_w', 'eff_rank', 'sv1', 'cond', 'src_val_acc',
            'kappa_out', 'rho_out', 'H_out', 'frac_neg_out', 'acc_out']
    with pd.option_context('display.width', 250):
        print(G[cols].round(4).to_string())
        print(P.round(3).to_string(index=False))
    print(json.dumps(reparam, indent=1))
    print(json.dumps(verdict, indent=1))


if __name__ == '__main__':
    main()
