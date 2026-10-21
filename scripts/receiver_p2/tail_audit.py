#!/usr/bin/env python3
"""P2D gate 0: no-training margin-tail decomposition over all existing Shen checkpoints
(ce R0/R1 + P1 ls + P2/P2B formal). Inference only, GPU, ~117 models.

Question (PROTOCOL_P2D §0, pre-declared): after mean-level alignment arms (rxnull/rxlogit/
ccalign) have flattened kappa, is the REMAINING outer-RX error explained by margin tail
statistics (variance / low quantile / realised P(M<=0)) rather than by kappa itself?

Per (model, outer RX): for every ordered pair (t,u) on true-class-t packets,
  mu_{t,u,r}   mean margin M = z_t - z_u          (GPT §4.1 statistics)
  sd_{t,u,r}   std of M
  q10_{t,u,r}  10% quantile of M                  (tail position)
  pflip        realised P(M <= 0)                 (pair-level flip rate)
  conf_{t->u,r} realised P(pred = u | y = t)      (directed confusion)
plus Cantelli bound b = sd^2/(sd^2+softplus(mu)^2) per (t,u,r) and its calibration:
corr between b and pflip across (t,u,r) cells.

Fixed-factor analyses (per fold, across models x outer RX cells):
  A) among models with |kappa - kappa_ref| small (within-quantile), does lower tail risk
     predict higher per-RX accuracy?   (partial Spearman, controlling kappa)
  B) among models with similar tail risk, does kappa still predict accuracy? (control)
Outputs tail_cells.csv, tail_models.csv, tail_verdict.json.

    python3 tail_audit.py --out reports/receiver_p2d/tail_audit
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


def model_list():
    import torch
    runs = []
    for j in C.jobs('shen'):
        if j['method'] == 'ce':
            C.load_model(j['ckpt'])
            runs.append(dict(arm='ce', fold=j['fold'], seed=j['seed'], path=Path(j['ckpt']) / 'selected.pt'))
    for phase in ('receiver_p1', 'receiver_p2', 'receiver_p2b'):
        for d in sorted((GAI / 'outputs' / phase / 'formal').glob('shen_rank16_CV*_formal_seed*')):
            if not (d / 'status.json').exists():
                continue
            j = torch.load(d / 'selected.pt', map_location='cpu', weights_only=False)['job']
            runs.append(dict(arm=j['method'], fold=j['fold'], seed=j['seed'], path=d / 'selected.pt'))
    return runs


def forward_logits(model, x, device):
    import torch
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            xb = torch.as_tensor(np.ascontiguousarray(x[i:i + 512]), device=device)
            outs.append(model(xb).double().cpu())
    return torch.cat(outs).numpy()


def cell_stats(z, y, rx, rxs, K=10):
    """Per (t,u,r): mean, sd, q10, P(M<=0), directed confusion, Cantelli bound."""
    rows = []
    for r in rxs:
        for t in range(K):
            m = (rx == r) & (y == t)
            if not m.any():
                continue
            zr = z[m]
            pred = zr.argmax(1)
            for u in range(K):
                if u == t:
                    continue
                marg = zr[:, t] - zr[:, u]
                mu = float(marg.mean())
                sd = float(marg.std())
                q10 = float(np.quantile(marg, 0.1))
                pflip = float((marg <= 0).mean())
                conf = float((pred == u).mean())
                b = sd ** 2 / (sd ** 2 + np.log1p(np.exp(mu)) ** 2 + 1e-12)
                rows.append(dict(rx=r, t=t, u=u, mu=mu, sd=sd, q10=q10, pflip=pflip,
                                conf=conf, cantelli=float(b), n=int(m.sum())))
    return rows


def main():
    import torch
    from b1_baselines.model import FeatureCNN
    from receiver_r01.data import shen as load_shen
    from receiver_r01.core import metrics
    import p2_losses as L

    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = C.registration()
    device = torch.device('cuda:0')
    runs = model_list()
    print(f'{len(runs)} checkpoints')

    cells, models = [], []
    for fold in (0, 1):
        f = reg['shen_folds'][fold]
        cache = None
        for ru in [q for q in runs if q['fold'] == fold]:
            obj = torch.load(ru['path'], map_location='cpu', weights_only=False)
            sc = obj['scaler']
            if cache is None or abs(cache['sc']['mean'] - sc['mean']) > 1e-12:
                keys = f['inner_train'] + f['inner_validation']
                cache = dict(sc=sc, src=load_shen(reg, keys, sc, 'train')[0],
                             val=load_shen(reg, keys, sc, 'val')[0],
                             out=load_shen(reg, f['outer_evaluation'], sc, 'all')[0])
            model = FeatureCNN()
            model.load_state_dict(obj['model'])
            model = model.to(device).eval()
            zo = forward_logits(model, cache['out']['x'], device)
            y, rx = cache['out']['y'], cache['out']['rx']
            rxs = np.unique(rx)
            acc_r = [float((zo[rx == r].argmax(1) == y[rx == r]).mean()) for r in rxs]

            # kappa on source (reuse audit definition via p2_losses on cached features)
            from receiver_r01.data import shen as _ls
            src = cache['src']
            pres = []
            with torch.no_grad():
                xs = torch.as_tensor(src['x'], device=device)
                for i in range(0, len(xs), 512):
                    xb = xs[i:i + 512]
                    import torch.nn.functional as F
                    h = F.relu(model.stem(F.pad(xb, (2, 3, 2, 3))))
                    h = model.blocks(h).permute(0, 2, 3, 1).reshape(xb.shape[0], -1)
                    fq = F.normalize(model.feature(h), p=2, dim=1, eps=1e-12)
                    pres.append(F.relu(model.tx_input(fq)).float().cpu())
            pre = torch.cat(pres)
            Wt = model.classifier.weight.detach().float().cpu()
            kap = float(L.rxlogit(pre, Wt, torch.as_tensor(src['y']), torch.as_tensor(src['rx'])))

            tailr, pflips, confs, cants = [], [], [], []
            for row in cell_stats(zo, y, rx, rxs):
                cells.append(dict(fold=fold, arm=ru['arm'], seed=ru['seed'], **row))
                tailr.append(row['q10'])
                pflips.append(row['pflip'])
                confs.append(row['conf'])
                cants.append(row['cantelli'])
            models.append(dict(fold=fold, arm=ru['arm'], seed=ru['seed'], kappa=kap,
                               acc=float(np.mean(acc_r)), worst_rx=float(np.min(acc_r)),
                               mean_q10=float(np.mean(tailr)), mean_pflip=float(np.mean(pflips)),
                               mean_conf=float(np.mean(confs)), mean_cantelli=float(np.mean(cants))))
            print(fold, ru['arm'], ru['seed'], 'kap=%.4f acc=%.4f pflip=%.4f' % (kap, np.mean(acc_r), np.mean(pflips)), flush=True)

    CE = pd.DataFrame(cells)
    M = pd.DataFrame(models)
    CE.to_csv(out / 'tail_cells.csv', index=False)
    M.to_csv(out / 'tail_models.csv', index=False)

    # fixed-factor analyses per fold
    def spear(x, y):
        return float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1])

    def partial(x, y, z):
        rx_, ry_, rz = (pd.Series(v).rank().values for v in (x, y, z))
        res = lambda q: q - np.polyval(np.polyfit(rz, q, 1), rz)
        return float(np.corrcoef(res(rx_), res(ry_))[0, 1])

    verdict = dict(n_checkpoints=len(M))
    for fold in (0, 1):
        d = M[M.fold == fold]
        verdict[f'fold{fold}'] = dict(
            n=len(d),
            spear_kappa_acc=spear(d.kappa, d.acc),
            spear_pflip_acc=spear(d.pflip if 'pflip' in d else d.mean_pflip, d.acc),
            spear_q10_acc=spear(d.mean_q10, d.acc),
            spear_cantelli_acc=spear(d.mean_cantelli, d.acc),
            partial_q10_given_kappa=partial(d.mean_q10, d.acc, d.kappa),
            partial_kappa_given_q10=partial(d.kappa, d.acc, d.mean_q10),
            spear_cantelli_pflip_cells=float(pd.DataFrame(
                CE[CE.fold == fold]).groupby(['arm', 'seed']).apply(
                lambda g: spear(g.cantelli, g.pflip)).mean()),
        )
    # calibration: does Cantelli bound track realised flip rate at cell level (per fold pooled)
    for fold in (0, 1):
        g = CE[CE.fold == fold]
        verdict[f'fold{fold}']['cell_cantelli_pflip_spear'] = spear(g.cantelli, g.pflip)
    (out / 'tail_verdict.json').write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))


if __name__ == '__main__':
    main()
