#!/usr/bin/env python3
"""P2C post-hoc ISR-Multiclass control (PROTOCOL_P2C_ISR.md). Inference + linear-head refit only.

Per base checkpoint (ce, rxlogit, rxnull) and fold, on frozen pre-head features:
  refit         linear head refit, no projection
  isr           refit on Null(P') (ISR-Multiclass on the source per-class per-RX means)
  isr_normfilt  as isr, keeping only directions whose displacement norm > 10% of class separation

    python3 isr_posthoc.py --out reports/receiver_p2c
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
SEEDS = (20261013, 20261014, 20261015)


def pick(method, subdir):
    import torch
    out = []
    for d in sorted((GAI / 'outputs' / subdir).glob(f'shen_rank16_CV*_{method}*_formal_seed*')):
        if (d / 'status.json').exists():
            j = torch.load(d / 'selected.pt', map_location='cpu', weights_only=False)['job']
            if j['method'] == method and j['seed'] in SEEDS:
                out.append((j['fold'], j['seed'], d / 'selected.pt'))
    return out


def features(model, x, device):
    import torch
    import torch.nn.functional as F
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            xb = torch.as_tensor(np.ascontiguousarray(x[i:i + 512]), device=device)
            h = F.relu(model.stem(F.pad(xb, (2, 3, 2, 3))))
            h = model.blocks(h).permute(0, 2, 3, 1).reshape(xb.shape[0], -1)
            outs.append(F.relu(model.tx_input(F.normalize(model.feature(h), p=2, dim=1, eps=1e-12))).float().cpu())
    return torch.cat(outs).numpy().astype(np.float64)


def isr_basis(f, y, rx, norm_frac=None):
    """Null(P') basis from source per-class per-RX means. Returns projector P_null (d,d)."""
    d = f.shape[1]
    rxs = np.unique(rx)
    bass = []
    for t in range(10):
        mus = np.stack([f[(y == t) & (rx == r)].mean(0) for r in rxs])   # (E, d)
        D = mus - mus.mean(0, keepdims=True)                 # (E, d)
        _, S, Vt = np.linalg.svd(D, full_matrices=False)     # Vt: rows are feature-space directions
        keep = S > 1e-6 * (S.max() + 1e-30)
        Vt, S = Vt[keep], S[keep]
        if norm_frac is not None:
            class_norm = np.linalg.norm(mus.mean(0) - np.stack([f[y == q].mean(0) for q in range(10)]).mean(0))
            # audit B5: the threshold used class_norm / sqrt(d), i.e. ~0.9% of the class separation for
            # d=128, not the documented 10%. Fixed; the committed isr_normfilt rows used the old threshold.
            mk = S / np.sqrt(len(rxs)) > norm_frac * (class_norm + 1e-12)
            Vt, S = Vt[mk], S[mk]
        if len(S):
            bass.append(Vt * S[:, None])                     # (r, d): direction scaled by its strength
    if not bass:
        return np.eye(d), 0
    M = np.concatenate(bass, axis=0)                         # (sum r, d)
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    keep = S > 1e-3 * (S.max() + 1e-30)
    Q = Vt[keep].T                                           # (d, k) orthonormal basis of P'
    return np.eye(d) - Q @ Q.T, int(keep.sum())


def main():
    import torch
    from b1_baselines.model import FeatureCNN
    from train_receiver_r01 import build   # noqa: F401  (keeps import parity with P0 refit)
    from receiver_r01.data import shen as load_shen
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = C.registration()
    device = torch.device('cuda:0')
    jobs = [('ce', *x) for x in [(j['fold'], j['seed'], Path(j['ckpt']) / 'selected.pt')
                                 for j in C.jobs('shen') if j['method'] == 'ce' and j['seed'] in SEEDS]]
    jobs += [('rxlogit', *x) for x in pick('rxlogit', 'receiver_p2b/formal')]
    jobs += [('rxnull', *x) for x in pick('rxnull', 'receiver_p2/formal')]

    rows = []
    import torch.nn.functional as F
    for base, fold, seed, path in jobs:
        obj = torch.load(path, map_location='cpu', weights_only=False)
        model = FeatureCNN()
        model.load_state_dict(obj['model'])
        model = model.to(device).eval()
        f = reg['shen_folds'][fold]
        keys = f['inner_train'] + f['inner_validation']
        src = load_shen(reg, keys, obj['scaler'], 'train')[0]
        outd = load_shen(reg, f['outer_evaluation'], obj['scaler'], 'all')[0]
        hsrc = features(model, src['x'], device)
        hout = features(model, outd['x'], device)
        Pn_isr, k_isr = isr_basis(hsrc, src['y'], src['rx'])
        Pn_nf, k_nf = isr_basis(hsrc, src['y'], src['rx'], norm_frac=0.10)
        frozen = model.classifier(torch.as_tensor(hout).float().to(device)).detach().cpu().numpy()
        for tag, Pn, cut in (('frozen_head', np.eye(hsrc.shape[1]), 0),
                             ('refit', np.eye(hsrc.shape[1]), 0),
                             ('isr', Pn_isr, k_isr),
                             ('isr_normfilt', Pn_nf, k_nf)):
            hs = torch.as_tensor(hsrc @ Pn).float().to(device)
            ho = torch.as_tensor(hout @ Pn).float().to(device)
            y = torch.as_tensor(src['y'], device=device)
            if tag == 'frozen_head':
                head = model.classifier
            else:
                # P2D fix: refit arms (refit/isr/isr_normfilt) all use the SAME refit recipe
                # (fresh head, fixed seed BEFORE instantiation, CE + L2=1e-4 per REFIT),
                # so the ISR projection is the only difference between them.
                torch.manual_seed(C.REFIT['seed'])
                head = torch.nn.Linear(hsrc.shape[1], 10).to(device)
                opt = torch.optim.Adam(head.parameters(), lr=C.REFIT['lr'])
                for _ in range(C.REFIT['steps']):
                    opt.zero_grad()
                    loss = F.cross_entropy(head(hs), y)
                    loss = loss + C.REFIT['l2'] * sum((p ** 2).sum() for n, p in head.named_parameters() if 'weight' in n)
                    loss.backward()
                    opt.step()
                head.eval()
            with torch.no_grad():
                lg_o = head(ho).cpu().numpy()
                lg_s = head(hs).cpu().numpy()
            rxo = outd['rx']
            accs = [float((lg_o[rxo == r].argmax(1) == outd['y'][rxo == r]).mean()) for r in np.unique(rxo)]
            accs_s = [float((lg_s[src['rx'] == r].argmax(1) == src['y'][src['rx'] == r]).mean())
                      for r in np.unique(src['rx'])]
            pl = reg['shen_allowed'].index('shen_pluto_1_train')
            rows.append(dict(base=base, fold=fold, seed=seed, arm=tag, removed_dirs=cut,
                             src_fit_acc=float(np.mean(accs_s)), acc=float(np.mean(accs)),   # audit B5: refit packets, not val
                             worst_rx=float(np.min(accs)),
                             pluto=float(accs[list(np.unique(rxo)).index(pl)]) if pl in np.unique(rxo) else np.nan))
            print(rows[-1], flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / 'isr_posthoc.csv', index=False)
    S = R.groupby(['base', 'arm']).mean(numeric_only=True).round(4)
    S.to_csv(out / 'summary.csv')
    print(S.to_string())


if __name__ == '__main__':
    main()
