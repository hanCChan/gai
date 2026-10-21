#!/usr/bin/env python3
"""Outer-RX evaluation of P2 formal checkpoints (+ frozen ce from R0/R1 and P1 ls formal).
Same metrics as P1 eval: RX-equal acc/macro-F1, worst outer RX, Pluto, per RX/TX recall,
90 ordered-pair confusions + normalized margins, Rq-set and non-Rq aggregates, and the
end-of-training rho_row exposure diagnostic on source packets.

    python3 eval_p2.py --formal outputs/receiver_p2/formal --p1formal outputs/receiver_p1/formal \
        --jobs reports/receiver_p2/jobs_formal.txt --sets reports/receiver_p1/sets_p1.json --out reports/receiver_p2/eval
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
import p2_losses as L   # noqa: E402

C.add_root()


def main():
    import torch
    import torch.nn.functional as F
    from b1_baselines.model import FeatureCNN
    from receiver_r01.core import metrics
    from receiver_r01.data import shen as load_shen

    ap = argparse.ArgumentParser()
    ap.add_argument('--formal', required=True)
    ap.add_argument('--p1formal', required=True)
    ap.add_argument('--jobs', required=True)
    ap.add_argument('--sets', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = C.registration()
    sets = json.loads(Path(a.sets).read_text())['folds']
    device = torch.device('cuda:0')

    runs = []
    n_jobs = 0
    for d in sorted(Path(a.formal).glob('shen_rank16_CV*_formal_seed*')):
        assert (d / 'status.json').exists(), f'incomplete: {d}'
        j = torch.load(d / 'selected.pt', map_location='cpu', weights_only=False)['job']
        runs.append(dict(path=d / 'selected.pt', fold=j['fold'], method=j['method'], lam=j['lam'],
                         seed=j['seed']))
        n_jobs += 1
    expected = sum(1 for l in Path(a.jobs).read_text().splitlines() if l.strip() and not l.startswith('#'))
    assert n_jobs == expected, (n_jobs, expected)
    for j in C.jobs('shen'):   # frozen R0/R1 ce anchors (hash-guarded)
        if j['method'] == 'ce':
            C.load_model(j['ckpt'])
            runs.append(dict(path=Path(j['ckpt']) / 'selected.pt', fold=j['fold'], method='ce',
                             lam=0., seed=j['seed']))
    for d in sorted(Path(a.p1formal).glob('shen_rank16_CV*_formal_seed*')):   # P1 ls control
        j = torch.load(d / 'selected.pt', map_location='cpu', weights_only=False)['job']
        if j['method'] == 'ls' and j['set'] == 'Rq':
            runs.append(dict(path=d / 'selected.pt', fold=j['fold'], method='ls', lam=0., seed=j['seed']))

    R, RX, TX, PR = [], [], [], []
    for ru in runs:
        obj = torch.load(ru['path'], map_location='cpu', weights_only=False)
        model = FeatureCNN()
        model.load_state_dict(obj['model'])
        model = model.to(device).eval()
        fold = ru['fold']
        f = reg['shen_folds'][fold]
        Pset = set(tuple(p) for p in sets[str(fold)]['Rq'])
        row = dict(fold=fold, method=ru['method'], lam=ru['lam'], seed=ru['seed'])
        data, _ = load_shen(reg, f['outer_evaluation'], obj['scaler'], 'all')
        y, rx = data['y'], data['rx']
        src, _ = load_shen(reg, f['inner_train'] + f['inner_validation'], obj['scaler'], 'train')

        # end-of-training rho_row diagnostic (source packets)
        pres = []
        with torch.no_grad():
            xs = torch.as_tensor(src['x'], device=device)
            for i in range(0, len(xs), 512):
                xb = xs[i:i + 512]
                h = F.relu(model.stem(F.pad(xb, (2, 3, 2, 3))))
                h = model.blocks(h).permute(0, 2, 3, 1).reshape(xb.shape[0], -1)
                fq = F.normalize(model.feature(h), p=2, dim=1, eps=1e-12)
                pres.append(F.relu(model.tx_input(fq)).float().cpu())
        pre = torch.cat(pres)
        Wt = model.classifier.weight.detach().float().cpu()
        rho = L._rho(Wt, L.class_disp(pre, torch.as_tensor(src['y']), torch.as_tensor(src['rx'])), 1e-4)
        row['rho_row_end'] = float(np.mean(rho.numpy()))

        logits = C.run_batches(model, data['x'], device)
        ms = {}
        for r in np.unique(rx):
            m = rx == r
            k = reg['shen_allowed'][int(r)]
            mm = metrics(torch.as_tensor(logits[m]).float(), torch.as_tensor(y[m]))
            ms[k] = mm
            RX.append(dict(fold=fold, method=ru['method'], lam=ru['lam'], seed=ru['seed'], rx=k, **mm))
        accs = [v['accuracy'] for v in ms.values()]
        row.update(acc=float(np.mean(accs)),
                   macro_f1=float(np.mean([v['macro_f1'] for v in ms.values()])),
                   worst_rx_acc=float(np.min(accs)),
                   pluto_acc=float(ms['shen_pluto_1_train']['accuracy']))

        # per RX x TX recall
        pred = logits.argmax(1)
        for r in np.unique(rx):
            k = reg['shen_allowed'][int(r)]
            for t in range(10):
                mt = (rx == r) & (y == t)
                if mt.any():
                    TX.append(dict(fold=fold, method=ru['method'], lam=ru['lam'], seed=ru['seed'],
                                   rx=k, t=t, recall=float((pred[mt] == t).mean())))

        # all 90 ordered-pair confusions + normalized margins
        Wnp = Wt.numpy()
        pc, npc, nm_mean, nm_q10 = [], [], [], []
        for r in np.unique(rx):
            key = reg['shen_allowed'][int(r)]
            for t in range(10):
                mt = y == t
                if not mt.any():
                    continue
                for u in range(10):
                    if u == t:
                        continue
                    marg = logits[mt, t] - logits[mt, u]
                    conf = float((pred[mt] == u).mean())
                    nmr = marg / (np.linalg.norm(Wnp[t] - Wnp[u]) + 1e-6)
                    inp = (t, u) in Pset
                    PR.append(dict(fold=fold, method=ru['method'], lam=ru['lam'], seed=ru['seed'],
                                   rx_key=key, t=t, u=u, conf=conf, nm_mean=float(nmr.mean()),
                                   nm_q10=float(np.quantile(nmr, 0.1)), in_P=inp))
                    (pc if inp else npc).append(conf)
                    if inp:
                        nm_mean.append(float(nmr.mean()))
                        nm_q10.append(float(np.quantile(nmr, 0.1)))
        row.update(P_conf=float(np.mean(pc)), nonP_conf=float(np.mean(npc)),
                   P_nm_mean=float(np.mean(nm_mean)), P_nm_q10=float(np.mean(nm_q10)))
        R.append(row)
        print(row['fold'], row['method'], row['seed'], 'acc=%.4f' % row['acc'],
              'rho=%.3f' % row['rho_row_end'], flush=True)

    pd.DataFrame(R).to_csv(out / 'runs.csv', index=False)
    pd.DataFrame(RX).to_csv(out / 'per_rx.csv', index=False)
    pd.DataFrame(TX).to_csv(out / 'per_rx_tx.csv', index=False)
    pd.DataFrame(PR).to_csv(out / 'pairs.csv', index=False)
    print('wrote', out)


if __name__ == '__main__':
    main()
