#!/usr/bin/env python3
"""WiSig single-job cost probe (PROTOCOL §5). Timing only: 3 epochs of CE and of rxlogit on fold W0's
10 source RX, no checkpoint is kept, nothing is evaluated on held/internal RX. Writes seconds/epoch so
the formal budget can be stated before requesting approval.

    python3 wisig_cost_probe.py --out reports/receiver_p2/wisig_probe.json [--epochs 3]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'receiver_p0')]
import p0_common as C   # noqa: E402
import p2_losses as L   # noqa: E402

C.add_root()


def main():
    import torch
    import torch.nn.functional as F
    from wisig_w01_common import OfficialCNN
    from receiver_r01.core import balanced_schedule, predict, metrics
    from receiver_r01.data import wisig as load_wisig, groups

    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=3)
    a = ap.parse_args()
    reg = C.registration()
    device = torch.device('cuda:0')
    # P2D fix 3: the shared wisig() loader does NOT filter encoder TX; W0 convention (and
    # the protocol) trains on TX 0-7 only. Filter here so the 8-class head matches labels.
    train_all = load_wisig(reg, reg['wisig_inner_train'], [0, 1], [0, 120])
    keep = train_all['y'] < 8
    train = {k: v[keep] for k, v in train_all.items() if k != 'stats'}
    val_all = load_wisig(reg, reg['wisig_inner_train'], [0, 1], [120, 160])
    keepv = val_all['y'] < 8
    val = {k: v[keepv] for k, v in val_all.items() if k != 'stats'}
    assert not np.any(train['y'] >= 8) and not np.any(val['y'] >= 8)
    schedule = balanced_schedule(groups(train), a.epochs, reg['search_seed'])
    x = torch.as_tensor(train['x'], device=device)
    y = torch.as_tensor(train['y'], device=device)
    rx = torch.as_tensor(train['rx'], device=device)
    vx = torch.as_tensor(val['x'], device=device)
    vy = torch.as_tensor(val['y'], device=device)
    vrx = torch.as_tensor(val['rx'], device=device)
    res = dict(train_packets=int(len(y)), batches_per_epoch=int(schedule.shape[1]),
               params=None, runs={})
    for method in ('ce', 'rxlogit'):
        torch.manual_seed(reg['search_seed'])
        torch.cuda.manual_seed_all(reg['search_seed'])
        torch.use_deterministic_algorithms(True)
        # P2D fix 1: 8-class head to match the W0 convention (OfficialCNN() default) and the
        # protocol's TX0-7 training task - the 10-class head added two never-trained rows.
        model = OfficialCNN().to(device)
        res['params'] = int(sum(p.numel() for p in model.parameters()))
        opt = torch.optim.Adam(model.parameters(), lr=5e-4, eps=1e-7)
        ep = []
        for epoch in range(a.epochs):
            t0 = time.perf_counter()
            for ids in schedule[epoch]:
                ix = torch.tensor(ids, device=device)
                model.train()
                opt.zero_grad(set_to_none=True)
                h = model.encode(x[ix])
                logits = model.head(model.drop(h))
                ce = F.cross_entropy(logits, y[ix])
                if method == 'rxlogit':
                    W = model.head.weight
                    # P2D fix 2: same base regularizer as the CE arm in BOTH branches
                    loss = ce + model.regularization() + L.rxlogit(h, W, y[ix], rx[ix])
                else:
                    loss = ce + model.regularization()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                opt.step()
            torch.cuda.synchronize()
            vce = []
            model.eval()
            with torch.no_grad():
                for r in torch.unique(vrx):
                    m = vrx == r
                    lg = predict(model, vx[m]).cpu()
                    # receiver-equal CE; direct cross_entropy (metrics() assumes 10 classes)
                    vce.append(float(F.cross_entropy(lg, vy[m].cpu())))
            ep.append(time.perf_counter() - t0)
            print(method, 'epoch', epoch + 1, 'sec=%.1f' % ep[-1], 'val_CE=%.4f' % float(np.mean(vce)), flush=True)
        res['runs'][method] = dict(sec_per_epoch=ep,
                                    mean_sec_per_epoch=float(np.mean(ep[1:])) if len(ep) > 1 else float(ep[0]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    s = max(v['mean_sec_per_epoch'] for v in res['runs'].values())
    print(f"mean {s:.1f} s/epoch -> 50-ep search fit ≈ {50 * s / 60:.1f} min, "
          f"100-ep formal fit ≈ {100 * s / 60:.1f} min; 30 search + 36 formal ≈ "
          f"{(30 * 50 + 36 * 100) * s / 3600:.1f} job-h")


if __name__ == '__main__':
    main()
