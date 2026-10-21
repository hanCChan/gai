#!/usr/bin/env python3
"""P0-A/B: pairwise decision-failure localization on the 30 frozen Shen models.

For every model (2 folds x 5 methods x 3 seeds), one GPU pass computes:
  P0-A  full 90-ordered-pair margin statistics on every source RX and every outer RX
        (mean, q10, q50, frac_neg, directional confusion conf_{t->u}), plus true
        10-class prediction accuracy so margins are never read without the full picture.
  P0-B  encoder-vs-head decomposition for the same pairs:
          - 512-d normalized feature space: class mean distance / within scatter (d_class),
            effective rank, per-pair discriminative-direction cosine (w_t - w_u via frozen head)
          - pre-head 128-d space (after tx_input+ReLU)
          - frozen head geometry: ||w_t - w_u||, bias differences
          - SOURCE-ONLY refit head (r1 recipe) evaluated on outer RX: does a plain CE head
            refit on source features reproduce the same directional collapses?
            (separates "features already collapsed" from "head boundary is fragile")

Source pools use ONLY the fitting packets (first 160 per TX per RX); the checkpoint-selection
validation packets (last 40) are excluded from risk rules, matching R0/R1 selection hygiene.
Outer RX labels are recorded but NO threshold in this script uses them.

    python3 run_p0a.py --out OUT [--shard i/n] [--limit N] [--device cuda:0]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import p0_common as C   # noqa: E402

C.add_root()

N = 10


def features_of(model, x, device):
    """(feats512, pre_head_128, logits) with bitwise-checked logits == model(x)."""
    import torch
    import torch.nn.functional as F
    feats512, pre, logits = [], [], []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            xb = torch.as_tensor(np.ascontiguousarray(x[i:i + 512]), device=device)
            h = F.relu(model.stem(F.pad(xb, (2, 3, 2, 3))))
            h = model.blocks(h).permute(0, 2, 3, 1).reshape(xb.shape[0], -1)
            f = F.normalize(model.feature(h), p=2, dim=1, eps=1e-12)
            p = F.relu(model.tx_input(f))
            feats512.append(f.cpu())
            pre.append(p.cpu())
            logits.append(model.classifier(p).cpu())
    f512 = torch.cat(feats512).numpy()
    p128 = torch.cat(pre).numpy()
    lg = torch.cat(logits).numpy()
    # bitwise check against the model's own forward (no reimplementation drift)
    ref = C.run_batches(model, x, device)
    assert np.array_equal(lg, ref), 'logit reimplementation mismatch'
    return f512, p128, lg


def head_geometry(model):
    W = model.classifier.weight.detach().cpu().numpy()      # (10, 128)
    b = model.classifier.bias.detach().cpu().numpy()         # (10,)
    return W, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    import torch
    device = torch.device(a.device)
    si, ns = (int(v) for v in a.shard.split('/'))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    reg = C.registration()
    all_jobs = C.jobs('shen')
    jobs = [j for k, j in enumerate(all_jobs) if k % ns == si]
    if a.limit:
        jobs = jobs[:a.limit]

    pair_rows, geom_rows, refit_rows, recall_rows = [], [], [], []
    t0 = time.perf_counter()
    for j in jobs:
        ckpt_obj, status = C.load_model(j['ckpt'])
        model, scaler = C.build_model(j, ckpt_obj)
        model = model.to(device)
        fold = j['fold']
        src, outer = C.shen_fold_pools(reg, fold, scaler, partition='train')
        srcv, _ = C.shen_fold_pools(reg, fold, scaler, partition='val')   # selection packets, margins only recorded
        keys = reg['shen_allowed']

        f512s, p128s, lgs = {}, {}, {}
        for tag, data in (('source_train', src), ('source_val', srcv), ('outer', outer)):
            f512s[tag], p128s[tag], lgs[tag] = features_of(model, data['x'], device)

        # ---- P0-A margins: source train, source val (diagnostic), outer ----
        for tag, data in (('source_train', src), ('source_val', srcv), ('outer', outer)):
            st = C.margin_stats(lgs[tag], data['y'], data['rx'])
            acc = float((lgs[tag].argmax(1) == data['y']).mean())
            recall_rows.append(dict(run_id=j['run_id'], fold=fold, method=j['method'], seed=j['seed'],
                                    pool=tag, accuracy=acc))
            for (r, t, u), s in st.items():
                pair_rows.append(dict(run_id=j['run_id'], fold=fold, method=j['method'], seed=j['seed'],
                                      pool=tag, rx=int(r), rx_key=keys[r], t=t, u=u, **s))

        # ---- P0-B geometry ----
        W, b = head_geometry(model)
        er512 = C.effective_rank(f512s['source_train'])
        er128 = C.effective_rank(p128s['source_train'])
        geom_rows.append(dict(run_id=j['run_id'], fold=fold, method=j['method'], seed=j['seed'],
                              eff_rank_512=er512, eff_rank_128=er128))
        for r in np.unique(src['rx']):
            for t in range(N):
                for u in range(t + 1, N):
                    d512 = C.class_stats(f512s['source_train'], src['y'], src['rx'] == r, t, u)
                    d128 = C.class_stats(p128s['source_train'], src['y'], src['rx'] == r, t, u)
                    wdir = W[t] - W[u]
                    geom_rows.append(dict(
                        run_id=j['run_id'], fold=fold, method=j['method'], seed=j['seed'],
                        pool='source_train', rx=int(r), rx_key=keys[r], t=t, u=u,
                        d_class_512=d512['d_class'], dist_512=d512['dist'], within_512=d512['within'],
                        d_class_128=d128['d_class'],
                        wdu=float(np.linalg.norm(wdir)),
                        bias_diff=float(b[t] - b[u])))
        for r in np.unique(outer['rx']):
            for t in range(N):
                for u in range(t + 1, N):
                    d512 = C.class_stats(f512s['outer'], outer['y'], outer['rx'] == r, t, u)
                    d128 = C.class_stats(p128s['outer'], outer['y'], outer['rx'] == r, t, u)
                    geom_rows.append(dict(
                        run_id=j['run_id'], fold=fold, method=j['method'], seed=j['seed'],
                        pool='outer', rx=int(r), rx_key=keys[r], t=t, u=u,
                        d_class_512=d512['d_class'], dist_512=d512['dist'], within_512=d512['within'],
                        d_class_128=d128['d_class'],
                        wdu=float(np.linalg.norm(W[t] - W[u])),
                        bias_diff=float(b[t] - b[u])))

        # ---- P0-B refit control: fresh CE head on source-train 512-d features ----
        import torch
        from torch import nn
        make = lambda: nn.Linear(512, N)
        h, info = C.refit_head(make, f512s['source_train'], src['y'], device)
        lr_out = C.run_batches(h, f512s['outer'], device)
        st = C.margin_stats(lr_out, outer['y'], outer['rx'])
        for (r, t, u), s in st.items():
            refit_rows.append(dict(run_id=j['run_id'], fold=fold, method=j['method'], seed=j['seed'],
                                   rx=int(r), rx_key=keys[r], t=t, u=u, **s, **info))
        refit_acc = float((lr_out.argmax(1) == outer['y']).mean())
        refit_rows.append(dict(run_id=j['run_id'], fold=fold, method=j['method'], seed=j['seed'],
                               rx=-1, rx_key='OVERALL', t=-1, u=-1, n=len(outer['y']),
                               mean=np.nan, q10=np.nan, q50=np.nan, frac_neg=np.nan,
                               conf=refit_acc, **info))
        del model
        print(f'done {j["run_id"]} (+{time.perf_counter()-t0:.0f}s)', flush=True)

    import pandas as pd
    pd.DataFrame(pair_rows).to_csv(out / f'pairs_shard{si}.csv', index=False)
    pd.DataFrame(geom_rows).to_csv(out / f'geometry_shard{si}.csv', index=False)
    pd.DataFrame(refit_rows).to_csv(out / f'refit_shard{si}.csv', index=False)
    pd.DataFrame(recall_rows).to_csv(out / f'recall_shard{si}.csv', index=False)
    C.dump_json(out / f'manifest_shard{si}.json', dict(
        shard=a.shard, jobs=[j['run_id'] for j in jobs], seconds=time.perf_counter() - t0,
        code_sha256=C.manifest([HERE / 'p0_common.py', Path(__file__)]),
        source_partition='first160', refit=C.REFIT))
    print('wrote', out)


if __name__ == '__main__':
    main()
