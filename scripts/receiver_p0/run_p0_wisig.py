#!/usr/bin/env python3
"""WiSig mirror of P0-A: pairwise margins on the frozen WiSig official/resnet CE models.

Constraints (PROTOCOL_P0.md):
  - Only the 3 CE official + 3 CE resnet formal checkpoints (D-phase pool) are used;
    all 10 TX classes are supported on every development RX/date cell.
  - Pools: source RX, dates 0-1, positions 120-199 (positions 120-159 were the R0/R1
    model-selection pool; positions 160-199 the L0/D evaluation pool; BOTH are development
    data, so both may be analyzed, but rows are tagged so no selection-contamination
    illusion arises), and internal unseen RX dates 0-1 positions 0-199. Date 2 and the
    final 8 RX are NEVER read.
  - Same margin statistics as Shen P0-A; source-risk predictability computed the same way
    (per (t,u): median source q10, spread) against internal-RX confusion.

    python3 run_p0_wisig.py --out DIR [--device cuda:0]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import p0_common as C   # noqa: E402

C.add_root()


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    reg = C.registration()
    from receiver_r01.data import wisig as load_wisig
    src = load_wisig(reg, reg['wisig']['roles']['source'], [0, 1], [120, 200])
    internal = load_wisig(reg, reg['wisig']['roles']['internal_rx'], [0, 1], [0, 200])
    names = reg['wisig']['rx_names']

    pair_rows = []
    for j in C.jobs('wisig'):
        if j['method'] != 'ce':
            continue
        obj, status = C.load_model(j['ckpt'])
        model, scaler_unused = C.build_model(j, obj)
        model = model.to(device).eval()
        for tag, data in (('source', src), ('internal', internal)):
            logits = C.run_batches(model, data['x'], device)
            st = C.margin_stats(logits, data['y'], data['rx'])
            for (r, t, u), s in st.items():
                pair_rows.append(dict(run_id=j['run_id'], backbone=j['backbone'], seed=j['seed'],
                                      pool=tag, rx=int(r), rx_key=names[r], t=t, u=u, **s))
        print('done', j['run_id'], flush=True)
        del model
    df = pd.DataFrame(pair_rows)
    df.to_csv(out / 'wisig_pairs.csv', index=False)

    # risk analysis mirror (source -> internal)
    from scipy import stats
    srcm = df[df.pool == 'source'].groupby(['backbone', 'seed', 't', 'u']).agg(
        q_med=('q10', 'median'), q_min=('q10', 'min'), q_max=('q10', 'max'), conf=('conf', 'mean')).reset_index()
    srcm['R_q'] = -srcm.q_med
    srcm['R_spread'] = srcm.q_max - srcm.q_min
    inm = df[df.pool == 'internal'].groupby(['backbone', 't', 'u']).conf.mean().reset_index(name='internal_conf')
    m = srcm.groupby(['backbone', 't', 'u']).agg(R_q=('R_q', 'mean'), R_spread=('R_spread', 'mean'),
                                                 R_conf=('conf', 'mean')).reset_index().merge(inm, on=['backbone', 't', 'u'])
    rows = []
    for bb in m.backbone.unique():
        d = m[m.backbone == bb]
        for risk in ('R_q', 'R_spread', 'R_conf'):
            rho, p = stats.spearmanr(d[risk], d.internal_conf)
            rows.append(dict(backbone=bb, risk=risk, n=len(d), spearman=float(rho), p=float(p)))
    pd.DataFrame(rows).to_csv(out / 'wisig_risk.csv', index=False)
    C.dump_json(out / 'manifest.json', dict(code_sha256=C.manifest([HERE / 'p0_common.py', Path(__file__)]),
                                            jobs=sorted(df.run_id.unique().tolist())))
    print('wrote', out)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
