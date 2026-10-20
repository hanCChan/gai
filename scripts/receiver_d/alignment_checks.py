#!/usr/bin/env python3
"""Post-hoc geometry checks for the D report (model-free; Shen part joins saved predictions).

1. WiSig: is the dominant source receiver-difference axis of the 12-tone shape aligned with
   transmitter identity? Share of TX-effect energy on that axis vs random unit directions in the
   centred 11-dim tone-shape space (permutation-free null, 20000 draws).
2. Shen: does a receiver's offset push packets along transmitter-identity axes? Nearest-prototype
   collapse score per ordered pair (u -> w) at each outer RX, in the in-band Strict CIS map:
       s(u->w) = o_u . (a_w - a_u) / (0.5 ||a_w - a_u||^2),  s > 1 predicts u -> w
   with o_u = M[u,r] - g - a_u (uses target labels: diagnostic) or the label-free o = mean(r) - g.
   Spearman against the observed CE / SWAD confusion over the 90 ordered pairs.

    python3 alignment_checks.py --root /workspace/rl1/gai_retrain --out OUT
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'interaction_l0')]
import d_common as C  # noqa: E402
import rxphys as R  # noqa: E402
from physics import stf_descriptors, to_complex  # noqa: E402

SEEDS = (20261013, 20261014, 20261015)


def wisig_axis(reg):
    from receiver_r01.data import wisig
    fit = wisig(reg, reg['wisig']['roles']['source'], [0, 1], [0, 120])
    d = stf_descriptors(to_complex(fit['x']))
    tpl = R.response_templates(d['tone_shape'], d['cfo_hz'], fit['y'], fit['rx'], fit['day'])
    a, b = R.centre_shape(tpl['a']), R.centre_shape(tpl['b'])
    sb = np.linalg.svd(b - b.mean(0), full_matrices=False)
    ub = sb[2][0]
    share = float(np.sum((a @ ub) ** 2) / np.sum(a ** 2))
    rng = np.random.default_rng(20261019)
    P = np.eye(12) - 1 / 12
    null = []
    for _ in range(20000):
        v = P @ rng.normal(size=12)
        v /= np.linalg.norm(v)
        null.append(np.sum((a @ v) ** 2) / np.sum(a ** 2))
    null = np.array(null)
    sa = np.linalg.svd(a - a.mean(0), compute_uv=False)
    order = np.argsort(R.TONE_HZ)
    return dict(rx_axis_variance_share=float(sb[1][0] ** 2 / (sb[1] ** 2).sum()), tx_energy_on_rx_axis=share,
                null_median=float(np.median(null)), null_p95=float(np.percentile(null, 95)),
                p_value=float((null >= share).mean()), tx_top_axis_share=float(sa[0] ** 2 / (sa ** 2).sum()),
                rx_axis_by_tone_freq_mhz={f'{R.TONE_HZ[k] / 1e6:+.2f}': float(ub[k]) for k in order})


def shen_collapse(reg, root):
    keys = reg['shen_allowed']
    inb = np.abs(np.arange(52) - 26) <= 8
    V, Y = {}, {}
    for k in keys:
        x = np.load(Path(reg['shen_cache']) / k / 'strict_cis.npy', mmap_mode='r')
        V[k] = np.asarray(x[:, 0][:, inb, :], np.float64).reshape(len(x), -1)
        Y[k] = np.load(Path(reg['shen_cache']) / k / 'labels.npy')
    M = {k: np.stack([V[k][Y[k] == t].mean(0) for t in range(10)]) for k in keys}
    ev = root / 'outputs/receiver_r01/evaluation'

    def conf(fold, m, rxi):
        Cm = np.zeros((10, 10))
        for s in SEEDS:
            d = np.load(ev / f'shen_rank16_CV{fold}_{m}_seed{s}_source_pool_outer_RX.npz')
            sel = d['rx'] == rxi
            np.add.at(Cm, (d['y'][sel], d['prediction'][sel]), 1)
        return Cm / Cm.sum(1, keepdims=True)

    rows = []
    for fold, fd in enumerate(reg['shen_folds']):
        S = fd['inner_train'] + fd['inner_validation']
        MS = np.stack([M[k] for k in S], 1)
        g = MS.mean((0, 1))
        a = MS.mean(1) - g
        for k in fd['outer_evaluation']:
            o_full = M[k] - g - a
            o_free = V[k].mean(0) - g
            Cce, Csw = conf(fold, 'ce', keys.index(k)), conf(fold, 'swad', keys.index(k))
            for u in range(10):
                for w in range(10):
                    if u != w:
                        dw = a[w] - a[u]
                        den = 0.5 * dw @ dw
                        rows.append(dict(fold=fold, rx=k, u=u, w=w, s_full=o_full[u] @ dw / den, s_free=o_free @ dw / den,
                                         ce=Cce[u, w], swad=Csw[u, w]))
    D = pd.DataFrame(rows)
    per = []
    for k, g_ in D.groupby('rx'):
        per.append(dict(rx=k, rho_full_ce=C.spearman(g_.s_full, g_.ce), rho_free_ce=C.spearman(g_.s_free, g_.ce),
                        rho_free_swad=C.spearman(g_.s_free, g_.swad), max_s_free=float(g_.s_free.max()),
                        max_s_full=float(g_.s_full.max()),
                        top_ce=[f'{int(r.u)}->{int(r.w)}' for r in g_.sort_values('ce', ascending=False).head(4).itertuples()]))
    return D, pd.DataFrame(per), dict(pooled_rho_full_ce=C.spearman(D.s_full, D.ce), pooled_rho_free_ce=C.spearman(D.s_free, D.ce),
                                      pooled_rho_free_swad=C.spearman(D.s_free, D.swad),
                                      pooled_rho_free_swad_minus_ce=C.spearman(D.s_free, D.swad - D.ce))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root = C.add_root(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = json.loads((root / 'configs/receiver_r01/registration.json').read_text())
    w = wisig_axis(reg)
    D, per, pooled = shen_collapse(reg, root)
    D.to_csv(out / 'shen_collapse_score.csv', index=False)
    per.to_csv(out / 'shen_collapse_by_rx.csv', index=False)
    S = dict(wisig_axis=w, shen_collapse=pooled)
    (out / 'alignment.json').write_text(json.dumps(C.jsonable(S), indent=1))
    print(json.dumps(C.jsonable(S), indent=1))
    print(per.round(3).to_string())


if __name__ == '__main__':
    main()
