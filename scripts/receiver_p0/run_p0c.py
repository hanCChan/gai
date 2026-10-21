#!/usr/bin/env python3
"""P0-C: can source-RX pair risk predict unseen-RX pair failure? (pure post-processing on P0-A CSVs)

Frozen rules (docs/receiver_p0/PROTOCOL_P0.md):
  Source risk per (fold, t, u), computed ONLY on source_train margins of that fold:
    R_q     = - median_r q_{r,t,u}                       low worst-case margin
    R_spread= max_r q_{r,t,u} - min_r q_{r,t,u}          cross-source-RX instability
    R_neg   = max_r frac_neg_{r,t,u}                     any source RX already flipping
    R_conf  = mean_r conf_{r,t,u}                        plain source difficulty baseline
  Outer failure per (fold, t, u): mean over outer RX / seeds / methods of conf_{t->u},
    plus per-RX breakdown; comparison arms: per-seed and per-method.

Questions answered (all pairs, 90 ordered per fold; Pluto cases only reported separately):
  1. Spearman/rank-IOU between each source risk and outer confusion, per fold and pooled
     (with permutation p over pairs, and a leave-Pluto-out check).
  2. Increment over the plain source difficulty baseline R_conf (partial Spearman,
     and AUC of "top-K source-risk pairs capture top-K outer-failure pairs").
  3. Stability across seeds and the two folds (is the risk ranking the same?).
  4. Capacity control: does d_class_512 on source (P0-B) add predictive value?
Output: risk_correlation.csv, top_pairs.csv, answers.json.

    python3 run_p0c.py --p0a DIR --out DIR
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import p0_common as C   # noqa: E402

SEED = 20261021


def perm_spearman(x, y, n_perm=20000, rng=None):
    """Spearman with permutation p-value."""
    from scipy import stats
    rho = stats.spearmanr(x, y).statistic
    rng = rng or np.random.default_rng(SEED)
    y = np.asarray(y)
    count = 0
    for _ in range(n_perm):
        r = stats.spearmanr(x, rng.permutation(y)).statistic
        if abs(r) >= abs(rho):
            count += 1
    return float(rho), (count + 1) / (n_perm + 1)


def topk_overlap(a, b, k=10):
    """|top-k(a) ∩ top-k(b)| / k with equal-tie handling by ranking."""
    ia = set(pd.Series(-a).rank(method='first').sort_values().index[:k])
    ib = set(pd.Series(-b).rank(method='first').sort_values().index[:k])
    return len(ia & ib) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--p0a', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=10)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = []
    for f in sorted(Path(a.p0a).glob('pairs_shard*.csv')):
        pairs.append(pd.read_csv(f))
    pairs = pd.concat(pairs, ignore_index=True)
    geom = pd.concat([pd.read_csv(f) for f in sorted(Path(a.p0a).glob('geometry_shard*.csv'))], ignore_index=True)

    src = pairs[pairs.pool == 'source_train']
    outp = pairs[pairs.pool == 'outer']
    # aggregate outer: mean over seeds and outer RX within (fold, method, t, u); then over methods
    oagg = outp.groupby(['fold', 'method', 't', 'u']).agg(
        outer_conf=('conf', 'mean'), outer_q10=('q10', 'mean'), outer_frac_neg=('frac_neg', 'mean')).reset_index()
    omean = oagg.groupby(['fold', 't', 'u']).agg(
        outer_conf=('outer_conf', 'mean'), outer_conf_max=('outer_conf', 'max')).reset_index()

    sagg = src.groupby(['fold', 'run_id', 'seed', 't', 'u']).agg(
        q_med=('q10', 'median'), q_min=('q10', 'min'), q_max=('q10', 'max'),
        neg_max=('frac_neg', 'max'), conf_mean=('conf', 'mean')).reset_index()
    # risk definitions at the (fold, seed-run) level, then seed-mean for the frozen rule
    sagg['R_q'] = -sagg.q_med
    sagg['R_spread'] = sagg.q_max - sagg.q_min
    sagg['R_neg'] = sagg.neg_max
    sagg['R_conf'] = sagg.conf_mean
    sseed = sagg.groupby(['fold', 't', 'u']).agg(
        R_q=('R_q', 'mean'), R_spread=('R_spread', 'mean'), R_neg=('R_neg', 'mean'),
        R_conf=('R_conf', 'mean'), R_q_std=('R_q', 'std')).reset_index()

    m = omean.merge(sseed, on=['fold', 't', 'u'])
    # capacity: source d_class (mean over source RX, per run) as control
    gg = geom[(geom.pool == 'source_train')]
    gg = pd.concat([gg, gg.rename(columns={'t': 'u', 'u': 't'})], ignore_index=True)   # symmetrize: d_class(t,u)==d_class(u,t)
    gsrc = gg.groupby(['fold', 'run_id', 't', 'u']).d_class_512.mean().reset_index()
    gcap = gsrc.groupby(['fold', 't', 'u']).d_class_512.mean().reset_index().rename(columns={'d_class_512': 'src_dclass'})
    m = m.merge(gcap, on=['fold', 't', 'u'])

    rows = []
    for fold in sorted(m.fold.unique()):
        d = m[m.fold == fold]
        n = len(d)
        for risk in ('R_q', 'R_spread', 'R_neg', 'R_conf', 'src_dclass'):
            rho, p = perm_spearman(d[risk].values, d.outer_conf.values)
            rows.append(dict(fold=fold, risk=risk, n=n, spearman=rho, p=p))
        # partial Spearman of R_q / R_spread over R_conf
        from scipy import stats
        for risk in ('R_q', 'R_spread', 'R_neg'):
            pr = stats.spearmanr(d[risk], d.outer_conf).statistic
            # rank-partial via residuals of ranks
            rx = stats.rankdata(d[risk]); ry = stats.rankdata(d.outer_conf); rc = stats.rankdata(d.R_conf)
            bx = np.polyfit(rc, rx, 1); by = np.polyfit(rc, ry, 1)
            rho_p = stats.spearmanr(rx - np.polyval(bx, rc), ry - np.polyval(by, rc)).statistic
            rows.append(dict(fold=fold, risk=risk + '|R_conf', n=n, spearman=rho_p, p=np.nan))
        for risk in ('R_q', 'R_conf'):
            rows.append(dict(fold=fold, risk=f'top{a.k}_iou_' + risk, n=n,
                             spearman=topk_overlap(d[risk].values, d.outer_conf.values, a.k), p=np.nan))
    corr = pd.DataFrame(rows)
    corr.to_csv(out / 'risk_correlation.csv', index=False)

    # per-pair table ranked by frozen source rule R_q (combined z-score across risks not allowed;
    # the rule is: rank by R_q; report R_spread and R_conf alongside)
    m['risk_rank_R_q'] = m.groupby('fold').R_q.rank(ascending=False, method='first')
    m = m.sort_values(['fold', 'risk_rank_R_q'])
    m.to_csv(out / 'top_pairs.csv', index=False)

    # Pluto historical cases: where do they sit in the source ranking?
    hist = m[(m.fold == 0) & (((m.t == 1) & (m.u == 0)) | ((m.t == 3) & (m.u == 2)) | ((m.t == 8) & (m.u == 7)) | ((m.t == 4) & (m.u == 1)))]
    hist_cases = hist[['fold', 't', 'u', 'R_q', 'R_spread', 'R_conf', 'outer_conf', 'risk_rank_R_q']]

    answers = dict(
        per_fold_correlation=corr.to_dict('records'),
        pluto_cases=hist_cases.to_dict('records'),
        seed_stability=None,
    )
    # seed stability: rank correlation of R_q between seed pairs
    piv = sagg.pivot_table(index=['fold', 't', 'u'], columns='seed', values='R_q')
    from scipy import stats
    seed_rhos = []
    cols = [c for c in piv.columns if len(piv[c].dropna()) > 0]
    for i in range(len(cols)):
        for k in range(i + 1, len(cols)):
            sub = piv[[cols[i], cols[k]]].dropna()
            seed_rhos.append(dict(seed_a=int(cols[i]), seed_b=int(cols[k]),
                                  spearman=float(stats.spearmanr(sub.iloc[:, 0], sub.iloc[:, 1]).statistic)))
    answers['seed_stability'] = seed_rhos

    # method contrast: does SWAD hurt the SAME pairs that CE already finds hard?
    ce = oagg[oagg.method == 'ce'].set_index(['fold', 't', 'u']).outer_conf
    sw = oagg[oagg.method == 'swad'].set_index(['fold', 't', 'u']).outer_conf
    df = pd.concat([ce.rename('ce'), sw.rename('swad')], axis=1).dropna()
    answers['swad_vs_ce_pair_level'] = dict(
        spearman_delta=float(stats.spearmanr(df.ce, df.swad - df.ce).statistic),
        n_pairs_hurt=(df.swad - df.ce > 0.05).sum(),
        n_pairs_helped=(df.swad - df.ce < -0.05).sum(),
        mean_abs_delta=float((df.swad - df.ce).abs().mean()))
    C.dump_json(out / 'answers.json', answers)
    print('wrote', out)
    print(corr.to_string(index=False))


if __name__ == '__main__':
    main()
