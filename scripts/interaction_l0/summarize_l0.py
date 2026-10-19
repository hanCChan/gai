#!/usr/bin/env python3
"""L0-C: frozen decision rules over run_l0.py (+ optional atlas_l0.py) outputs. CPU only.

Verdicts use three levels: SUPPORTED / INCONCLUSIVE / UNSUPPORTED_FOR_CURRENT_REPRESENTATION.
Development RX were viewed before: every verdict is exploratory evidence for choosing the next
experiment, never a final claim. Thresholds below are fixed before any server output exists.

    python summarize_l0.py --l0 OUT [--atlas OUT_ATLAS]
"""
import argparse, json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from link_core import spearman

PRIMARY = 'wisig/official/r01'
TARGET = 'acc_internal_unseen_date'
K = 8
RULES = dict(
    S_within_seed_partial_rho_max=-0.4,          # link_ratio vs target after seed means and controls removed
    S_seeds_negative_min_fraction=2 / 3,
    I_median_link_specific_gain_pp_min=1.0,      # gain(link) - max(gain(rx), gain(random)), both heads
    I_positive_fraction_min=2 / 3,
    I_source_drop_pp_max=2.0,
    I_logit_norm_ratio=(0.8, 1.25),
    cond_better_margin_pp=1.0,
    cfo_tx_moved_min=0.3, cfo_rx_flip_min=0.10, cfo_lift_min=0.15,
    cfo_tx_moved_none=0.1, cfo_lift_none=0.05,
    tone_link_flip_min=0.10, tone_link_over_rx_min=1.5, tone_link_flip_none=0.03,
    snr_flip_min=0.10)


def ranks(v):
    return pd.Series(np.asarray(v, float)).rank().to_numpy()


def partial_rank_corr(x, y, controls, groups):
    """Spearman partial correlation after removing group (seed) means and linear rank controls."""
    def prep(v):
        r = ranks(v); return r - pd.Series(r).groupby(groups).transform('mean').to_numpy()
    X, Y = prep(x), prep(y)
    C = np.column_stack([prep(c) for c in controls] + [np.ones(len(X))]) if controls else np.ones((len(X), 1))
    res = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    rx_, ry_ = res(X), res(Y)
    return float(np.corrcoef(rx_, ry_)[0, 1]) if rx_.std() > 0 and ry_.std() > 0 else np.nan


def statistical(R):
    P = R[R.group == PRIMARY].copy()
    out = dict(n_runs=len(P), n_seeds=int(P.seed.nunique()), n_methods=int(P.method.nunique()))
    if len(P) < 6:
        return out, False
    seed_mean = P.groupby('seed')[[TARGET, 'feat_link_ratio']].mean()
    out['between_seed_rho_n3'] = spearman(seed_mean.feat_link_ratio, seed_mean[TARGET])
    out['seed_means'] = seed_mean.round(4).to_dict()
    g = P.seed.to_numpy()
    for name, ctrl in (('none', []), ('acc_source', [P.acc_source]), ('acc_source+separability', [P.acc_source, P.feat_separability])):
        out[f'within_seed_partial_rho_link_ratio|{name}'] = partial_rank_corr(P.feat_link_ratio, P[TARGET], ctrl, g)
    for alt in ('acc_source', 'feat_separability', 'feat_loro_proto', 'feat_rx_share'):
        out[f'within_seed_rho_{alt}'] = partial_rank_corr(P[alt], P[TARGET], [], g)
    # consistency: the method-level association must hold inside most seeds, not one seed
    per_seed = {int(k): spearman(v.feat_link_ratio, v[TARGET]) for k, v in P.groupby('seed')}
    out['per_seed_rho_over_methods'] = per_seed
    neg = sum(bool(r < 0) for r in per_seed.values())  # NaN (constant input) counts as not negative
    rho = out['within_seed_partial_rho_link_ratio|acc_source+separability']
    # between-seed (n=3) is descriptive only: the seed-level effect needs the many-seed factorial (P1)
    ok = bool(np.isfinite(rho) and rho <= RULES['S_within_seed_partial_rho_max'] and neg >= RULES['S_seeds_negative_min_fraction'] * len(per_seed))
    out['S_pass'] = ok
    return out, ok


def intervention(E, R):
    P = E[(E.group == PRIMARY)]
    out = {}
    if P.empty:
        return out, False, {}
    base = P[P.kind == 'none'].set_index('run_id')
    flags = {}
    gains = {}
    for head in ('frozen', 'refit'):
        col = f'{head}_acc_internal_unseen_date'
        g = {}
        for kind in ('link', 'rx', 'cond', 'random'):
            sub = P[(P.kind == kind) & (P.k == K)].groupby('run_id')[col].mean()
            g[kind] = 100 * (sub - base.loc[sub.index, col])
        spec = g['link'] - np.maximum(g['rx'], g['random'])
        gains[head] = g
        out[f'{head}_median_gain_pp'] = {k: float(v.median()) for k, v in g.items()}
        out[f'{head}_median_link_specific_pp'] = float(spec.median())
        out[f'{head}_link_specific_positive_fraction'] = float((spec > 0).mean())
        out[f'{head}_cond_minus_link_median_pp'] = float((g['cond'] - g['link']).median())
    # intervention strength and identity retention per arm (descriptive; same rank != same strength)
    for col in ('feature_change_DT', 'tx_energy_retained', 'link_energy_retained', 'rx_energy_retained'):
        if col in P.columns:
            out[f'median_{col}_k{K}'] = {kind: float(P[(P.kind == kind) & (P.k == K)][col].median()) for kind in ('link', 'rx', 'cond', 'random')}
    link = P[(P.kind == 'link') & (P.k == K)].set_index('run_id')
    flags['source_drop_pp'] = float(100 * (base.loc[link.index, 'frozen_acc_source'] - link.frozen_acc_source).median())
    ratio = (link.frozen_mean_logit_norm / base.loc[link.index, 'frozen_mean_logit_norm']).median()
    flags['logit_norm_ratio'] = float(ratio)
    for k in (4, 16):
        sub = {kind: P[(P.kind == kind) & (P.k == k)].groupby('run_id')['frozen_acc_internal_unseen_date'].mean() for kind in ('link', 'rx', 'random')}
        if all(len(v) for v in sub.values()):
            spec = 100 * (sub['link'] - np.maximum(sub['rx'], sub['random']))
            out[f'frozen_median_link_specific_pp_k{k}'] = float(spec.median())
    lo, hi = RULES['I_logit_norm_ratio']
    artifact = not (flags['source_drop_pp'] <= RULES['I_source_drop_pp_max'] and lo <= ratio <= hi)
    ok = all(out[f'{h}_median_link_specific_pp'] >= RULES['I_median_link_specific_gain_pp_min'] and
             out[f'{h}_link_specific_positive_fraction'] >= RULES['I_positive_fraction_min'] for h in ('frozen', 'refit'))
    out.update(flags=flags, artifact_flag=bool(artifact), I_pass=bool(ok),
               cond_better_than_link=bool(min(out['frozen_cond_minus_link_median_pp'], out['refit_cond_minus_link_median_pp'])
                                          >= RULES['cond_better_margin_pp']))
    return out, ok, gains


def physics(S, A, R):
    out = {}
    if S.empty:
        return out
    P = S[S.group == PRIMARY]
    med = P.groupby('intervention')[['flip_rate', 'moved_to_target', 'acc_drop_pp']].median()
    out['median_by_intervention'] = med.round(4).to_dict(orient='index')
    flip = med.flip_rate
    cfo_rx_flip = float(flip[['cfo_rx_+1sd', 'cfo_rx_-1sd']].mean())
    lift = {}
    if not A.empty:
        a = A[(A.group == PRIMARY) & (A.role == 'internal_unseen_date')]
        for cue in ('cfo', 'tone'):
            s = a[a.cue == cue]; lift[cue] = float((s.agree_given_cnn_wrong - s.agree_given_cnn_wrong_null).median())
            out[f'{cue}_rule_accuracy_internal_unseen'] = float(s.cue_rule_accuracy.median())
    out['error_lift'] = lift
    moved_cfo = float(med.loc['cfo_tx_swap', 'moved_to_target'])
    c = RULES
    if moved_cfo >= c['cfo_tx_moved_min'] and cfo_rx_flip >= c['cfo_rx_flip_min'] and lift.get('cfo', 0) >= c['cfo_lift_min']:
        out['H_CFO_rx_offset'] = 'SUPPORTED'
    elif moved_cfo < c['cfo_tx_moved_none'] and lift.get('cfo', 0) < c['cfo_lift_none']:
        out['H_CFO_rx_offset'] = 'UNSUPPORTED_FOR_CURRENT_REPRESENTATION'
    else:
        out['H_CFO_rx_offset'] = 'INCONCLUSIVE'
    tl, tr = float(flip['tone_link_swap']), float(flip['tone_rx_swap'])
    if tl >= c['tone_link_flip_min'] and tl >= c['tone_link_over_rx_min'] * max(tr, 1e-9):
        out['H_tone_link_sensitivity'] = 'SUPPORTED'
    elif tl < c['tone_link_flip_none']:
        out['H_tone_link_sensitivity'] = 'UNSUPPORTED_FOR_CURRENT_REPRESENTATION'
    else:
        out['H_tone_link_sensitivity'] = 'INCONCLUSIVE'
    out['SNR_link_sensitive'] = bool(float(flip['snr_link_swap']) >= c['snr_flip_min'])
    out['cfo_rx_flip_1sd'] = cfo_rx_flip
    # which sensitivity tracks run-level (within-seed) variation of the target?
    Rp = R[R.group == PRIMARY].set_index('run_id')
    piv = P.pivot_table(index='run_id', columns='intervention', values='flip_rate')
    piv = piv.join(Rp[[TARGET, 'seed']], how='inner')
    out['within_seed_rho_target_vs_flip'] = {col: partial_rank_corr(piv[col], piv[TARGET], [], piv.seed.to_numpy())
                                             for col in piv.columns if col not in (TARGET, 'seed')}
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--l0', required=True); p.add_argument('--atlas')
    a = p.parse_args(); d = Path(a.l0)
    read = lambda n: pd.read_csv(d / n) if (d / n).exists() and (d / n).stat().st_size > 1 else pd.DataFrame()
    R, E, S, A = read('run_metrics.csv'), read('erasure.csv'), read('sensitivity.csv'), read('attribution.csv')
    ens, ov = read('ensemble.csv'), read('error_overlap.csv')
    stat, s_ok = statistical(R)
    inter, i_ok, _ = intervention(E, R)
    phys = physics(S, A, R)
    if s_ok and i_ok and not inter.get('artifact_flag', True):
        verdict = 'SUPPORTED'
    elif s_ok or i_ok:
        verdict = 'INCONCLUSIVE'
    else:
        verdict = 'UNSUPPORTED_FOR_CURRENT_REPRESENTATION'
    summary = dict(rules=RULES, primary_group=PRIMARY, target=TARGET, statistical=stat, intervention=inter,
                   physical=phys, interaction_shortcut_verdict=verdict,
                   scope='exploratory; development RX already viewed; frozen-network linear probes only')
    if not ens.empty:
        e = ens[(ens.group == PRIMARY) & (ens.role == 'internal_unseen_date')]
        summary['ensemble'] = e[['method', 'single_mean', 'single_min', 'single_max_posthoc_oracle', 'prob_average', 'majority_vote']].round(4).to_dict(orient='records')
    if not ov.empty:
        o = ov[(ov.group == PRIMARY) & (ov.role == 'internal_unseen_date')]
        summary['error_overlap_median'] = o[['jaccard', 'joint_over_independent']].median().round(3).to_dict()
    if a.atlas:
        summary['atlas'] = json.loads((Path(a.atlas) / 'atlas.json').read_text())
    (d / 'summary.json').write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(dict(verdict=verdict, S_pass=stat.get('S_pass'), I_pass=inter.get('I_pass'),
                          artifact=inter.get('artifact_flag'), H_CFO=phys.get('H_CFO_rx_offset'),
                          H_tone_link=phys.get('H_tone_link_sensitivity')), indent=2))


if __name__ == '__main__':
    main()
