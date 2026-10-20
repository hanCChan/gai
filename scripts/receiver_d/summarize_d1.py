#!/usr/bin/env python3
"""Apply the frozen D1 rules (docs/receiver_d/PROTOCOL_D.md) to one or more run_d1 output shards.

    python3 summarize_d1.py --shards OUT/shard0 OUT/shard1 ... --out OUT/merged
Writes summary.json and SUMMARY.md (tables for the report); prints the verdicts.
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

R1_RULES = dict(nonfinite=0, periodicity_after_min=0.95, cfo_abs_hz_max=500.0, err_rms_strong_max=0.2,
                slope_min=0.8, within_source_min=0.8)
R2_MIN_RATIO = 0.5
R3_NULL_X, R3_TILT_X = 2.0, 1.5
R4_SPECIFIC, R4_GENERIC = 1.5, 1.2
NULLS = ('null_splithalf', 'null_rx_other_date', 'null_cell_other_date')
TILTS = ('ctl_tilt_plus', 'ctl_tilt_minus')
NULLS_R1 = tuple(n + '_r1' for n in NULLS)
TILTS_R1 = tuple(t + '_r1' for t in TILTS)


def read(path):
    # the arm called 'null' must not become NaN
    return pd.read_csv(path, keep_default_na=False, na_values=[''])


def merge(shards, out):
    out.mkdir(parents=True, exist_ok=True)
    frames = {}
    for name in ('arms', 'per_rx', 'confusion', 'explain'):
        frames[name] = pd.concat([read(Path(s) / f'{name}.csv') for s in shards], ignore_index=True)
        frames[name].to_csv(out / f'{name}.csv', index=False)
    val = read(Path(shards[0]) / 'validity.csv')
    for s in shards[1:]:
        other = read(Path(s) / 'validity.csv')
        assert np.allclose(val.select_dtypes('number').fillna(0).values, other.select_dtypes('number').fillna(0).values), \
            'shards built different interventions'
    val.to_csv(out / 'validity.csv', index=False)
    diag = {}
    for s in shards:
        for f in Path(s).glob('diag_*.json'):
            diag[f.stem[5:]] = json.loads(f.read_text())
    meta = json.loads((Path(shards[0]) / 'arms_meta.json').read_text())
    mani = [json.loads((Path(s) / 'manifest.json').read_text()) for s in shards]
    assert len({json.dumps(m['files'], sort_keys=True) for m in mani}) == 1, 'shards ran different code'
    timing = [json.loads((Path(s) / 'timing.json').read_text()) for s in shards]
    return frames, val, diag, meta, mani, timing


def present(arms, names):
    """Only the arms this run actually built (keeps the summarizer usable on older shards)."""
    have = set(arms.arm.unique())
    return tuple(n for n in names if n in have)


def flips(arms, arm):
    return arms[arms.arm == arm].set_index('run_id')['flip']


def rand_cov(arms, suffix=''):
    names = [a for a in sorted(arms.arm.unique()) if a.startswith('ctl_random_cov_') and a.endswith(suffix)
             and (suffix or not a.endswith('_r1'))]
    return pd.concat([flips(arms, a) for a in names], axis=1).median(1), names


def rand_iso(arms):
    names = [a for a in sorted(arms.arm.unique()) if a.startswith('ctl_random_iso_') and not a.endswith('_r1')]
    return pd.concat([flips(arms, a) for a in names], axis=1).median(1)


def by_seed(series, meta_runs):
    """median over the official methods within each seed."""
    s = series[series.index.isin(meta_runs.index)]
    return s.groupby(meta_runs.loc[s.index, 'seed']).median()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shards', nargs='+', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = Path(a.out)
    fr, val, diag, meta, mani, timing = merge(a.shards, out)
    arms = fr['arms']
    runs = arms[['run_id', 'backbone', 'method', 'seed']].drop_duplicates().set_index('run_id')
    off = runs[runs.backbone == 'official']
    res = runs[runs.backbone == 'resnet']
    S = {}

    # ---------------------------------------------------------------- R1 (model-free)
    v = val.set_index('arm')
    rx = v.loc['rx']
    r1 = dict(nonfinite=int(rx.nonfinite), periodicity_after=float(rx.periodicity_after),
              cfo_abs_hz=float(rx.cfo_change_hz_median_abs), err_rms=float(rx.err_rms),
              err_rms_strong=float(rx.err_rms_strong), slope=float(rx.realised_over_requested),
              within_source=float(rx.frac_within_source_range), nearest_is_partner=float(rx.frac_nearest_is_target))
    r1['pass'] = bool(r1['nonfinite'] == R1_RULES['nonfinite'] and r1['periodicity_after'] >= R1_RULES['periodicity_after_min']
                      and r1['cfo_abs_hz'] <= R1_RULES['cfo_abs_hz_max'] and r1['err_rms_strong'] <= R1_RULES['err_rms_strong_max']
                      and r1['slope'] >= R1_RULES['slope_min'] and r1['within_source'] >= R1_RULES['within_source_min'])
    S['R1'] = r1

    # ---------------------------------------------------------------- per-model flip table
    F = {k: flips(arms, k) for k in arms.arm.unique()}
    F['rand_cov'], cov_names = rand_cov(arms)
    F['rand_iso'] = rand_iso(arms)
    F['tilt'] = pd.concat([F[t] for t in present(arms, TILTS)], axis=1).max(1)
    F['null_max'] = pd.concat([F[n] for n in present(arms, NULLS)], axis=1).max(1)
    if present(arms, NULLS_R1):
        F['tilt_r1'] = pd.concat([F[t] for t in present(arms, TILTS_R1)], axis=1).max(1)
        F['null_max_r1'] = pd.concat([F[n] for n in present(arms, NULLS_R1)], axis=1).max(1)
        F['rand_cov_r1'], cov_names_r1 = rand_cov(arms, '_r1')
    med = lambda s, idx=off.index: float(s[s.index.isin(idx)].median())

    # ---------------------------------------------------------------- R2 artefact
    S['R2'] = dict(F_fir=med(F['rx']), F_r1_heldout=med(F['rx_r1']), F_v0=med(F['rx_v0circ']),
                   F_minphase=med(F['rx_minphase']), F_L0_insample=med(F['L0_tone_rx_swap']),
                   ratio_fir_over_r1=med(F['rx']) / med(F['rx_r1']))
    S['R2']['artefact'] = bool(S['R2']['ratio_fir_over_r1'] < R2_MIN_RATIO)

    # ---------------------------------------------------------------- R3 original ratios
    seeds = sorted(off.seed.unique())
    def r3(idx):
        f, n_, t_ = med(F['rx'], idx), med(F['null_max'], idx), med(F['tilt'], idx)
        return dict(F_rx=f, F_null_max=n_, F_tilt_max=t_, ratio_null=f / max(n_, 1e-9), ratio_tilt=f / max(t_, 1e-9),
                    pass_null=bool(f >= R3_NULL_X * n_), pass_tilt=bool(f >= R3_TILT_X * t_))
    S['R3'] = dict(overall=r3(off.index), per_seed={int(s): r3(off.index[off.seed == s]) for s in seeds},
                   resnet=r3(res.index))
    S['R3']['pass'] = S['R3']['overall']['pass_null'] and S['R3']['overall']['pass_tilt']
    S['R3']['seed_consistent'] = all(d['pass_null'] and d['pass_tilt'] for d in S['R3']['per_seed'].values())

    # ---------------------------------------------------------------- R4 direction specificity
    def q(idx, ctl):
        return med(F['rx'], idx) / max(med(F[ctl], idx), 1e-9)
    ctls = dict(Q_cov='rand_cov', Q_iso='rand_iso', Q_tilt='tilt', Q_shuffle='ctl_shuffle', Q_negated='ctl_negated',
                Q_link_dir='ctl_link_dir', Q_tx_dir='ctl_tx_dir')
    S['R4'] = dict(overall={k: q(off.index, c) for k, c in ctls.items()},
                   per_seed={int(s): {k: q(off.index[off.seed == s], c) for k, c in ctls.items()} for s in seeds},
                   resnet={k: q(res.index, c) for k, c in ctls.items()})
    per_model_q = (F['rx'] / F['rand_cov'].clip(lower=1e-9))
    qm = per_model_q[per_model_q.index.isin(off.index)].values
    S['R4']['per_model_Q_cov_median'] = float(np.median(qm))
    S['R4']['per_model_Q_cov_CI95'] = C.bootstrap_ci(qm, np.median)
    S['R4']['F_rx_minus_F_cov_pp_CI95'] = [100 * v for v in C.bootstrap_ci(
        (F['rx'] - F['rand_cov'])[F['rx'].index.isin(off.index)].values, np.median)]
    qo = S['R4']['overall']['Q_cov']
    qs = [d['Q_cov'] for d in S['R4']['per_seed'].values()]
    S['R4']['verdict'] = ('RX_SPECIFIC' if qo >= R4_SPECIFIC and all(x >= R4_SPECIFIC for x in qs)
                          else 'GENERIC' if qo <= R4_GENERIC else 'PARTIAL')
    # implementation robustness: the same comparison with controls realised by the L0 r1 gain, which
    # reproduces the requested tone change exactly (no notched-tone realisation error). Post-hoc
    # check, not part of the frozen verdict.
    if 'rand_cov_r1' in F:
        def qr(idx, ctl):
            return med(F['rx_r1'], idx) / max(med(F[ctl], idx), 1e-9)
        S['R4b_r1_implementation'] = dict(
            overall=dict(Q_cov=qr(off.index, 'rand_cov_r1'), Q_tilt=qr(off.index, 'tilt_r1'),
                         Q_null=med(F['rx_r1'], off.index) / max(med(F['null_max_r1'], off.index), 1e-9)),
            per_seed={int(s): dict(Q_cov=qr(off.index[off.seed == s], 'rand_cov_r1'),
                                   Q_tilt=qr(off.index[off.seed == s], 'tilt_r1')) for s in seeds},
            resnet=dict(Q_cov=qr(res.index, 'rand_cov_r1'), Q_tilt=qr(res.index, 'tilt_r1')))
        pm = (F['rx_r1'] / F['rand_cov_r1'].clip(lower=1e-9))
        S['R4b_r1_implementation']['per_model_Q_cov_median'] = float(np.median(pm[pm.index.isin(off.index)]))
        S['R4b_r1_implementation']['per_model_Q_cov_CI95'] = C.bootstrap_ci(pm[pm.index.isin(off.index)].values,
                                                                           np.median)

    # ---------------------------------------------------------------- per-RX breakdown (own RX)
    pr = fr['per_rx']
    own = pr[(pr.role == 'own') & pr.run_id.isin(off.index)]
    t_rx = own[own.arm == 'rx'].groupby('rx').flip.median()
    t_cov = pd.concat([own[own.arm == n].groupby('rx').flip.median() for n in cov_names], axis=1).median(1)
    t_tilt = pd.concat([own[own.arm == n].groupby('rx').flip.median() for n in TILTS], axis=1).max(1)
    per_rx = pd.DataFrame(dict(F_rx=t_rx, F_rand_cov=t_cov, F_tilt=t_tilt))
    per_rx['Q_cov'] = per_rx.F_rx / per_rx.F_rand_cov.clip(lower=1e-9)
    hw = {int(h['rx_index']): h['hardware'] for h in json.loads(
        (Path(mani[0]['root']) / 'configs/receiver_r01/registration.json').read_text())['wisig']['hardware_and_coverage']}
    per_rx['hardware'] = [hw.get(int(r), '?') for r in per_rx.index]
    per_rx.to_csv(out / 'per_rx_specificity.csv')
    S['per_rx'] = dict(n_rx=int(len(per_rx)), n_Q_cov_ge_1p5=int((per_rx.Q_cov >= 1.5).sum()),
                       n_Q_cov_le_1p2=int((per_rx.Q_cov <= 1.2).sum()), Q_cov_min=float(per_rx.Q_cov.min()),
                       Q_cov_max=float(per_rx.Q_cov.max()))
    part = pr[(pr.role == 'partner') & pr.run_id.isin(off.index) & (pr.arm == 'rx')].groupby('rx').flip.median()
    S['per_rx']['partner_flip_range'] = [float(part.min()), float(part.max())]

    # ---------------------------------------------------------------- R6 dose
    dose = arms[arms.series.notna() & arms.run_id.isin(off.index)]
    curve = dose.groupby(['series', 'alpha']).flip.median().unstack(0)
    curve.to_csv(out / 'dose_curve.csv')
    nrx = float(v.loc['rx', 'requested_norm_median'])
    def at(series, level=0.2):
        c = curve[series].dropna()
        xs, ys = np.r_[0.0, c.index.values], np.r_[0.0, c.values]
        if ys.max() < level:
            return float('inf')
        k = int(np.argmax(ys >= level))
        return float((xs[k - 1] + (level - ys[k - 1]) * (xs[k] - xs[k - 1]) / (ys[k] - ys[k - 1])) * nrx)
    S['R6'] = dict(norm_median_rx_swap=nrx, norm_at_F20={s: at(s) for s in curve.columns},
                   F_by_alpha={s: {float(k): float(x) for k, x in curve[s].items()} for s in curve.columns})
    # second, implementation-independent read: interpolate every dose series on the *realised*
    # 12-tone change (tone_norm_mean), so an arm that realises more than requested is not credited
    real_norm = {a: float(v.loc[a, 'tone_norm_mean']) for a in v.index}
    rows = []
    for (s, al), g in dose.groupby(['series', 'alpha']):
        rows.append(dict(series=s, alpha=al, flip=g.flip.median(), realised_norm=real_norm.get(f'dose_{s}_{al:g}', np.nan)))
    rc = pd.DataFrame(rows)
    rc.to_csv(out / 'dose_curve_realised.csv', index=False)
    S['R6']['F_at_realised_norm'] = {}
    for target in (1.0, 2.0, 3.0):
        S['R6']['F_at_realised_norm'][str(target)] = {}
        for s_, g in rc.groupby('series'):
            g = g.sort_values('realised_norm')
            S['R6']['F_at_realised_norm'][str(target)][s_] = float(np.interp(target, g.realised_norm, g.flip))

    # ---------------------------------------------------------------- R7 explanatory
    unseen = pd.Series({k: d['unseen']['acc'] for k, d in diag.items()})
    seen = pd.Series({k: d['seen']['acc'] for k, d in diag.items()})
    preds = dict(F_rx='rx', F_rx_r1='rx_r1', F_L0='L0_tone_rx_swap', F_rand_cov='rand_cov', F_tilt='tilt',
                 F_minphase='rx_minphase', F_cell='cell', F_null_cell_other_date='null_cell_other_date')
    idx = off.index
    def within_seed(xs):
        vals = []
        for s in seeds:
            ii = off.index[off.seed == s]
            vals.append(C.spearman(xs.loc[ii], unseen.loc[ii]))
        return vals
    S['R7'] = dict(unseen_acc_median=float(unseen.loc[idx].median()), n_models=int(len(idx)),
                   spearman_overall={k: C.spearman(F[a].loc[idx], unseen.loc[idx]) for k, a in preds.items()},
                   spearman_within_seed={k: within_seed(F[a]) for k, a in preds.items()},
                   spearman_source_acc=C.spearman(pd.Series({k: d['source_eval_null_acc'] for k, d in diag.items()}).loc[idx],
                                                  unseen.loc[idx]))
    # confusion-structure similarity with the internal unseen-date confusion
    conf = fr['confusion']
    sims = []
    for rid in idx:
        U = np.array(diag[rid]['unseen']['conf'], float)
        U = U / U.sum(1, keepdims=True)
        np.fill_diagonal(U, 0)
        for arm, g in conf[conf.run_id == rid].groupby('arm'):
            A = g.sort_values('true')[[f'p{q}' for q in range(10)]].values.astype(float)
            A = A / np.maximum(A.sum(1, keepdims=True), 1)
            np.fill_diagonal(A, 0)
            den = np.linalg.norm(A) * np.linalg.norm(U)
            sims.append(dict(run_id=rid, arm=arm, cosine=float((A * U).sum() / den) if den > 0 else np.nan,
                             offdiag_mass=float(A.sum() / 10)))
    sims = pd.DataFrame(sims)
    sims.to_csv(out / 'confusion_similarity.csv', index=False)
    S['R7']['confusion_cosine_median'] = sims.groupby('arm').cosine.median().to_dict()
    ex = fr['explain'][fr['explain'].run_id.isin(idx)]
    S['R7']['flip_follows_tone_rule'] = ex.groupby('arm')[['flip_to_tone_rule_own_rx', 'flip_to_tone_rule_partner_rx']].median().to_dict('index')

    # ---------------------------------------------------------------- R8 transplant
    S['R8'] = {k: med(F[k]) for k in ('rx', 'link_fir', 'tx_fir', 'cell', 'cell_cfo', 'real_cell_rx', 'ctl_link_dir',
                                      'ctl_tx_dir', 'link_r1', 'tx_r1')}
    S['R8'].update({f'acc_drop_pp_{k}': float(arms[(arms.arm == k) & arms.run_id.isin(idx)].acc_drop_pp.median())
                    for k in ('rx', 'rx_r1', 'L0_tone_rx_swap', 'cell', 'cell_cfo', 'real_cell_rx', 'rand_cov_0')
                    if (arms.arm == k).any()})

    # ---------------------------------------------------------------- full arm table
    tab = arms[arms.run_id.isin(idx)].groupby('arm')[['flip', 'acc_drop_pp', 'p_true_median', 'feature_rel_change',
                                                     'pred_entropy_bits']].median()
    tab = tab.join(v[['kind', 'requested_norm_median', 'tone_norm_mean', 'err_rms', 'err_rms_strong',
                      'input_rel_change', 'stf_rel_change', 'ltf_rel_change', 'cfo_change_hz_median_abs',
                      'periodicity_after', 'frac_within_source_range', 'frac_nearest_is_target']], how='left')
    tab.to_csv(out / 'arm_table.csv')

    # ---------------------------------------------------------------- decision
    if not r1['pass']:
        dec = 'INVALID_INTERVENTION'
    elif S['R2']['artefact']:
        dec = 'ARTEFACT'
    else:
        dec = S['R4']['verdict']
    S['decision'] = dec
    S['crr_prior'] = {'RX_SPECIFIC': 'V2 > V1 expected', 'GENERIC': 'V1 ~ V2 expected; strong random spectral baseline first',
                      'PARTIAL': 'no prior', 'INVALID_INTERVENTION': 'no CRR training', 'ARTEFACT': 'no CRR training'}[dec]
    S['budget'] = dict(reserved_gpuh=float(sum(t['seconds'] for t in timing) / 3600), shards=len(timing),
                       peak_MiB=float(max(t['peak_allocated_MiB'] for t in timing)))
    S['rules'] = dict(R1=R1_RULES, R2_MIN_RATIO=R2_MIN_RATIO, R3=(R3_NULL_X, R3_TILT_X), R4=(R4_SPECIFIC, R4_GENERIC))
    (out / 'summary.json').write_text(json.dumps(C.jsonable(S), indent=1))
    print(json.dumps(C.jsonable(dict(decision=dec, R1=r1['pass'], R2=S['R2'], R3=S['R3']['overall'],
                                     R4=S['R4']['overall'], R4_seeds=S['R4']['per_seed'], verdict=S['R4']['verdict'])), indent=1))


if __name__ == '__main__':
    main()
