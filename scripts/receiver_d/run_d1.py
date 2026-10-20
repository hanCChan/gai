#!/usr/bin/env python3
"""D1: are the WiSig receiver-response interventions credible receiver-condition changes?

Rules are frozen in docs/receiver_d/PROTOCOL_D.md (written before any D1 model output). In short:
templates from positions 0-119, evaluation on the disjoint positions 160-199 of the same source
receivers, four implementations of the same requested change (L0 r1_segment, v0_circular, fir_zero,
fir_min), every control norm-matched to the real RX swap, natural-size same-receiver nulls, dose
series, per-RX / per-seed breakdowns, and the model-free checks R1 (credibility) and R2 (artefact)
that are decided before any accuracy number is read.

Inference only: frozen R0/R1 checkpoints, no fitting, no new checkpoints, final 8 RX / date 3 /
equalized never read. Shard with --shard i/n over GPUs.

    python3 run_d1.py --root /workspace/rl1/gai_retrain --out OUT --shard 0/4
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
L0 = HERE.parent / 'interaction_l0'
sys.path[:0] = [str(HERE), str(L0)]

from link_core import rx_equal_accuracy, cell_means, effects  # noqa: E402  (r1 versions)
from physics import TONE_BINS, PERIOD, stf_descriptors, shift_cfo, to_cache, to_complex  # noqa: E402
import physics  # noqa: E402
import rxphys as R  # noqa: E402
import d_common as C  # noqa: E402
import run_l0 as R1  # noqa: E402  (reproduces the exact L0 interventions)

FIT_POS, EVAL_POS, POS_SEL = (0, 120), (160, 200), (120, 160)
EVAL_DAYS, INTERNAL_DAYS = [0, 1], [0, 1, 2]
SEED = 20261019
DOSES = (0.25, 0.5, 1.0, 1.5, 2.0)
PRIMARY = 'fir_zero'
MODES = ('r1_segment', 'v0_circular', 'fir_zero', 'fir_min')
N_RANDOM = 4


# ------------------------------------------------------------------------------------- arms

def pair_up(tpl, tx, rx, rng):
    """One partner RX, one mismatched RX and one partner TX per packet (shared by all arms)."""
    rxn = np.asarray(tpl['rx_names'], int)
    txn = np.asarray(tpl['tx_names'], int)
    partner = np.empty(len(rx), int)
    mism = np.empty(len(rx), int)
    for r in rxn:
        idx = np.flatnonzero(rx == r)
        others = [int(v) for v in rxn if v != r]
        partner[idx] = rng.choice(others, len(idx))
        mism[idx] = rng.choice(others, len(idx))
    tpartner = np.empty(len(tx), int)
    for t in txn:
        idx = np.flatnonzero(tx == t)
        tpartner[idx] = rng.choice([int(v) for v in txn if v != t], len(idx))
    return partner, mism, tpartner


def deltas(tpl, tpl_day, tx, rx, partner, mism, tpartner, day):
    """Requested 12-tone changes for every direction, per packet (templates of the packet's date)."""
    pos = {int(r): k for k, r in enumerate(tpl['rx_names'])}
    tpos = {int(t): k for k, t in enumerate(tpl['tx_names'])}
    own, par, mis = (np.array([pos[int(v)] for v in arr]) for arr in (rx, partner, mism))
    ti = np.array([tpos[int(t)] for t in tx])
    tpi = np.array([tpos[int(t)] for t in tpartner])
    out = dict(
        rx=R.centre_shape(tpl['b'][par] - tpl['b'][own]),
        link=R.centre_shape(tpl['i'][ti, par] - tpl['i'][ti, own]),
        tx=R.centre_shape(tpl['a'][tpi] - tpl['a'][ti]),
        cell=R.centre_shape(tpl['M'][ti, par] - tpl['M'][ti, own]),
        cfo=(tpl['cfo_M'][ti, par] - tpl['cfo_M'][ti, own])[:, None])   # Hz, not a shape: never centred
    # same-RX natural nulls: another fit-pool date, same RX (and same cell for the second)
    dt = np.zeros_like(out['rx'])
    dc = np.zeros_like(out['rx'])
    for dd in np.unique(day):
        other = [d for d in sorted(tpl_day) if d != dd]
        if not other:
            break
        od = other[0]
        m = day == dd
        po = [pos[int(v)] for v in rx[m]]
        dt[m] = R.centre_shape(tpl_day[od]['b'][po] - tpl_day[dd]['b'][po])
        dc[m] = R.centre_shape(tpl_day[od]['M'][ti[m], own[m]] - tpl_day[dd]['M'][ti[m], own[m]])
    out['rx_other_date'] = dt
    out['cell_other_date'] = dc
    out['_partner'], out['_mismatched'], out['_tx_partner'] = partner, mism, tpartner
    return out


def build_arms(z, tx, rx, day, delta, tpl, split_half, rng):
    """{arm: dict}; controls are norm-matched to ||b[r2]-b[r]|| per packet (see PROTOCOL_D.md)."""
    n = len(z)
    nrx = np.linalg.norm(delta['rx'], axis=1)
    X = R.centre_shape(tpl['b'])
    X = X - X.mean(0, keepdims=True)
    cov = np.cov(np.array([R.centre_shape(X[j] - X[k])[0] for j in range(len(X)) for k in range(len(X)) if j != k]).T)
    arms = {}

    def add(name, zz, real, req, note, kind):
        arms[name] = dict(z=zz, realised=real, requested=R.centre_shape(req), note=note, kind=kind,
                          series=None, alpha=None)

    add('null', z, np.zeros((n, 12)), np.zeros((n, 12)), 'no intervention', 'null')
    for mode in MODES:
        tag = {'r1_segment': 'rx_r1', 'v0_circular': 'rx_v0circ', 'fir_zero': 'rx', 'fir_min': 'rx_minphase'}[mode]
        zz, real = R.apply_response(z, delta['rx'], mode)
        add(tag, zz, real, delta['rx'], f'RX main-effect swap, held-out templates, {mode}', 'rx')
        if mode in ('r1_segment', 'fir_zero'):
            for k in ('link', 'tx'):
                zz, real = R.apply_response(z, delta[k], mode)
                add(f'{k}_{"r1" if mode == "r1_segment" else "fir"}', zz, real, delta[k],
                    f'{k} swap, held-out templates, {mode}', k)
    zz, real = R.apply_response(z, delta['cell'], PRIMARY)
    add('cell', zz, real, delta['cell'], 'cell transplant M[t,r2]-M[t,r] (incl. link), fir_zero', 'cell')
    zcfo = shift_cfo(zz, delta['cfo'][:, 0])
    add('cell_cfo', zcfo, real, delta['cell'], 'cell transplant + RX CFO difference, fir_zero', 'cell')
    zz, real = R.apply_response(z, delta['real_cell'], PRIMARY)
    add('real_cell_rx', zz, real, delta['real_cell'],
        'real source packet of this TX, relabelled to the partner RX (its own link term kept)', 'rx')

    # ---- matched-norm controls (per packet ||delta|| = ||b[r2]-b[r]||)
    controls = dict(random_iso=[R.gaussian_direction(n, nrx, rng) for _ in range(N_RANDOM)],
                    random_cov=[R.gaussian_direction(n, nrx, rng, cov) for _ in range(N_RANDOM)],
                    tilt_plus=[R.smooth_tilt(n, nrx, +1)], tilt_minus=[R.smooth_tilt(n, nrx, -1)],
                    shuffle=[R.shuffle_tones(delta['rx'], rng)], negated=[-delta['rx']])
    controls['link_dir'] = [R.match_norm(delta['link'], nrx)]
    controls['tx_dir'] = [R.match_norm(delta['tx'], nrx)]
    sh = np.stack([split_half[int(r)] for r in rx])
    nulls = dict(splithalf=sh, rx_other_date=delta['rx_other_date'], cell_other_date=delta['cell_other_date'])
    # every control and null in the primary physical filter and, as the implementation-robustness
    # check, in the L0 r1 measurement-domain gain (which realises the requested change exactly)
    for mode, suffix in ((PRIMARY, ''), ('r1_segment', '_r1')):
        for key, lst in controls.items():
            for k, d in enumerate(lst):
                zz, real = R.apply_response(z, d, mode)
                add(f'ctl_{key}' + (f'_{k}' if len(lst) > 1 else '') + suffix, zz, real, d,
                    f'matched-norm control: {key}, {mode}', 'control')
        for key, d in nulls.items():
            zz, real = R.apply_response(z, d, mode)
            add(f'null_{key}{suffix}', zz, real, d, f'same-RX natural null: {key}, {mode}', 'null')
    # ---- dose series (alpha x the packet's own ||b[r2]-b[r]||, every direction norm-matched)
    for al in DOSES:
        for series, d in (('rx', delta['rx']), ('rand_cov', controls['random_cov'][0]),
                          ('tilt', controls['tilt_plus'][0]), ('link', controls['link_dir'][0]),
                          ('tx', controls['tx_dir'][0])):
            zz, real = R.apply_response(z, d * al, PRIMARY)
            add(f'dose_{series}_{al:g}', zz, real, d * al, f'{series} x{al:g}', 'rx' if series == 'rx' else 'control')
            arms[f'dose_{series}_{al:g}'].update(series=series, alpha=al)
    return arms


# -------------------------------------------------------------------------- measurements

def arm_measure(z0, arm):
    m = R.measure(z0, arm['z'])
    m.update(R.realisation(z0, arm['realised'], arm['requested']))
    return m


def _cache_to_complex(x):
    """(n,1,256,2) float32 cache layout -> (n,256) complex, as physics.to_complex does."""
    a = np.asarray(x)
    return a[:, 0, :, 0].astype(np.float64) + 1j * a[:, 0, :, 1].astype(np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    si, sn = map(int, a.shard.split('/'))
    assert 0 <= si < sn

    root = C.add_root(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    import pandas as pd
    import torch
    from receiver_r01.data import wisig
    from train_receiver_r01 import build

    reg = json.loads((root / 'configs/receiver_r01/registration.json').read_text())
    roles = reg['wisig']['roles']
    jobs = [dict(j, ckpt=root / 'outputs/receiver_r01/formal' / j['run_id'])
            for j in reg['formal_jobs'] if j['dataset'] == 'wisig']
    jobs = jobs[si::sn]
    if a.limit:
        jobs = jobs[:a.limit]
    assert jobs

    src, internal = roles['source'], roles['internal_rx']
    fit = wisig(reg, src, EVAL_DAYS, list(FIT_POS))
    ev = wisig(reg, src, EVAL_DAYS, list(EVAL_POS))
    sel = wisig(reg, src, EVAL_DAYS, list(POS_SEL))
    assert set(roles['final_rx']).isdisjoint(set(src)), 'final RX in source role'
    key = lambda d: set(map(tuple, np.c_[d['rx'], d['day'], d['y'], d['position']]))
    assert not (key(fit) & key(ev)) and not (key(fit) & key(sel)) and not (key(ev) & key(sel))

    z_fit, z_ev = to_complex(fit['x']), to_complex(ev['x'])
    d_fit, d_ev = stf_descriptors(z_fit), stf_descriptors(z_ev)
    tpl = R.response_templates(d_fit['tone_shape'], d_fit['cfo_hz'], fit['y'], fit['rx'], fit['day'])
    # per-date templates (same fit pool) for the same-RX-other-date null
    tpl_day = {}
    for dd in np.unique(fit['day']):
        m = fit['day'] == dd
        tpl_day[int(dd)] = R.response_templates(d_fit['tone_shape'][m], d_fit['cfo_hz'][m], fit['y'][m],
                                                fit['rx'][m], fit['day'][m])
    split_half = R.split_half_rx_delta(d_fit['tone_shape'], d_fit['cfo_hz'], fit['y'], fit['rx'], fit['day'])

    rng = np.random.default_rng(SEED)
    partner, mism, tpartner = pair_up(tpl, ev['y'], ev['rx'], rng)
    delta = deltas(tpl, tpl_day, ev['y'], ev['rx'], partner, mism, tpartner, ev['day'])
    # mean within-cell deviation of the fit pool, keyed (date, TX, RX): a real packet of this TX
    # relabelled to the partner RX keeps its own packet deviation and takes the partner cell mean
    dev_mean = {}
    for dd in np.unique(fit['day']):
        for i, t in enumerate(tpl['tx_names']):
            for j, r in enumerate(tpl['rx_names']):
                mm = (fit['day'] == dd) & (fit['y'] == t) & (fit['rx'] == r)
                if mm.any():
                    dev_mean[(int(dd), int(t), int(r))] = d_fit['tone_shape'][mm].mean(0) - tpl['M'][i, j]
    pos = {int(r): k for k, r in enumerate(tpl['rx_names'])}
    tpos = {int(t): k for k, t in enumerate(tpl['tx_names'])}
    real_cell = delta['rx'].copy()
    for k in range(len(z_ev)):
        dv = dev_mean.get((int(ev['day'][k]), int(ev['y'][k]), int(ev['rx'][k])))
        ti, oi, pi = tpos[int(ev['y'][k])], pos[int(ev['rx'][k])], pos[int(partner[k])]
        extra = 0.0 if dv is None else dv
        real_cell[k] = R.centre_shape(extra + tpl['M'][ti, pi] - tpl['M'][ti, oi])[0]
    delta['real_cell'] = real_cell

    arms = build_arms(z_ev, ev['y'], ev['rx'], ev['day'], delta, tpl, split_half, rng)
    phys = R1.wisig_physics(dict(source=ev))
    for name, (xc, _) in phys['interventions'].items():  # same eval pool, L0 in-sample templates
        zz = _cache_to_complex(xc)
        arms[f'L0_{name}'] = dict(z=zz, realised=stf_descriptors(zz)['tone_shape'] - d_ev['tone_shape'],
                                  requested=np.zeros((len(zz), 12)), note=f'L0 reproduction: {name} (in-sample templates)',
                                  kind='l0', series=None, alpha=None)

    # model-free records
    val = []
    for name, arm in arms.items():
        val.append(dict(arm=name, note=arm['note'], kind=arm['kind'], series=arm.get('series'), alpha=arm.get('alpha'),
                                requested_norm_median=float(np.median(np.linalg.norm(arm['requested'], axis=1))),
                                **{k: v for k, v in arm_measure(z_ev, arm).items() if k != 'd_tone'},
                                **R.plausibility(d_ev['tone_shape'] + arm['realised'], ev['y'],
                                                 partner if arm['kind'] in ('rx', 'cell', 'l0') else ev['rx'], tpl)))
    pd.DataFrame(val).to_csv(out / 'validity.csv', index=False)
    meas = {name: arm_measure(z_ev, arm) for name, arm in arms.items()}
    (out / 'arms_meta.json').write_text(json.dumps(C.jsonable(dict(
        arms={k: dict(v) for k, v in meas.items()}, partner=partner.tolist(), mismatched=mism.tolist(),
        tx_partner=tpartner.tolist(), geometry=R.template_geometry(tpl), rx_names=[int(v) for v in tpl['rx_names']],
        split_half_norm={int(k): float(np.linalg.norm(v)) for k, v in split_half.items()},
        l0_validation=phys['validation'], l0_sigma_b_hz=phys['sigma_b_hz'],
        notes={k: arms[k]['note'] for k in arms}, kinds={k: arms[k]['kind'] for k in arms})), indent=1))
    (out / 'manifest.json').write_text(json.dumps(dict(
        files={f.name: C.sha(f) for f in sorted(HERE.glob('*.py'))}, code_sha256_l0=R1.code_shas(),
        jobs=[j['run_id'] for j in jobs], shard=a.shard, root=str(root), seed=SEED,
        fit_positions=list(FIT_POS), eval_positions=list(EVAL_POS), selection_positions=list(POS_SEL),
        network_parameters_changed=False, head_refits=False, final_RX_read=False, date3_read=False,
        equalized_read=False, started_unix=time.time()), indent=1, default=str))

    # ---------------------------------------------------------------------------- inference
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    x_arms = {name: to_cache(arm['z']) for name, arm in arms.items()}
    internal_pool = {name: wisig(reg, internal, days, [0, 200])
                     for name, days in (('seen', [0, 1]), ('unseen', [2]))}

    names = sorted(arms)
    pos_arr = np.array([pos[int(v)] for v in ev['rx']])
    par_arr = np.array([pos[int(v)] for v in partner])
    shapes_after = {k: d_ev['tone_shape'] + arms[k]['realised'] for k in names}
    rule_own = {k: _tone_rule(tpl, shapes_after[k], pos_arr) for k in STRUCTURE_ARMS if k in arms}
    rule_par = {k: _tone_rule(tpl, shapes_after[k], par_arr) for k in STRUCTURE_ARMS if k in arms}
    y, rx, dday = ev['y'], ev['rx'], ev['day']
    keep = d_ev['periodicity'] >= 0.5
    evd = dict(y=y, rx=rx, day=dday)

    rows, per_rx, conf_rows, expl = [], [], [], []
    pred_store, ptrue_store = {}, {}
    tick = time.perf_counter()
    for job in jobs:
        cp, _ = C.load_model(job['ckpt'])
        model = build(job).to(device)
        model.load_state_dict(cp['model'])
        model.eval()
        enc, head, _ = C.split(model)
        with torch.no_grad():
            h = torch.as_tensor(x_arms['null'][:256], device=device)
            assert float((model(h) - head(enc(h))).abs().max()) < 1e-4, job['run_id']
        feats = {k: C.run_batches(enc, x_arms[k], device, 2048) for k in names}
        lg = {k: C.run_batches(head, feats[k], device, 4096) for k in names}
        pred = {k: v.argmax(1) for k, v in lg.items()}
        ptrue = {k: C.logits_to_probs(v)[np.arange(len(y)), y] for k, v in lg.items()}
        base, base_f = pred['null'], feats['null']
        base_acc = rx_equal_accuracy(base, y, rx, dday)
        base_rx = C.per_rx_accuracy(base, evd)
        ident = dict(run_id=job['run_id'], backbone=job['backbone'], method=job['method'], seed=job['seed'])
        pred_store[job['run_id']] = np.stack([pred[k].astype(np.int8) for k in names])
        ptrue_store['p_' + job['run_id']] = np.stack([ptrue[k].astype(np.float16) for k in names])
        fnorm = np.maximum(np.linalg.norm(base_f, axis=1), 1e-12)
        for k in names:
            pr, arm = pred[k], arms[k]
            fl = pr != base
            acc = rx_equal_accuracy(pr, y, rx, dday)
            _, cnt = np.unique(pr[keep], return_counts=True)
            share = cnt / cnt.sum()
            rows.append(dict(**ident, arm=k, kind=arm['kind'], series=arm['series'], alpha=arm['alpha'],
                             acc=acc, acc_drop_pp=100 * (base_acc - acc), flip=float(fl.mean()),
                             flip_kept=float(fl[keep].mean()), p_true_median=float(np.median(ptrue[k])),
                             p_true_mean=float(ptrue[k].mean()),
                             feature_rel_change=float(np.median(np.linalg.norm(feats[k] - base_f, axis=1) / fnorm)),
                             pred_entropy_bits=float(-(share * np.log2(share)).sum()), top_class_share=float(share.max())))
            acc_rx = C.per_rx_accuracy(pr, evd)
            for rv in np.unique(rx):
                m = rx == rv
                per_rx.append(dict(**ident, arm=k, role='own', rx=int(rv), acc=acc_rx[int(rv)],
                                   acc_drop_pp=100 * (base_rx[int(rv)] - acc_rx[int(rv)]), flip=float(fl[m].mean())))
                mp = partner == rv
                per_rx.append(dict(**ident, arm=k, role='partner', rx=int(rv), acc=np.nan, acc_drop_pp=np.nan,
                                   flip=float(fl[mp].mean())))
            if k in STRUCTURE_ARMS:
                for t in range(10):
                    conf_rows.append(dict(**ident, arm=k, true=t, **{f'p{q}': int(((pr == q) & (y == t)).sum())
                                                                    for q in range(10)}))
                wrong = fl & (pr != y)
                expl.append(dict(**ident, arm=k, n_flip=int(fl.sum()), n_flip_wrong=int(wrong.sum()),
                                 flip_to_tone_rule_own_rx=float((pr[wrong] == rule_own[k][wrong]).mean()) if wrong.any() else np.nan,
                                 flip_to_tone_rule_partner_rx=float((pr[wrong] == rule_par[k][wrong]).mean()) if wrong.any() else np.nan,
                                 tone_rule_own_is_true=float((rule_own[k] == y).mean()),
                                 tone_rule_partner_is_true=float((rule_par[k] == y).mean())))
        # R7 diagnostics: internal RX (already-viewed development pools)
        diag = {}
        for pname, dint in internal_pool.items():
            pr = C.run_batches(lambda t: model(t), dint['x'], device, 2048).argmax(1)
            diag[pname] = dict(acc=rx_equal_accuracy(pr, dint['y'], dint['rx'], dint['day']),
                               per_rx=C.per_rx_accuracy(pr, dint),
                               conf=[[int(((pr == q) & (dint['y'] == t)).sum()) for q in range(10)] for t in range(10)])
        diag['source_eval_null_acc'] = base_acc
        (out / f'diag_{job["run_id"]}.json').write_text(json.dumps(C.jsonable(diag), indent=1))
        del model, feats, lg
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        for nm, rws in (('arms', rows), ('per_rx', per_rx), ('confusion', conf_rows), ('explain', expl)):
            pd.DataFrame(rws).to_csv(out / f'{nm}.csv', index=False)
        show = lambda k: round(float(rx_equal_accuracy(pred[k], y, rx, dday)), 4)
        print(json.dumps(dict(run_id=job['run_id'], null=round(base_acc, 4), L0=show('L0_tone_rx_swap'),
                              r1=show('rx_r1'), fir=show('rx'), minph=show('rx_minphase'),
                              rand=show('ctl_random_cov_0'), tilt=show('ctl_tilt_plus'),
                              unseen=round(diag['unseen']['acc'], 4))) + f' {time.perf_counter() - tick:.0f}s', flush=True)

    np.savez_compressed(out / 'predictions.npz', arms=np.array(names), y=y, rx=rx, day=dday, partner=partner,
                        **pred_store, **ptrue_store)
    timing = dict(seconds=time.perf_counter() - tick, jobs=len(jobs), device=str(device),
                  peak_allocated_MiB=float(torch.cuda.max_memory_allocated() / 2 ** 20) if device.type == 'cuda' else 0.0)
    (out / 'timing.json').write_text(json.dumps(timing, indent=1))
    print('done', json.dumps(timing))


STRUCTURE_ARMS = ('null', 'rx', 'rx_r1', 'rx_minphase', 'L0_tone_rx_swap', 'ctl_random_cov_0', 'ctl_random_iso_0',
                  'ctl_tilt_plus', 'ctl_shuffle', 'ctl_negated', 'cell', 'cell_cfo', 'real_cell_rx', 'link_fir',
                  'tx_fir', 'null_splithalf', 'null_cell_other_date')


def _tone_rule(tpl, shapes, rx_index):
    """Nearest-TX guess: which TX's real fit-pool cell mean at receiver `rx_index` (per packet) is
    nearest in 12-tone shape to the packet's (possibly modified) tone shape. Asks whether flips
    land on the TX whose real spectrum the modified packet now resembles."""
    M = tpl['M']                                         # [T,R,12]
    cells = np.transpose(M[:, rx_index], (1, 0, 2))      # [n,T,12]
    return np.asarray(tpl['tx_names'])[((cells - shapes[:, None, :]) ** 2).sum(-1).argmin(1)]


if __name__ == '__main__':
    main()
