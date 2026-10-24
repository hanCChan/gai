#!/usr/bin/env python3
"""Stage A: frozen-head comparison H0-H5 on encoder-held source_selection RX (EXPERIMENT_PROTOCOL.md §5).

Real run (requires an ACTIVE reservation with allow_feature_extraction):
    python scripts/rffi_calibration/frozen_heads.py --ckpt outputs/rffi_calibration/<job_id> --out reports/rffi_calibration/E4_frozen_heads/<job_id>
  - the checkpoint's status.json must say completed and its SHA must match;
  - the evaluated RX must be absent from the checkpoint's fit RX (encoder-held); otherwise the run is
    labelled INTERNAL_DIAGNOSTIC and excluded from the gate;
  - source statistics use the checkpoint's FIT RX (fit positions only);
  - target support/query come from the registered manifest rule; query labels enter metrics only.

Synthetic dry run (no data access, writes to --out):
    python scripts/rffi_calibration/frozen_heads.py --synthetic --out /tmp/rffi_stageA_dry
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from rffi_calibration.access import AccessGuard  # noqa: E402
from rffi_calibration.evaluator import (BUDGETS, FeatureSet, aggregate, crossfit_family_rows, evaluate_target,  # noqa: E402
                                        fit_heads, paired)
from rffi_calibration.heads import default_heads  # noqa: E402
from rffi_calibration.registry import Registry, file_sha  # noqa: E402
from rffi_calibration.sampling import RecordTable, build_manifest, manifest_sha  # noqa: E402

COMPARE = [('H4', 'H2*'), ('H4', 'H3'), ('H4', 'H2L'), ('H4', 'H1'), ('H4', 'H4-AK'), ('H4', 'H4-pooled'),
           ('H2*', 'H2L'), ('H5c*', 'H4'), ('H5b', 'H4')]


def synthetic(T=10, R_fit=6, R_sel=3, d=16, n=200, seed=0):
    """Features with class-specific receiver shifts inside a low-rank span + packet noise, two dates."""
    rng = np.random.default_rng(seed)
    proto = 2.5 * rng.standard_normal((T, d))
    E = rng.standard_normal((T, d, 3))
    out = []
    for r in range(R_fit + R_sel):
        shift = np.einsum('tdk,tk->td', E, 0.8 * rng.standard_normal((T, 3)))
        for date in (0, 1):
            for t in range(T):
                out.append((t, r, date, proto[t] + shift[t] + rng.standard_normal((n // 2, d))))
    tx = np.concatenate([[t] * len(z) for t, _, _, z in out])
    rx = np.concatenate([[r] * len(z) for _, r, _, z in out])
    date = np.concatenate([[dd] * len(z) for _, _, dd, z in out])
    pos = np.concatenate([np.arange(len(z)) + 1000 * t for t, _, _, z in out])
    z = np.concatenate([z for *_, z in out])
    tab = RecordTable('synthetic', tx, rx, date, pos, 'cache_position',
                      capture=np.array([f'rx{a}_d{b}' for a, b in zip(rx, date)]))
    return tab, z, list(range(R_fit)), list(range(R_fit, R_fit + R_sel))


def subset(fs, mask):
    t = fs.table
    sub = RecordTable(t.dataset_id, t.tx[mask], t.rx[mask], t.date[mask], t.position[mask], t.sample_unit,
                      capture=None if t.capture is None else t.capture[mask])
    return FeatureSet(fs.z[mask], sub, fs.meta)


def run(source, targets, T, rule, q_per_class, out, n_draws, native=None, log=print):
    out.mkdir(parents=True, exist_ok=True)
    heads = default_heads(native=native)
    fitted, skipped = fit_heads(heads, source, T, log)
    rows, manifests = [], {}
    for rx, fs in targets.items():
        man = build_manifest(fs.table, rx, T, rule, q_per_class=q_per_class, n_draws=n_draws)
        manifests[str(rx)] = dict(sha=manifest_sha(man), coverage=man['coverage'], full=man['full_coverage'])
        if not man['full_coverage']:
            log(f'rx {rx}: partial coverage, excluded from the 10-class table')
            continue
        r, logits = evaluate_target(fitted, fs, man, T, keep_logits=True)
        keys = list(logits)
        np.savez_compressed(out / f'predictions_rx{rx}.npz', scores=np.stack([logits[k] for k in keys]),
                            heads=np.array([k[0] for k in keys]), budgets=np.array([k[1] for k in keys]),
                            draws=np.array([k[2] for k in keys]), labels=fs.table.tx[fs.rows(man['query_ids'])],
                            query_ids=np.array(man['query_ids']))
        (out / f'manifest_rx{rx}.json').write_text(json.dumps(man, indent=1, default=str))
        rows += r
        log(f'rx {rx}: {len(r)} rows')
    rows += crossfit_family_rows([r for r in rows if r['head'].startswith(('H2[', 'H5c['))])
    agg = aggregate(rows)
    pairs = [p for a, b in COMPARE for m in BUDGETS if (p := paired(rows, a, b, m))]
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / 'rows.csv', index=False)
    summary = dict(skipped=skipped, manifests=manifests,
                   family_crossfit_status='available' if len(manifests) > 1 else
                   'unavailable: one encoder-held RX; cross-task encoders may have fitted this RX',
                   probability_metrics='unscaled (temperature=1); temperature fitting not executed',
                   aggregate=[dict(head=h, m=m, **v) for (h, m), v in sorted(agg.items())], paired=pairs)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    return rows, summary


def table(summary, heads=('H0', 'H0S', 'H1', 'H1e', 'H2fix', 'H2L', 'H2*', 'H3', 'H4', 'H4-AK', 'H4-pooled',
                          'H5a', 'H5b', 'H5c*')):
    agg = {(a['head'], a['m']): a for a in summary['aggregate']}
    lines = ['head      ' + ''.join(f'  m={m:<5}' for m in BUDGETS)]
    for h in heads:
        cells = [f'{100 * agg[(h, m)]["mean"]:7.1f} ' if (h, m) in agg else '      - ' for m in BUDGETS]
        lines.append(f'{h:<10}' + ''.join(cells))
    return '\n'.join(lines)


def run_legacy_wisig8(reg, res, ck, fold_name, out, n_draws, t0):
    """Stage A0: historical 8-class (TX 0-7) WiSig encoders from train_wisig_p2.py search stage.
    Their gradients used fold-fit RX positions 0:120 and the epoch was chosen on fold-fit RX positions
    120:160, so the fold's source_selection RX are encoder-held. Reported as an 8-class historical-
    compatibility diagnostic; never mixed with the 10-class main table."""
    from rffi_calibration.datasets import wisig
    from rffi_calibration.encoders import build_wisig_official, extract
    obj = torch.load(ck / 'best.pt', map_location='cpu', weights_only=False)
    if obj['args']['fold'] != fold_name:
        sys.exit(f'{ck}: checkpoint fold {obj["args"]["fold"]} != --fold {fold_name}')
    fold = reg.roles['wisig']['folds'][fold_name]
    guard = AccessGuard(reg, res, log_path=ROOT / 'reports/rffi_calibration/access_log.jsonl', stage='E4-A0')
    w = build_wisig_official(obj['model'], classes=8)
    st, xs = wisig.load(reg, guard, fold['fit'], [0, 1], (0, 120), numeric=True, purpose='E4-A0 source stats')
    tt, xt = wisig.load(reg, guard, fold['source_selection'], [0, 1], (0, 200), numeric=True, purpose='E4-A0 target')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    source = subset(FeatureSet(extract(w, xs, device=device), st, {}), st.tx < 8)
    tgt = subset(FeatureSet(extract(w, xt, device=device), tt, {}), tt.tx < 8)
    targets = {r: subset(tgt, tgt.table.rx == r) for r in np.unique(tgt.table.rx)}
    sq = reg.protocol['support_query']['wisig']
    rows, summary = run(source, targets, 8, sq['rule'], sq['q_per_class'], out, n_draws, native=w.linear_head())
    summary.update(mode='E4_A0_WISIG8_HISTORICAL', checkpoint=str(ck), checkpoint_sha256=file_sha(ck / 'best.pt'),
                   ckpt_args=obj['args'], ckpt_epoch=obj.get('epoch'), fit_rx=fold['fit'],
                   evaluated_rx=fold['source_selection'], registry_fingerprint=reg.fingerprint(),
                   seconds=time.time() - t0)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    print(table(summary))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt')
    ap.add_argument('--out', required=True)
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--draws', type=int, default=None)
    ap.add_argument('--legacy-wisig8', help='Stage A0: dir of a historical 8-class WiSig search checkpoint (best.pt)')
    ap.add_argument('--fold', choices=['w0', 'w1'])
    a = ap.parse_args()
    reg = Registry()
    proto = reg.protocol
    out = Path(a.out)
    t0 = time.time()
    if a.synthetic:
        tab, z, fit, sel = synthetic()
        fs = FeatureSet(z, tab, {})
        source = subset(fs, np.isin(tab.rx, fit) & (tab.date == 0))
        targets = {r: subset(fs, tab.rx == r) for r in sel}
        rule = {'type': 'by_date', 'support_dates': [0], 'query_dates': [1]}
        rows, summary = run(source, targets, 10, rule, 20, out, a.draws or 5)
        summary['mode'] = 'SYNTHETIC_DRY_RUN (no real data, not evidence)'
        (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
        print(table(summary))
        print(f'synthetic dry run ok: {len(rows)} rows, {time.time() - t0:.1f}s -> {out}')
        return
    res = reg.reservation()
    if not res.get('active') or not res.get('allow_feature_extraction'):
        sys.exit('Stage A real-data evaluation needs an active reservation with allow_feature_extraction; nothing was read')
    if a.legacy_wisig8:
        return run_legacy_wisig8(reg, res, Path(a.legacy_wisig8), a.fold, out, a.draws or proto['n_draws'], t0)
    ck = Path(a.ckpt)
    status = json.loads((ck / 'status.json').read_text())
    if status.get('state') != 'completed' or file_sha(ck / 'selected.pt') != status['checkpoint_sha256']:
        sys.exit(f'{ck}: checkpoint not completed or SHA mismatch')
    job = status['job']
    obj = torch.load(ck / 'selected.pt', map_location='cpu', weights_only=False)
    guard = AccessGuard(reg, res, log_path=ROOT / 'reports/rffi_calibration/access_log.jsonl', stage='E4')
    ds, T = job['dataset'], reg.num_classes(job['dataset'])
    from rffi_calibration.encoders import build_shen_featurecnn, build_wisig_official, extract
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if ds == 'wisig':
        from rffi_calibration.datasets import wisig
        fold = reg.roles['wisig']['folds'][job['split']]
        fit, sel = fold['fit'], fold['source_selection']
        w = build_wisig_official(obj['model'], classes=T)
        st, xs = wisig.load(reg, guard, fit, [0, 1], (0, 120), numeric=True, purpose='E4 source stats')
        tt, xt = wisig.load(reg, guard, sel, [0, 1], (0, 200), numeric=True, purpose='E4 target')
    else:
        from rffi_calibration.datasets import shen
        fit, sel = job['fit_rx'], job['eval_rx']
        w = build_shen_featurecnn(obj['model'])
        st, xs = shen.load(reg, guard, fit, 'fit', numeric=True, scaler=obj['scaler'], purpose='E4 source stats')
        tt, xt = shen.load(reg, guard, sel, 'all', numeric=True, scaler=obj['scaler'], purpose='E4 target')
    held = not (set(map(str, sel)) & set(map(str, fit)))
    source = FeatureSet(extract(w, xs, device=device), st, {})
    zt = extract(w, xt, device=device)
    targets = {r: subset(FeatureSet(zt, tt, {}), tt.rx == r) for r in np.unique(tt.rx)}
    sq = proto['support_query'][ds]
    rows, summary = run(source, targets, T, sq['rule'], sq['q_per_class'], out, a.draws or proto['n_draws'],
                        native=w.linear_head())
    summary.update(mode='E4_REAL' if held else 'INTERNAL_DIAGNOSTIC', checkpoint=str(ck),
                   checkpoint_sha256=status['checkpoint_sha256'], fit_rx=fit, evaluated_rx=list(map(str, sel)),
                   registry_fingerprint=reg.fingerprint(), seconds=time.time() - t0)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    print(table(summary))


if __name__ == '__main__':
    main()
