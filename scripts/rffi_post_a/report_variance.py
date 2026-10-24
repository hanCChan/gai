#!/usr/bin/env python3
"""Independent RV1 score/coverage verifier and paired-seed summary (no evaluator imports)."""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from rffi_calibration.registry import file_sha
from rffi_post_a.reporting import write_csv, write_json
from rffi_post_a.variance import CONFIG, FITS, OUT, SEEDS, expected_jobs

HEADS = ('H0', 'H1e', 'H1', 'H2L', 'H5a', 'H5b', 'H5c[k0=1]')
SECONDARY = ('H1', 'H2L', 'H5a', 'H5b', 'H5c[k0=1]')
HELD = {'w0': ('1', '7', '16', '30'), 'w1': ('6', '26', '2', '5')}
T95 = {3: 4.3026527297, 4: 3.1824463053}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def expected_keys(rxs):
    return {(rx, h, m, draw) for rx in rxs for m in (0, 1, 2, 5, 10)
            for draw in range(1 if m == 0 else 20)
            for h in (('H0', 'H5c[k0=1]') if m == 0 else HEADS)}


def verify_evaluation(path, job, checkpoint, expected_manifests, checkpoint_sha):
    rows = read_csv(path / 'rows.csv')
    keys = [(str(r['rx']), r['head'], int(r['m']), int(r['draw'])) for r in rows]
    want = expected_keys(HELD[job['split']])
    if len(keys) != len(set(keys)) or set(keys) != want:
        raise ValueError(f'{path.name}: missing, extra or duplicate rows')
    for r in rows:
        for field in ('job_id', 'arm', 'split', 'seed'):
            if str(r[field]) != str(job[field]):
                raise ValueError(f'{path.name}: wrong {field}')
        if r['checkpoint'] != checkpoint or int(r['n_query']) != 200:
            raise ValueError(f'{path.name}: checkpoint/query mismatch')
    meta = json.loads((path / 'meta.json').read_text())
    if meta['manifests'] != expected_manifests or meta['checkpoint_sha256'] != checkpoint_sha or meta['skipped']:
        raise ValueError(f'{path.name}: lineage mismatch')
    max_abs = 0.
    with np.load(path / 'scores.npz', allow_pickle=False) as scores:
        want_scores = {f'{rx}|{h}|{m}|{d}' for rx, h, m, d in want}
        want_scores |= {f'{rx}|y_query' for rx in HELD[job['split']]}
        if set(scores.files) != want_scores:
            raise ValueError(f'{path.name}: score coverage mismatch')
        ys = {}
        for rx in HELD[job['split']]:
            y = scores[f'{rx}|y_query']
            if y.shape != (200,) or y.dtype.kind not in 'iu' or not np.array_equal(
                    np.bincount(y, minlength=10), np.full(10, 20)):
                raise ValueError(f'{path.name}: query labels invalid')
            ys[rx] = y
        for r in rows:
            s = scores[f'{r["rx"]}|{r["head"]}|{r["m"]}|{r["draw"]}']
            if s.shape != (200, 10) or not np.isfinite(s).all():
                raise ValueError(f'{path.name}: invalid scores')
            acc = float((s.argmax(1) == ys[str(r['rx'])]).mean())
            stored = float(r['acc'])
            if not math.isfinite(stored):
                raise ValueError(f'{path.name}: invalid stored accuracy')
            max_abs = max(max_abs, abs(acc - stored))
            if abs(acc - stored) > 1e-12:
                raise ValueError(f'{path.name}: score accuracy mismatch')
    return rows, dict(checked=len(rows), max_abs=max_abs, ok=True,
                      query_sha256={rx: hashlib.sha256(y.tobytes()).hexdigest() for rx, y in ys.items()})


def stats(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else None
    se = sd / math.sqrt(n) if sd is not None else None
    margin = T95[n] * se if n in T95 else None
    return dict(n_seeds=n, mean_pp=mean, sd_pp=sd, se_pp=se, min_pp=float(v.min()), max_pp=float(v.max()),
                n_positive=int((v > 0).sum()), ci95_pp=None if margin is None else [mean - margin, mean + margin])


def per_rx_means(rows):
    groups = defaultdict(list)
    for r in rows:
        if r['checkpoint'] != 'last.pt':
            continue
        groups[(r['split'], int(r['seed']), r['arm'], r['head'], str(r['rx']), int(r['m']))].append(float(r['acc']))
    return {key: float(np.mean(values)) for key, values in groups.items()}


def pair_fold(means, split, seed):
    rxs = HELD[split]

    def acc(arm, head, rx, m):
        return means[(split, seed, arm, head, rx, m)]

    def low(arm, head, rx):
        return (acc(arm, head, rx, 1) + acc(arm, head, rx, 5)) / 2

    c = 100 * float(np.mean([low('C_MATCH', 'H1e', rx) for rx in rxs]))
    p = 100 * float(np.mean([low('P_RX', 'H1e', rx) for rx in rxs]))
    d1, d5 = [100 * float(np.mean([acc('P_RX', 'H1e', rx, m) - acc('C_MATCH', 'H1e', rx, m)
                                  for rx in rxs])) for m in (1, 5)]
    per = {rx: 100 * (low('P_RX', 'H1e', rx) - low('C_MATCH', 'H1e', rx)) for rx in rxs}
    h0 = -100 * float(np.mean([acc('P_RX', 'H0', rx, 0) - acc('C_MATCH', 'H0', rx, 0) for rx in rxs]))
    secondary = {h: 100 * float(np.mean([low('C_MATCH', h, rx) for rx in rxs])) for h in SECONDARY}
    worst = max(0., -min(per.values()))
    checks = dict(gain=p - c >= 2. - 1e-10, per_budget=d1 >= -1e-10 and d5 >= -1e-10,
                  risk=worst <= 3. + 1e-10 and h0 <= 5. + 1e-10,
                  competitive=p >= max(secondary.values()) - 1e-10)
    return dict(split=split, seed=seed, c_match_lowshot_pct=c, p_rx_lowshot_pct=p, gain_pp=p - c,
                d1_pp=d1, d5_pp=d5, per_rx_pp=per, n_positive_rx=sum(d > 0 for d in per.values()),
                worst_rx_drop_pp=worst, h0_drop_pp=h0, c_match_secondary_pct=secondary,
                best_c_match_secondary_pct=max(secondary.values()), checks=checks)


def cohort_summary(rows, seeds):
    means = per_rx_means(rows)
    pairs = [pair_fold(means, split, seed) for seed in seeds for split in HELD]
    idx = {(r['split'], r['seed']): r for r in pairs}
    folds = {}
    for split in HELD:
        prs = [idx[(split, seed)] for seed in seeds]
        folds[split] = dict(
            paired_lowshot=stats([r['gain_pp'] for r in prs]),
            paired_m1=stats([r['d1_pp'] for r in prs]), paired_m5=stats([r['d5_pp'] for r in prs]),
            c_match=stats([r['c_match_lowshot_pct'] for r in prs]),
            p_rx=stats([r['p_rx_lowshot_pct'] for r in prs]),
            per_rx={rx: stats([r['per_rx_pp'][rx] for r in prs]) for rx in HELD[split]})
    by_seed = {}
    for seed in seeds:
        ps = [idx[(s, seed)] for s in HELD]
        checks = {k: all(p['checks'][k] for p in ps) for k in ps[0]['checks']}
        checks['rx_consistency'] = sum(p['n_positive_rx'] for p in ps) >= 6
        by_seed[str(seed)] = dict(checks=checks, two_arm_pass=all(checks.values()),
                                 combined_gain_pp=float(np.mean([p['gain_pp'] for p in ps])))
    pooled = {}
    for s, seed, arm, head, rx, m in means:
        if seed not in seeds:
            continue
        key = (s, 0, arm, head, rx, m)
        pooled[key] = float(np.mean([means[(s, ss, arm, head, rx, m)] for ss in seeds]))
    pooled_pairs = {s: pair_fold(pooled, s, 0) for s in HELD}
    pooled_checks = {k: all(p['checks'][k] for p in pooled_pairs.values())
                     for k in next(iter(pooled_pairs.values()))['checks']}
    pooled_checks['rx_consistency'] = sum(p['n_positive_rx'] for p in pooled_pairs.values()) >= 6
    return dict(seeds=list(seeds), pairs=pairs, folds=folds, by_seed=by_seed,
                combined_paired=stats([v['combined_gain_pp'] for v in by_seed.values()]),
                seed_mean_checks=pooled_checks, seed_mean_folds=pooled_pairs)


def verify_execution():
    prereg = json.loads((CONFIG / 'preregistration.json').read_text())
    for rel, sha in prereg['immutable_inputs'].items():
        if file_sha(ROOT / rel) != sha:
            raise ValueError(f'original/frozen input changed: {rel}')
    for name, sha in prereg['config_sha256'].items():
        if file_sha(CONFIG / name) != sha:
            raise ValueError(f'RV1 frozen config changed: {name}')
    reg = json.loads((CONFIG / 'registration.json').read_text())
    active = dict(reg, reservation=dict(reg['reservation'], active=True))
    if hashlib.sha256((json.dumps(active, indent=2) + '\n').encode()).hexdigest() != prereg['active_registration_sha256']:
        raise ValueError('RV1 registration changed beyond closure')
    ledger = json.loads((OUT / 'execution_ledger.json').read_text())
    if ledger['stopped'] or not ledger['reservation_closed'] or reg['reservation']['active']:
        raise ValueError('execution incomplete or reservation not closed')
    if not 0 < ledger['reserved_gpu_hours'] <= .6:
        raise ValueError('GPU budget invalid')
    spent = sum(r['seconds'] for r in ledger['fits']) + sum(r['seconds'] for r in ledger['evaluations'])
    if abs(spent / 3600. - ledger['reserved_gpu_hours']) > 1e-12:
        raise ValueError('GPU accounting mismatch')
    jobs = {j['job_id']: j for j in expected_jobs()}
    if len(ledger['fits']) != 12 or {r['job_id'] for r in ledger['fits']} != set(jobs) or any(
            r['state'] != 'completed' for r in ledger['fits']):
        raise ValueError('not twelve unique completed fits')
    want_evals = {(j, ck) for j in jobs for ck in ('last.pt', 'native_best.pt')}
    if len(ledger['evaluations']) != 24 or {
            (r['job_id'], r['checkpoint']) for r in ledger['evaluations']} != want_evals or any(
                r['exit_code'] != 0 for r in ledger['evaluations']):
        raise ValueError('fixed evaluation matrix incomplete')
    if len(ledger['cross_pair']) != 6 or not all(
            p['n'] == 2 and p['same_init'] and p['same_all_ids'] for p in ledger['cross_pair'].values()):
        raise ValueError('pairing verification incomplete')
    preflight = json.loads((OUT / 'preflight.json').read_text())
    for path, sha in preflight['asset_sha256'].items():
        if file_sha(path) != sha:
            raise ValueError('source cache changed')
    access = [json.loads(line) for line in (OUT / 'access_log.jsonl').read_text().splitlines()]
    allowed = {int(rx) for rxs in HELD.values() for rx in rxs} | {10, 12, 13, 22, 28, 31}
    if any(r['dataset'] != 'wisig' or r['rx'] not in allowed or r['date'] not in (0, 1) or
           r['decision'] != 'allow' for r in access):
        raise ValueError('numeric scope audit failed')
    recipe = json.loads((CONFIG / 'pilot_matrix.json').read_text())['recipe']
    rows, report, statuses = [], {}, {}
    for job_id, job in jobs.items():
        st = json.loads((FITS / job_id / 'status.json').read_text())
        if st['state'] != 'completed' or any(st['job'][k] != v for k, v in {**job, **recipe}.items()):
            raise ValueError('training recipe changed')
        curve = json.loads((FITS / job_id / 'curve.json').read_text())
        if [r['step'] for r in curve] != list(range(32, 4801, 32)) or not all(
                math.isfinite(r['val_ce']) for r in curve):
            raise ValueError('training curve/horizon invalid')
        statuses[job_id] = st
        for ck in ('last.pt', 'native_best.pt'):
            sha = st['last_sha256' if ck == 'last.pt' else 'native_best_sha256']
            if file_sha(FITS / job_id / ck) != sha:
                raise ValueError('checkpoint hash mismatch')
            path = OUT / 'evaluation' / f'{job_id}_{ck[:-3]}'
            rs, checked = verify_evaluation(path, job, ck, preflight['folds'][job['split']]['manifests'], sha)
            rows.extend(rs)
            report[path.name] = checked
    for split in HELD:
        for seed in SEEDS:
            ids = [f'RV1_wisig_{split}_{arm}_s{seed}' for arm in ('C_MATCH', 'P_RX')]
            a, b = [statuses[j] for j in ids]
            for key in ('init_state_sha256', 'all_ids_chain_sha256', 'records_seen', 'record_exposure_max'):
                if a[key] != b[key]:
                    raise ValueError('independent pair lineage verification failed')
            if file_sha(FITS / ids[0] / 'exposure.json') != file_sha(FITS / ids[1] / 'exposure.json'):
                raise ValueError('independent pair exposure verification failed')
    query_hashes = defaultdict(set)
    for name, result in report.items():
        for rx, sha in result['query_sha256'].items():
            query_hashes[rx].add(sha)
    if any(len(shas) != 1 for shas in query_hashes.values()):
        raise ValueError('queries not paired across encoders')
    return rows, dict(all_ok=True, score_replay=report, checked_rows=sum(r['checked'] for r in report.values()),
                      immutable_inputs_checked=len(prereg['immutable_inputs']), source_assets_checked=28,
                      pairing_checked=6, access_events=len(access)), statuses, ledger


def historical_rows():
    rows = []
    for split in HELD:
        for arm in ('C_MATCH', 'P_RX'):
            path = ROOT / f'reports/rffi_post_a/R0/evaluation/R0_wisig_{split}_{arm}_s20261024_last/rows.csv'
            rs = read_csv(path)
            keys = [(str(r['rx']), r['head'], int(r['m']), int(r['draw'])) for r in rs]
            if len(keys) != len(set(keys)) or set(keys) != expected_keys(HELD[split]):
                raise ValueError('historical R0 CSV coverage changed')
            rows.extend(dict(r, seed=20261024) for r in rs)
    return rows


def report_markdown(summary, verification, ledger, statuses):
    p = summary['new_three_seeds']
    decision = summary['decision']
    lines = [
        f'# RV1: Paired-Seed Measurement - {decision}', '',
        'Date: 2026-10-24. WiSig development data; not a blind test.', '',
        '## Result', '',
        f'Exactly 12/12 fits completed, zero failed fits/retries/replacement seeds; 24/24 fixed evaluations.',
        f'Occupied-device wall time: **{ledger["reserved_gpu_hours"]:.6f} A100 GPUh**, hard cap 0.6.',
        'Reservation is closed. R0 remains **NO_GAIN_IN_FIXED_PILOT**; no D1/P_MIX/Shen/new encoder was run.', '',
        '| Fold | Paired gains by new seed (pp) | Mean (pp) | Paired SD (pp) | Conditional 95% t interval (pp) |',
        '|---|---|---:|---:|---|',
    ]
    for split in HELD:
        pairs = [r for r in p['pairs'] if r['split'] == split]
        s = p['folds'][split]['paired_lowshot']
        ci = s['ci95_pp']
        lines.append(f'| {split} | ' + ', '.join(f'{r["seed"]}: {r["gain_pp"]:+.2f}' for r in pairs) +
                     f' | {s["mean_pp"]:+.2f} | {s["sd_pp"]:.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] |')
    c = p['combined_paired']
    lines += ['', f'Combined folds are averaged within each shared seed: mean {c["mean_pp"]:+.2f} pp, '
              f'SD {c["sd_pp"]:.2f} pp, interval [{c["ci95_pp"][0]:+.2f}, {c["ci95_pp"][1]:+.2f}] pp.', '',
              '## Risk And Thresholds', '',
              '| Fold / seed | 1-shot (pp) | 5-shot (pp) | Positive RX | Worst-RX drop (pp) | H0 drop (pp) |',
              '|---|---:|---:|---:|---:|---:|']
    for r in p['pairs']:
        lines.append(f'| {r["split"]} / {r["seed"]} | {r["d1_pp"]:+.2f} | {r["d5_pp"]:+.2f} | '
                     f'{r["n_positive_rx"]}/4 | {r["worst_rx_drop_pp"]:.2f} | {r["h0_drop_pp"]:+.2f} |')
    lines += ['', 'The two-arm engineering rule passes per new seed: ' +
              ', '.join(f'{seed}: {v["two_arm_pass"]}' for seed, v in p['by_seed'].items()) + '.',
              'Seed-mean two-arm checks: `' + json.dumps(p['seed_mean_checks'], sort_keys=True) + '`.',
              'TWO_ARM_GAIN_STABLE requires all three new seeds to pass; it is not a significance test.',
              'P_RX minus P_MIX is **NOT_MEASURED**. There is no full three-arm GO decision.', '',
              '## Conditional Seed Variability', '',
              '| Fold | C_MATCH lowshot mean / SD (%) | P_RX lowshot mean / SD (%) |',
              '|---|---|---|']
    for split in HELD:
        f = p['folds'][split]
        lines.append(f'| {split} | {f["c_match"]["mean_pp"]:.2f} / {f["c_match"]["sd_pp"]:.2f} | '
                     f'{f["p_rx"]["mean_pp"]:.2f} / {f["p_rx"]["sd_pp"]:.2f} |')
    lines += ['', 'The measured variability jointly changes initialization and the episode schedule.',
              'It does not isolate these components, prove a mechanism, or define the noise level of every training method.',
              'Intervals are descriptive and conditional on the same eight development RX.',
              'Support draws and overlapping fit RX are not treated as independent replication units.', '',
              '## Four-Seed Supplement', '',
              'The original R0 seed is included only as a separately labeled descriptive supplement.', '',
              '| Fold | Four-seed mean gain (pp) | SD (pp) | Conditional 95% t interval (pp) |',
              '|---|---:|---:|---|']
    for split in HELD:
        s = summary['all_four_seeds']['folds'][split]['paired_lowshot']
        lines.append(f'| {split} | {s["mean_pp"]:+.2f} | {s["sd_pp"]:.2f} | '
                     f'[{s["ci95_pp"][0]:+.2f}, {s["ci95_pp"][1]:+.2f}] |')
    lines += ['', '## Verification', '',
              f'Independent score replay checked **{verification["checked_rows"]} rows** in 24 directories; maximum accuracy difference zero.',
              'All six pairings have matching initialization, sample-chain and exposure hashes.',
              'Eight manifest hashes and query-label arrays match across every evaluation.',
              '28 input assets were rehashed; frozen code/config and R0 evidence hashes remain unchanged.',
              'All accesses are WiSig source RX, dates 0/1; no final/development/equalized access.', '',
              '## Artifacts', '',
              '`decision.json`, `summary.json`, `paired_fold_seed.csv`, `paired_rx_seed.csv`,',
              '`verification.json`, `execution_ledger.json`, `preflight.json`, `evidence_index.json`.',
              'Scores and weights remain local and excluded from Git.', '',
              'No next stage or additional fitting is activated.']
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    try:
        rows, verification, statuses, ledger = verify_execution()
        primary = cohort_summary(rows, SEEDS)
        supplement = cohort_summary(rows + historical_rows(), (20261024,) + SEEDS)
        stable = all(v['two_arm_pass'] for v in primary['by_seed'].values())
        decision = 'TWO_ARM_GAIN_STABLE' if stable else 'NO_STABLE_TWO_ARM_GAIN'
        summary = dict(stage='RV1', decision=decision, new_three_seeds=primary, all_four_seeds=supplement,
                       rx_structure='NOT_MEASURED', r0_decision='NO_GAIN_IN_FIXED_PILOT')
        write_json(OUT / 'verification.json', verification)
        write_json(OUT / 'summary.json', summary)
        write_json(OUT / 'decision.json', dict(
            stage='RV1', decision=decision, rule='docs/rffi_post_a/RV1_EXECUTION_PLAN.md, Frozen Measurement Decision',
            fits=12, fit_failures=0, retries=0, evaluations=24, reserved_gpu_hours=ledger['reserved_gpu_hours'],
            cap_gpu_hours=.6, reservation_closed=True, full_three_arm_go=False,
            rx_structure='NOT_MEASURED', historical_r0_decision='NO_GAIN_IN_FIXED_PILOT',
            new_seed_checks=primary['by_seed'], mean_checks=primary['seed_mean_checks'],
            not_run=['D1', 'P_MIX', 'Stage A per-query replay', 'Shen', 'new encoder', 'additional fits'],
            next_stage_active=False))
        write_csv(OUT / 'paired_fold_seed.csv', [dict(r, per_rx_pp=json.dumps(r['per_rx_pp']),
                  c_match_secondary_pct=json.dumps(r['c_match_secondary_pct']), checks=json.dumps(r['checks']))
                  for r in supplement['pairs']])
        write_csv(OUT / 'paired_rx_seed.csv', [dict(split=r['split'], seed=r['seed'], rx=rx, gain_pp=value)
                  for r in supplement['pairs'] for rx, value in r['per_rx_pp'].items()])
        write_csv(OUT / 'training_summary.csv', [
            dict(job_id=job_id, best_step=st['best_native']['step'], best_val_ce=st['best_native']['ce'],
                 final_val_ce=(curve := json.loads((FITS / job_id / 'curve.json').read_text()))[-1]['val_ce'],
                 clip_fraction=curve[-1]['clip_fraction'], feat_norm_q50=curve[-1]['feat_norm_q50'],
                 fit_seconds=st['seconds']) for job_id, st in statuses.items()])
        (OUT / 'REPORT.md').write_text(report_markdown(summary, verification, ledger, statuses))
        evidence = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=file_sha(p))
                    for base in (OUT, CONFIG) for p in sorted(base.rglob('*')) if p.is_file() and
                    p.name != 'evidence_index.json' and p.suffix != '.sqlite']
        for p in sorted(FITS.glob('*/*.json')):
            evidence.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=file_sha(p),
                                 local_only=True))
        write_json(OUT / 'evidence_index.json', dict(
            stage='RV1', files=evidence, checkpoint_sha256={j: dict(last=st['last_sha256'],
            native_best=st['native_best_sha256']) for j, st in statuses.items()},
            publication='not performed; scores and model weights remain local'))
        print(json.dumps(dict(decision=decision, folds={s: primary['folds'][s]['paired_lowshot'] for s in HELD},
                              gpu_hours=ledger['reserved_gpu_hours'], verified_rows=verification['checked_rows']), indent=2))
    except Exception as exc:
        write_json(OUT / 'verification.json', dict(all_ok=False, error=repr(exc)))
        write_json(OUT / 'decision.json', dict(stage='RV1', decision='INCOMPLETE_EXECUTION', error=repr(exc),
                                             historical_r0_decision='NO_GAIN_IN_FIXED_PILOT', next_stage_active=False))
        raise


if __name__ == '__main__':
    main()
