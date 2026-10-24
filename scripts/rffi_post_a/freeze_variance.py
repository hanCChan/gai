#!/usr/bin/env python3
"""Write-once RV1 protocol snapshots. No numeric data, weights or scores are opened."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from rffi_calibration.registry import file_sha
from rffi_post_a.variance import CAP, CONFIG, SEEDS, expected_jobs


def main():
    if CONFIG.exists():
        raise SystemExit('RV1 protocol already exists; no replacement')
    original = ROOT / 'configs/rffi_post_a'
    reg = json.loads((original / 'registration.json').read_text())
    if reg['reservation']['active']:
        raise SystemExit('R0 must be closed before RV1 registration')
    matrix = json.loads((original / 'pilot_matrix.json').read_text())
    matrix.update(version='RV1 paired-seed variance v1, frozen 2026-10-24', status='frozen',
                  jobs=expected_jobs(), stage='RV1', decision_rule=dict(
                      scope='two-arm stability measurement, not the original three-arm GO rule',
                      primary='new three seeds, last.pt, H1e lowshot; paired P_RX minus C_MATCH',
                      thresholds='both folds mean >=2 pp; both budgets >=0; >=6/8 RX positive; worst RX <=3 pp; H0 <=5 pp; competitive',
                      stability='stable only if the two-arm thresholds pass for each of the three new seeds',
                      not_go='rx_structure is NOT_MEASURED; never issue GO_DESIGN_SHEN',
                      outcomes=['TWO_ARM_GAIN_STABLE', 'NO_STABLE_TWO_ARM_GAIN', 'INCOMPLETE_EXECUTION']))
    matrix['evaluation_jobs_predicted_gpu_seconds'] = 24 * 25
    matrix['prediction_basis'] = 'R0 measured fits 85-104s, evaluations 10-12s; RV1 forecast ~0.4h, conservative launch forecast 12*120+24*25=2040s <2160s cap'
    reg['reservation'] = dict(
        stage='RV1', active=True,
        authorized_by='user 2026-10-24: 继续做执行吧; previous response selects twelve paired C_MATCH/P_RX fits',
        allowed_job_ids=[j['job_id'] for j in expected_jobs()], max_fits=12, max_reserved_gpu_hours=CAP,
        allow_feature_extraction=True, allow_final_numeric_access=False, allow_development_numeric_access=False,
        allow_failed_fit_retry=False, closed_after='RV1')
    reg['requests_not_active'] = {}
    CONFIG.mkdir()
    for name in ('datasets.json', 'roles.json', 'diagnostic_matrix.json'):
        (CONFIG / name).write_bytes((original / name).read_bytes())
    for name, obj in (('registration.json', reg), ('pilot_matrix.json', matrix)):
        (CONFIG / name).write_text(json.dumps(obj, indent=2) + '\n')
    immutable = [
        'configs/rffi_post_a/pilot_matrix.json', 'configs/rffi_post_a/registration.json',
        'reports/rffi_post_a/R0/REPORT.md', 'reports/rffi_post_a/R0/decision.json',
        'reports/rffi_post_a/R0/verification.json', 'reports/rffi_post_a/R0/execution_ledger.json',
        'docs/rffi_post_a/RV1_EXECUTION_PLAN.md',
    ]
    immutable += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'reports/rffi_post_a/R0/evaluation').glob('*/*'))
                  if p.name in ('rows.csv', 'meta.json')]
    code = sorted(p for d in ROOT.glob('src/rffi_*') for p in d.rglob('*.py'))
    code += sorted(ROOT.glob('scripts/rffi_post_a/*.py')) + [ROOT / 'scripts/wisig_w01_common.py']
    immutable += [str(p.relative_to(ROOT)) for p in code]
    obj = dict(stage='RV1', frozen_date='2026-10-24', seeds=list(SEEDS), max_reserved_gpu_hours=CAP,
               active_registration_sha256=file_sha(CONFIG / 'registration.json'),
               config_sha256={n: file_sha(CONFIG / n) for n in
                              ('datasets.json', 'roles.json', 'diagnostic_matrix.json', 'pilot_matrix.json')},
               immutable_inputs={n: file_sha(ROOT / n) for n in immutable},
               historical_r0_decision='NO_GAIN_IN_FIXED_PILOT',
               scope='WiSig only, same source/held RX, dates 0/1, no D1 or Stage A per-query replay')
    (CONFIG / 'preregistration.json').write_text(json.dumps(obj, indent=2) + '\n')
    print(json.dumps(dict(stage='RV1', fits=12, seeds=SEEDS, cap_gpu_hours=CAP, config=str(CONFIG))))


if __name__ == '__main__':
    main()
