"""RV1's finite paired-seed protocol; the completed R0 registration is never activated."""
from __future__ import annotations

import json
from pathlib import Path

from rffi_calibration.registry import file_sha

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / 'configs/rffi_post_a/variance_v1'
OUT = ROOT / 'reports/rffi_post_a/RV1'
FITS = ROOT / 'outputs/rffi_post_a/RV1'
SEEDS = (20261025, 20261026, 20261027)
ARMS = ('C_MATCH', 'P_RX')
CAP = 0.6


def expected_jobs():
    return [dict(job_id=f'RV1_wisig_{split}_{arm}_s{seed}', split=split, arm=arm, seed=seed,
                 predicted_gpu_seconds=120)
            for split in ('w0', 'w1') for seed in SEEDS for arm in ARMS]


def check_reservation(res, matrix):
    problems = []
    if not res.get('active') or res.get('stage') != 'RV1':
        problems.append('RV1 reservation inactive')
    if matrix.get('jobs', []) != expected_jobs():
        problems.append('matrix is not the twelve frozen RV1 jobs')
    ids = [j['job_id'] for j in expected_jobs()]
    if sorted(res.get('allowed_job_ids', [])) != sorted(ids) or res.get('max_fits') != 12:
        problems.append('allowlist/max_fits is not exactly twelve')
    if float(res.get('max_reserved_gpu_hours', 0)) != CAP:
        problems.append('cap must be 0.6 A100 GPUh')
    if not res.get('allow_feature_extraction'):
        problems.append('feature extraction not allowed')
    if any(res.get(k) for k in ('allow_failed_fit_retry', 'allow_final_numeric_access',
                               'allow_development_numeric_access')):
        problems.append('retry/final/development must be false')
    if problems:
        raise SystemExit('; '.join(problems) + '; nothing was read')


def verify_preregistration(config=CONFIG):
    p = json.loads((config / 'preregistration.json').read_text())
    for rel, sha in p['immutable_inputs'].items():
        if file_sha(ROOT / rel) != sha:
            raise ValueError(f'preregistered input changed: {rel}')
    for name, sha in p['config_sha256'].items():
        if file_sha(config / name) != sha:
            raise ValueError(f'RV1 config changed: {name}')
    matrix = json.loads((config / 'pilot_matrix.json').read_text())
    original = json.loads((ROOT / 'configs/rffi_post_a/pilot_matrix.json').read_text())
    for key in ('recipe', 'evaluation', 'dataset', 'num_classes'):
        if matrix[key] != original[key]:
            raise ValueError(f'RV1 does not match the original R0 {key}')
    for name in ('datasets.json', 'roles.json', 'diagnostic_matrix.json'):
        if (config / name).read_bytes() != (ROOT / 'configs/rffi_post_a' / name).read_bytes():
            raise ValueError(f'RV1 changed the data scope: {name}')
    if file_sha(config / 'registration.json') != p['active_registration_sha256']:
        raise ValueError('RV1 active registration changed')
    if json.loads((ROOT / 'configs/rffi_post_a/registration.json').read_text())['reservation']['active']:
        raise ValueError('the original R0 reservation must remain closed')
    return p


def cross_pair_checks(statuses, matrix):
    out = {}
    for split in ('w0', 'w1'):
        for seed in SEEDS:
            js = [j for j in matrix['jobs'] if (j['split'], j['seed']) == (split, seed)]
            ss = [statuses[j['job_id']] for j in js if j['job_id'] in statuses]
            out[f'{split}|{seed}'] = dict(
                n=len(ss), same_init=len(ss) == 2 and len({s['init_state_sha256'] for s in ss}) == 1,
                same_all_ids=len(ss) == 2 and len({s['all_ids_chain_sha256'] for s in ss}) == 1)
    return out
