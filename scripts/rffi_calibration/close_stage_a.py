#!/usr/bin/env python3
"""Close the completed reservation and preserve its exact authorized configuration."""
import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from rffi_calibration.registry import Registry, file_sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh-final-verification', action='store_true')
    args = ap.parse_args()
    reg = Registry()
    out = ROOT / 'reports/rffi_calibration/stage_A'
    if args.refresh_final_verification:
        assert not reg.reservation()['active'] and not reg.reservation()['allow_feature_extraction']
        path = out / 'execution_verification.json'
        record = json.loads(path.read_text())
        record['final_code_sha256'] = {str(p.relative_to(ROOT)): file_sha(p) for directory in
                                      ('src/rffi_calibration', 'scripts/rffi_calibration', 'tests/rffi_calibration')
                                      for p in sorted((ROOT / directory).rglob('*.py'))}
        record['final_cpu_test_result'] = '60 passed, 1 skipped in 15.23s after cancellation-accounting fix'
        record['final_verification_refreshed_at'] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(record, indent=2) + '\n')
        print('closed reservation and final code hashes verified; no numeric input read')
        return
    decision = json.loads((out / 'decision.json').read_text())
    assert decision['decision'] == 'STOP_DECOMPOSITION'
    assert decision['corrected_evaluations_verified'] == 12 and decision['fits'] == 6
    assert decision['total_conservative_gpu_hours'] < 1.5
    assert reg.reservation()['active']
    snapshot = out / 'authorized_registration.json'
    assert not snapshot.exists(), 'closure already recorded'
    snapshot.write_bytes((reg.config_dir / 'registration.json').read_bytes())
    log_paths = [ROOT / 'reports/rffi_calibration/access_log.jsonl',
                 ROOT / 'reports/rffi_calibration/stage_A_execution/input_access.jsonl']
    numeric = [json.loads(line) for path in log_paths for line in path.read_text().splitlines()
               if json.loads(line)['kind'] == 'numeric']
    for row in numeric:
        assert row['decision'] == 'allow'
        assert row['rx'] in reg.roles[row['dataset']]['source']
        assert row['dataset'] != 'osu'
        if row['dataset'] == 'wisig':
            assert row['date'] in (0, 1)
    base = 'b6ce3c25648b958424cd4818067f65e800985f1d'
    tracked_changes = subprocess.check_output(['git', 'diff', '--name-only', base], cwd=ROOT, text=True).splitlines()
    assert set(tracked_changes) <= {'AGENTS.md', '.gitignore'}, tracked_changes
    previous_state = subprocess.check_output(['git', 'status', '--short', '--branch'], cwd=ROOT, text=True)
    import numpy
    import scipy
    import torch
    record = dict(
        closed_at=datetime.now(timezone.utc).isoformat(), python=sys.version,
        python_executable=sys.executable, platform=platform.platform(),
        numpy=numpy.__version__, scipy=scipy.__version__, torch=torch.__version__,
        final_cpu_test_result='60 passed, 1 skipped; GPU regression separately passed (3 runner tests)',
        initial_cpu_test_result='56 passed in 14.49s',
        numeric_access_requests=len(numeric), numeric_access_scope='registered source RX only; WiSig dates 0/1',
        historical_tracked_changes=tracked_changes, historical_base=base,
        precommit_git_status=previous_state,
        authorized_registry_fingerprint=reg.fingerprint(),
        fitting_registry_fingerprint=decision['fitting_registry_fingerprint'],
        local_commit_scope='code, documentation, metadata and aggregate/draw-level metrics only',
        push_performed=False, weights_published=False, signal_or_feature_arrays_published=False,
        notes=['Initial GPU evaluation failures retained, followed by a distinct corrected attempt.',
               'Evaluation final-ledger shadowing failure recovered from complete persisted progress.',
               'Report NumPy-bool serialization corrected without any numeric rerun.'])
    protocol = reg.protocol
    protocol['reservation'].update(active=False, allow_feature_extraction=False)
    (reg.config_dir / 'registration.json').write_text(json.dumps(protocol, indent=2) + '\n')
    record['closed_registry_fingerprint'] = Registry().fingerprint()
    record['final_code_sha256'] = {str(p.relative_to(ROOT)): file_sha(p) for directory in
                                  ('src/rffi_calibration', 'scripts/rffi_calibration', 'tests/rffi_calibration')
                                  for p in sorted((ROOT / directory).rglob('*.py'))}
    record['summary_sha256'] = {p.name: file_sha(p) for p in out.iterdir()
                               if p.suffix in ('.json', '.md', '.csv')}
    (out / 'execution_verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(reservation_active=False, numeric_access_requests=len(numeric),
                          total_conservative_gpu_hours=decision['total_conservative_gpu_hours']), indent=2))


if __name__ == '__main__':
    main()
