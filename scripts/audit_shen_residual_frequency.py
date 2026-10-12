#!/usr/bin/env python3
"""Development-only residual phase audit; parameters are explicit assumptions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shen_stage1.frequency import periodic_phase_offset
from shen_stage1.protocol import DEVELOPMENT_FILES, development_path


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"finite_count": 0}
    return {"finite_count": len(values), "min": float(values.min()), "median": float(np.median(values)),
            "max": float(values.max()), "median_absolute": float(np.median(np.abs(values))),
            "p95_absolute": float(np.quantile(np.abs(values), .95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--audit', type=Path, default=ROOT / 'inventory/shen_stage1_audit_20261011/audit.json')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--assumed-sample-rate-hz', type=float, default=1_000_000)
    parser.add_argument('--assumed-chirp-samples', type=int, default=1024)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    data_root = args.root.resolve() / 'data/external/shen_receiver_agnostic'
    if out.is_relative_to(data_root.resolve()) or (out.exists() and any(out.iterdir())):
        parser.error('Use a new output directory outside original data')
    old = json.loads(args.audit.read_text())
    # No signal reads when the audit's file fingerprints have become stale.
    changed = []
    for item in old['files']:
        path = args.root / item['path']
        st = path.stat()
        if st.st_size != item['bytes'] or st.st_mtime_ns != item['mtime_ns']:
            changed.append(item['path'])
    if changed:
        raise RuntimeError(f'Files changed since previous SHA256 audit: {changed}')
    out.mkdir(parents=True, exist_ok=True)
    summaries, per_label = [], []
    columns = ['file', 'receiver', 'role', 'row_index', 'original_label', 'stored_CFO_value_unit_unknown',
               'stored_SNR_value_unit_unconfirmed', 'all_pairs_offset_hz_under_assumed_fs',
               'first_pair_offset_hz_under_assumed_fs', 'offset_cycles_per_sample_under_assumed_lag',
               'coherence', 'pair_phase_circular_std_hz_under_assumed_fs', 'valid']
    with (out / 'development_packet_diagnostics.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for item in old['files']:
            rel = item['dataset_relative_path']
            if rel not in DEVELOPMENT_FILES:
                continue
            path = development_path(data_root, rel)
            packets = []
            with h5py.File(path, 'r') as f:
                for name in ['data', 'label', 'CFO', 'SNR']:
                    if not isinstance(f.get(name, getlink=True), h5py.HardLink):
                        raise ValueError(f'Unsupported link in {rel}: {name}')
                labels = f['label'][:].reshape(-1).astype(int)
                cfo, snr = f['CFO'][:].reshape(-1), f['SNR'][:].reshape(-1)
                for start in range(0, len(labels), 128):
                    rows = np.asarray(f['data'][start:start + 128], dtype=np.float64)
                    z = rows[:, :8192] + 1j * rows[:, 8192:]
                    diagnostic = periodic_phase_offset(z, args.assumed_chirp_samples, args.assumed_sample_rate_hz)
                    for i in range(len(rows)):
                        idx = start + i
                        row = {'file': rel, 'receiver': item['receiver'], 'role': item['role'],
                               'row_index': idx, 'original_label': int(labels[idx]),
                               'stored_CFO_value_unit_unknown': float(cfo[idx]),
                               'stored_SNR_value_unit_unconfirmed': float(snr[idx])}
                        for key, values in diagnostic.items():
                            row[key] = bool(values[i]) if key == 'valid' else float(values[i])
                        writer.writerow(row)
                        packets.append(row)
            def summarize(selected):
                return {'packets': len(selected), 'invalid_estimates': sum(not x['valid'] for x in selected),
                        'phase_offset_hz_under_assumptions': stats([x['all_pairs_offset_hz_under_assumed_fs'] for x in selected]),
                        'coherence': stats([x['coherence'] for x in selected]),
                        'fraction_coherence_ge_0_9': float(np.mean([x['coherence'] >= .9 for x in selected])),
                        'stored_CFO_values_unit_unknown': stats([x['stored_CFO_value_unit_unknown'] for x in selected])}
            summary = {'file': rel, 'receiver': item['receiver'], 'role': item['role'], **summarize(packets)}
            summaries.append(summary)
            for label in range(31, 41):
                per_label.append({'file': rel, 'label': label, **summarize([x for x in packets if x['original_label'] == label])})
            print(json.dumps(summary), flush=True)
    # Verify that the read-only diagnostic did not alter source metadata.
    for item in old['files']:
        st = (args.root / item['path']).stat()
        if (st.st_size, st.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
            raise RuntimeError('Original data changed during audit')
    report = {'checked_utc': datetime.now(timezone.utc).isoformat(),
              'method': 'phase(conj(previous)*following); first-pair Shen Eq.13 and aggregate seven adjacent pairs',
              'assumed_parameters': {'sampling_rate_hz': args.assumed_sample_rate_hz,
                                     'chirp_samples': args.assumed_chirp_samples, 'local_verified': False},
              'principal_half_range_hz': args.assumed_sample_rate_hz / (2 * args.assumed_chirp_samples),
              'limitations': ['Modulo Fs/lag; no coarse/unwrapped CFO recovery',
                              'Lag corresponds to chirp duration only if actual parameters support it',
                              'Periodicity/coherence does not prove synchronization, file provenance or compensation history',
                              'CFO/SNR metadata units and their pre/post-compensation stage remain unknown',
                              'Do not infer transmitter-only drift from relative Tx/Rx frequency estimates'],
              'original_IQ_modified': False, 'additional_CFO_compensation_applied': False,
              'final_test_signal_accessed': False, 'model_training_started': False,
              'previous_audit_sha256': hashlib.sha256(args.audit.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'files': summaries, 'by_file_and_label': per_label}
    (out / 'residual_frequency_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    lines = ['# Shen 开发数据的条件频偏诊断', '',
             '只读检查两台接收机的 Day1 train、Day1 test、Day2 test；没有训练模型、再次补偿 CFO 或读取 Day3–4 信号。', '',
             '假定 Fs=1 MHz、每 chirp 1024 点。它们来自文献，尚未证明适用于本地文件。采用相邻重复片段的复相关相位，保存首对（对应原论文式13）及全部七对合并的估计。', '',
             '| 文件 | 包数 | 估计绝对值中位数（条件 Hz） | 绝对值 P95（条件 Hz） | 相干度中位数 |',
             '|---|---:|---:|---:|---:|']
    for s in summaries:
        fstats = s['phase_offset_hz_under_assumptions']
        lines.append(f"| {s['file']} | {s['packets']} | {fstats.get('median_absolute', float('nan')):.4f} | {fstats.get('p95_absolute', float('nan')):.4f} | {s['coherence'].get('median', float('nan')):.5f} |")
    lines += ['', f"此估计仅覆盖主值范围 ±{report['principal_half_range_hz']:.5f} Hz，频偏相差 Fs/lag 时会产生相同估计。例如在假定参数下，实际 1000 Hz 可估为 23.4375 Hz。小估计值不能独立证明补偿过 CFO。", '',
              'CFO 元数据保持原始数值，单位和记录阶段未确认，没有将它与条件 Hz 直接相减，也没有用它恢复补偿前的 IQ。', '',
              '论文支持作者先处理再保存的流程；本地文件来源、实际参数、日期/采集组和补偿历史仍需原始发布包或处理记录核验。此报告不能将本地文件升级为“已证实的 CFO 补偿后原版”。', '',
              '来源：[频偏估计式13–15](https://arxiv.org/pdf/2101.01668)、[处理后保存流程，Section III-B](https://arxiv.org/pdf/2207.02999)。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines), encoding='utf8')


if __name__ == '__main__':
    main()
