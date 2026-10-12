#!/usr/bin/env python3
"""Recheck strict source isolation and cached maps before interpreting A7."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from shen_stage1.data import ShenDevelopmentPackets
from shen_p2a.cache import cache_dir, write_json
from shen_p2a.representations import paired_maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preflight evidence is immutable; choose a new output path')
    protocol = ROOT / 'configs/shen_evidence/protocol.candidate.json'
    mask = np.abs(np.fft.fftfreq(8192, d=1e-6)) <= 62500.
    checks = []
    for receiver in ('n210_1', 'rtl_6'):
        for role in ('train', 'validation'):
            ds = ShenDevelopmentPackets(ROOT, protocol, receiver, role, signal_dtype=np.float64)
            folder = cache_dir(ROOT, receiver, role, strict_ib=True)
            manifest = json.loads((folder / 'manifest.json').read_text())
            assert manifest['status'] == 'completed' and manifest['strict_original_ib_projection'] is True
            np.testing.assert_array_equal(np.load(folder / 'labels.npy'), ds.labels)
            ids = [int(np.flatnonzero(ds.labels == label)[0]) for label in range(10)]
            raw = np.stack([ds[i][0][0] + 1j * ds[i][0][1] for i in ids])
            projected = np.fft.ifft(np.fft.fft(raw, axis=-1) * mask, axis=-1)
            spectrum = np.fft.fft(projected, axis=-1)
            leakage = float(np.max(np.abs(spectrum[:, ~mask])))
            scale = float(np.max(np.abs(spectrum)))
            assert leakage <= max(scale, 1.) * 1e-10
            spec, cis, _ = paired_maps(projected)
            np.testing.assert_array_equal(np.load(folder / 'spec.npy', mmap_mode='r')[ids], spec)
            np.testing.assert_array_equal(np.load(folder / 'cis.npy', mmap_mode='r')[ids], cis)
            checks.append({'receiver': receiver, 'role': role, 'ten_class_labels_exact': True,
                           'ten_class_spec_and_cis_exact': True,
                           'max_excluded_fft_magnitude': leakage,
                           'max_fft_magnitude': scale})
            ds.close()
    write_json(args.output, {'passed': True, 'checks': checks,
                             'strict_projection_before_STFT': True,
                             'final_signals_opened': 0})


if __name__ == '__main__':
    main()
