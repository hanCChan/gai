#!/usr/bin/env python3
"""Compare the CIS port against the unmodified author class on synthetic IQ."""
import argparse
import ast
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from shen_stage1.representations import channel_independent_spectrogram, iq_rms_channels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        parser.error('Use a new empty output directory')
    author_path = ROOT / 'external/receiverAgnosticRFFI/dataset_preparation.py'
    tree = ast.parse(author_path.read_text())
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'ChannelIndSpectrogram')
    # Execute only the original class AST, with its actual dependencies. This
    # avoids installing unrelated TensorFlow/natsort for a numerical reference.
    namespace = {'np': np, 'signal': signal, 'math': math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(author_path), 'exec'), namespace)
    author = namespace['ChannelIndSpectrogram']()
    rng = np.random.default_rng(20261012)
    x = rng.normal(size=(4, 8192)) + 1j * rng.normal(size=(4, 8192))
    original = x.copy()
    reference = author.channel_ind_spectrogram(x.copy()).transpose(0, 3, 1, 2)
    ours = channel_independent_spectrogram(x)
    np.testing.assert_allclose(ours, reference, rtol=1e-10, atol=1e-10)
    assert ours.shape == (4, 1, 52, 126)
    np.testing.assert_array_equal(original, x)
    iq = iq_rms_channels(x, dtype=np.float64)
    np.testing.assert_allclose(np.mean(iq[:, 0] ** 2 + iq[:, 1] ** 2, axis=1), 1, atol=1e-12)
    np.testing.assert_allclose(channel_independent_spectrogram(x * .032), ours, rtol=1e-10, atol=1e-10)
    try:
        channel_independent_spectrogram(np.zeros((1, 8192)))
    except ValueError:
        zero_rejected = True
    else:
        raise AssertionError('Zero-energy packet was not rejected')
    result = {'author_source_sha256': hashlib.sha256(author_path.read_bytes()).hexdigest(),
              'synthetic_packets': 4, 'real_research_packets_read': 0,
              'CIS_shape': list(ours.shape), 'max_absolute_difference_from_author': float(np.abs(ours - reference).max()),
              'comparison_rtol': 1e-10, 'comparison_atol': 1e-10,
              'input_unchanged': True, 'per_packet_unit_RMS': 'passed',
              'positive_amplitude_scaling_invariance': 'passed', 'zero_energy_rejected': zero_rejected,
              'additional_CFO_compensation': False, 'model_training_started': False}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'checks.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
