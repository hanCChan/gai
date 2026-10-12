#!/usr/bin/env python3
"""Independent result verification; does not refit or choose a new candidate.

Recompute packet aggregation from the 80000 exported trials, rederive source
responses by least squares, and independently transform 100 raw IQ packets.
The registered audit code and thresholds are never altered.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def independent_cis(iq):
    x = iq / np.sqrt(np.mean(np.abs(iq)**2, axis=-1, keepdims=True))
    s = signal.stft(x, window='boxcar', nperseg=128, noverlap=64, nfft=128,
                    return_onesided=False, padded=False, boundary=None, axis=-1)[2]
    s = np.fft.fftshift(s, axes=-2) + 1e-12
    return np.log10(np.abs(s[:, 38:90, 1:] / s[:, 38:90, :-1])**2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='outputs/virp/v1_v0_20261012')
    args = parser.parse_args()
    out = ROOT / args.output
    registration = ROOT / 'configs/virp/v1_registration.json'
    reg = json.loads(registration.read_text())
    seal = json.loads(registration.with_suffix('.seal.json').read_text())
    assert sha(registration) == seal['registration_sha256']
    assert sha(ROOT / 'docs/virp/USER_RULES_20261012.txt') == reg['user_rules_sha256']
    for path, expected in reg['immutable_code_sha256'].items():
        assert sha(ROOT / path) == expected
    assert json.loads((out / 'status.json').read_text())['status'] == 'completed'
    access = json.loads((out / 'signal_access_log.json').read_text())
    assert len(access) == 10 and {r['receiver'] for r in access} == set(reg['source_receivers'])
    assert all(r['role'] == 'source_train_raw' and r['mode'] == 'read_only' for r in access)
    f = np.fft.fftfreq(8192)
    b = np.stack([fn(2 * np.pi * h * f) for h in (1, 2, 3) for fn in (np.cos, np.sin)], axis=1)
    checks = []
    raw_packet_checks = 0
    trial_count = packet_count = 0
    max_delta_error = max_response_error = 0.
    for fold in (0, 1):
        folder = out / ('fold' + str(fold))
        fit = json.loads((folder / 'fit.frozen.json').read_text())
        decision = json.loads((folder / 'decision.json').read_text())
        assert not set(fit['fit_receivers']) & set(fit['heldout_receivers'])
        assert set(fit['fit_receivers']) | set(fit['heldout_receivers']) == set(reg['source_receivers'])
        assert sha(folder / 'responses.frozen.npz') == fit['response_file_sha256']
        stats = {}
        for rx in fit['fit_receivers']:
            with np.load(out / 'raw_source' / rx / 'statistics.npz') as z:
                stats[rx] = z['profiles'].copy()
        responses = np.load(folder / 'responses.frozen.npz')
        trial_by_packet = {}
        all_deltas, all_correlations = [], []
        for rx in fit['fit_receivers']:
            trials = read_csv(folder / ('trials_' + rx + '.csv'))
            assert len(trials) == 8000
            trial_count += len(trials)
            by_original_row = {}
            for row in trials:
                key = (row['receiver'], row['raw_row'])
                trial_by_packet.setdefault(key, []).append(float(row['delta']))
                by_original_row.setdefault(int(row['raw_row']), []).append(row)
                all_deltas.append(float(row['delta']))
                all_correlations.append(float(row['iq_correlation']))
            # Independently rederive each response: general lstsq instead of the analytic projection.
            for donor in fit['fit_receivers']:
                if donor == rx:
                    continue
                difference = stats[donor].mean(0) - stats[rx].mean(0)
                difference -= difference.mean()
                projected = b @ np.linalg.lstsq(b, difference, rcond=None)[0]
                scale = min(1., 3 / max(float(np.max(np.abs(projected))) * 20 / np.log(10), 1e-30))
                expected = projected * scale
                actual = responses[rx + '__to__' + donor]
                error = float(np.max(np.abs(actual - expected)))
                max_response_error = max(max_response_error, error)
                assert error < 1e-12
            # One prespecified first selected row per TX: covers all 100 TX/RX cells once.
            labels = np.load(out / 'raw_source' / rx / 'labels.npy')
            rows = np.load(out / 'raw_source' / rx / 'rows.npy')
            positions = [int(np.flatnonzero(labels == cls)[0]) for cls in range(10)]
            source_file = next(i for i in reg['source_files'] if i['receiver'] == rx)
            path = ROOT / 'data/external/shen_receiver_agnostic' / source_file['relative']
            stored_iq = np.load(out / 'raw_source' / rx / 'iq.npy', mmap_mode='r')
            with h5py.File(path, 'r') as hf:
                for pos in positions:
                    row_index = int(rows[pos])
                    raw = np.asarray(hf['data'][row_index], dtype='float64')
                    iq = (raw[:8192] + 1j * raw[8192:])[None]
                    np.testing.assert_array_equal(iq[0], stored_iq[pos])
                    assert int(hf['label'][...].reshape(-1)[row_index]) == int(labels[pos]) + 31
                    clean = independent_cis(iq)
                    for trial in by_original_row[row_index]:
                        h = np.exp(responses[rx + '__to__' + trial['donor']])
                        aug = np.fft.ifft(np.fft.fft(iq, axis=-1) * h, axis=-1)
                        c = independent_cis(aug)
                        delta = float(np.sqrt(np.mean(((c - clean) / (fit['scalar_population_std'] + 1e-12))**2)))
                        error = abs(delta - float(trial['delta']))
                        max_delta_error = max(max_delta_error, error)
                        assert error < 1e-10
                    raw_packet_checks += 1
        packets = read_csv(folder / 'packets.csv')
        assert len(packets) == 10000 and len(trial_by_packet) == 10000
        packet_count += len(packets)
        delta = []
        for packet in packets:
            values = trial_by_packet[packet['receiver'], packet['raw_row']]
            assert len(values) == 4
            got = float(np.median(values))
            assert abs(got - float(packet['delta'])) < 1e-12
            delta.append(got)
        aggregate = dict(median_packet_delta_min=float(np.median(delta)),
                         packet_fraction_delta_ge_005_min=float(np.mean(np.array(delta) >= .05)),
                         all_realizations_delta_q95_max=float(np.quantile(all_deltas, .95)),
                         all_realizations_iq_correlation_min=float(np.min(all_correlations)))
        for name, value in aggregate.items():
            gate = decision['gates'][name]
            assert abs(value - gate['value']) < 1e-12
            result = value >= gate['threshold'] if gate['relation'] == '>=' else value <= gate['threshold']
            assert result == gate['passed']
        checks.append(dict(fold=fold, recomputed_primary_gates=aggregate, go=decision['go']))
    overall = json.loads((out / 'decision.json').read_text())
    assert overall['V1_allowed'] == all(r['go'] for r in checks)
    assert packet_count == 20000 and trial_count == 80000 and raw_packet_checks == 100
    result = dict(status='passed', registration_and_code_seals_verified=True,
        packets_reaggregated=packet_count, perturbation_trials_reaggregated=trial_count,
        independent_raw_packets=raw_packet_checks, independent_raw_perturbations=raw_packet_checks * 4,
        max_response_absolute_error=max_response_error, max_delta_absolute_error=max_delta_error,
        source_only_access_verified=True, checks=checks)
    with (out / 'independent_verification.json').open('x') as fp:
        json.dump(result, fp, indent=2, allow_nan=False)
        fp.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
