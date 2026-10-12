#!/usr/bin/env python3
"""Run the sealed source-only V0, export all trials and enforce Stop/Go."""
import argparse
import csv
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from virp.spectral import (bounded_log_response, cis_details, correlation,
                           log_spectrum, packet_metrics, perturb, project_style)
from shen_stage1.frequency import periodic_phase_offset


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summary(x):
    a = np.asarray(x, dtype=float)
    if not np.isfinite(a).all():
        raise ValueError('Nonfinite aggregate')
    return dict(min=float(a.min()), q05=float(np.quantile(a, .05)),
                median=float(np.median(a)), q95=float(np.quantile(a, .95)), max=float(a.max()))


def prepare_source(item, reg, out, access):
    rx = item['receiver']
    assert rx in reg['source_receivers'] and item['relative'] == rx + '_train.h5'
    base = (ROOT / 'data/external/shen_receiver_agnostic').resolve()
    path = (base / item['relative']).resolve()
    assert path == base / item['relative'] and path.is_relative_to(base)
    access.append({'stage': 'V0', 'receiver': rx, 'path': str(path), 'role': 'source_train_raw', 'mode': 'read_only'})
    actual_sha = sha(path)
    if actual_sha != item['expected_sha256']:
        raise ValueError('Raw SHA mismatch: ' + rx)
    folder = out / 'raw_source' / rx
    folder.mkdir(parents=True, exist_ok=False)
    with h5py.File(path, 'r') as f:
        assert all(isinstance(f.get(k, getlink=True), h5py.HardLink) for k in ('data', 'label'))
        labels = np.asarray(f['label'][:]).reshape(-1)
        assert set(labels) == set(range(31, 41)) and f['data'].shape == (len(labels), 16384)
        rng = np.random.default_rng(reg['sample_selection_seed'])
        rows = np.sort(np.concatenate([rng.choice(np.flatnonzero(labels == y), 200, replace=False) for y in range(31, 41)]))
        y = labels[rows].astype('int64') - 31
        # Check the historical selection, but never reuse cached IQ or fitted statistics.
        cache = ROOT / 'outputs/shen_p2a8/g1_cache_v1' / item['key']
        np.testing.assert_array_equal(rows, np.load(cache / 'rows.npy'))
        np.testing.assert_array_equal(y, np.load(cache / 'labels.npy'))
        x = np.lib.format.open_memmap(folder / 'iq.npy', mode='w+', dtype='complex128', shape=(len(rows), 8192))
        profiles = np.zeros((10, 8192))
        halves = np.zeros((2, 10, 8192))
        means = np.zeros((10, 52, 126))
        counts = np.zeros(10, dtype='int64')
        totals = dict(count=0, sum=0., sum_squares=0.)
        for start in range(0, len(rows), 64):
            ids = rows[start:start + 64]
            stored = np.asarray(f['data'][ids], dtype='float64')
            iq = stored[:, :8192] + 1j * stored[:, 8192:]
            assert np.isfinite(iq).all()
            x[start:start + len(iq)] = iq
            c, _ = cis_details(iq)
            logs = log_spectrum(iq)
            for local, cls in enumerate(y[start:start + len(iq)]):
                half = int(counts[cls] >= 100)
                profiles[cls] += logs[local] / 200
                halves[half, cls] += logs[local] / 100
                means[cls] += c[local] / 200
                counts[cls] += 1
            totals['count'] += c.size
            totals['sum'] += float(c.sum())
            totals['sum_squares'] += float(np.square(c).sum())
        assert (counts == 200).all()
        x.flush()
    np.save(folder / 'labels.npy', y)
    np.save(folder / 'rows.npy', rows)
    np.savez(folder / 'statistics.npz', profiles=profiles, halves=halves, cis_means=means)
    write_json(folder / 'manifest.json', dict(source=item, actual_sha256=actual_sha, stats=totals,
               packets=len(rows), dtype='complex128 reconstructed from raw stored float64', counts=counts.tolist()))
    print('raw source prepared: ' + rx, flush=True)


def stats_for(out, rx):
    folder = out / 'raw_source' / rx
    with np.load(folder / 'statistics.npz') as s:
        result = {k: s[k].copy() for k in s.files}
    result['totals'] = json.loads((folder / 'manifest.json').read_text())['stats']
    return result


def check(value, relation, threshold):
    return {'value': float(value), 'relation': relation, 'threshold': threshold,
            'passed': bool(value >= threshold if relation == '>=' else value <= threshold)}


def fold_audit(fold, fitting, heldout, reg, out):
    folder = out / ('fold' + str(fold))
    folder.mkdir(exist_ok=False)
    # No source-held-out statistics are consulted until response construction is sealed.
    fitting_stats = {r: stats_for(out, r) for r in fitting}
    totals = [s['totals'] for s in fitting_stats.values()]
    count = sum(t['count'] for t in totals)
    mean = sum(t['sum'] for t in totals) / count
    sigma = np.sqrt(sum(t['sum_squares'] for t in totals) / count - mean**2)
    assert sigma > 0 and np.isfinite(sigma)
    responses = {}
    pair_rows = []
    dominance = []
    for rx in fitting:
        for donor in fitting:
            if donor == rx:
                continue
            a, b = fitting_stats[rx], fitting_stats[donor]
            difference = b['profiles'].mean(0) - a['profiles'].mean(0)
            projected = project_style(difference)
            log_h, shrink = bounded_log_response(difference)
            responses[rx, donor] = log_h
            half_profiles = [project_style(b['halves'][h].mean(0) - a['halves'][h].mean(0)) for h in (0, 1)]
            norms = [np.linalg.norm(p) for p in half_profiles]
            cosine = np.dot(*half_profiles) / max(norms[0] * norms[1], 1e-30)
            norm_ratio = norms[0] / max(norms[1], 1e-30)
            centered = difference - difference.mean()
            explained = np.sum(projected**2) / max(np.sum(centered**2), 1e-30)
            pair_rows.append(dict(source_rx=rx, donor_rx=donor, shrink=shrink,
                response_max_abs_db=float(np.max(np.abs(log_h)) * 20 / np.log(10)),
                smooth_explained_fraction=float(explained), half_cosine=float(cosine), half_norm_ratio=float(norm_ratio),
                half_stable=bool(cosine >= .8 and .5 <= norm_ratio <= 2)))
            for cls in range(10):
                others = np.arange(10) != cls
                loo = project_style(b['profiles'][others].mean(0) - a['profiles'][others].mean(0))
                dominance.append(dict(source_rx=rx, donor_rx=donor, TX=cls + 31,
                    relative_change=float(np.linalg.norm(loo - projected) / max(np.linalg.norm(projected), 1e-30))))
    csv_write(folder / 'source_style_pairs.csv', pair_rows)
    csv_write(folder / 'leave_one_TX_out.csv', dominance)
    np.savez(folder / 'responses.frozen.npz', **{r + '__to__' + d: v for (r, d), v in responses.items()})
    write_json(folder / 'fit.frozen.json', dict(fit_receivers=fitting, heldout_receivers=heldout,
        scalar_population_std=float(sigma), scalar_mean=float(mean), sigma_fit_count=count,
        response_file_sha256=sha(folder / 'responses.frozen.npz'), frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    packets = []
    controls = []
    synthetic_means = {}
    for rx in fitting:
        raw_folder = out / 'raw_source' / rx
        x = np.load(raw_folder / 'iq.npy', mmap_mode='r')
        y = np.load(raw_folder / 'labels.npy')
        rows = np.load(raw_folder / 'rows.npy')
        donors = [r for r in fitting if r != rx]
        sums = {d: np.zeros((10, 52, 126)) for d in donors}
        with (folder / ('trials_' + rx + '.csv')).open('w', newline='') as f:
            writer = None
            for start in range(0, len(x), 32):
                iq = np.asarray(x[start:start + 32])
                clean, den = cis_details(iq)
                original_coherence = periodic_phase_offset(iq, 1024, 1e6)['coherence']
                assert np.isfinite(original_coherence).all()
                metrics_by_donor = []
                for donor in donors:
                    changed = perturb(iq, responses[rx, donor])
                    assert np.isfinite(changed).all()
                    c, _ = cis_details(changed)
                    metrics = packet_metrics(clean, c, den, sigma)
                    metrics['iq_correlation'] = correlation(iq, changed)
                    metrics['coherence_change'] = np.abs(periodic_phase_offset(changed, 1024, 1e6)['coherence'] - original_coherence)
                    assert all(np.isfinite(v).all() for v in metrics.values())
                    metrics_by_donor.append(metrics)
                    for local, cls in enumerate(y[start:start + len(iq)]):
                        sums[donor][cls] += c[local] / 200
                        row = dict(fold=fold, receiver=rx, TX=int(cls + 31), raw_row=int(rows[start + local]), donor=donor,
                                   **{k: float(v[local]) for k, v in metrics.items()})
                        if writer is None:
                            writer = csv.DictWriter(f, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                for local, cls in enumerate(y[start:start + len(iq)]):
                    trial_values = lambda k: [m[k][local] for m in metrics_by_donor]
                    packets.append(dict(fold=fold, receiver=rx, TX=int(cls + 31), raw_row=int(rows[start + local]),
                        delta=float(np.median(trial_values('delta'))), delta_core=float(np.median(trial_values('delta_core'))),
                        delta_max=float(np.max(trial_values('delta'))), iq_correlation_min=float(np.min(trial_values('iq_correlation')))))
                for name, g in (('zero_fft_ifft', 0.), ('constant_plus3dB', 3 * np.log(10) / 20)):
                    control_iq = perturb(iq, np.full(8192, g))
                    control_c, _ = cis_details(control_iq)
                    delta = np.sqrt(np.square((control_c - clean) / (sigma + 1e-12)).mean(axis=(1, 2)))
                    for local in range(len(iq)):
                        controls.append(dict(receiver=rx, raw_row=int(rows[start + local]), control=name, delta=float(delta[local])))
        synthetic_means.update({(rx, d): a for d, a in sums.items()})
        print('fold ' + str(fold) + ' perturbation audit complete: ' + rx, flush=True)
    csv_write(folder / 'packets.csv', packets)
    csv_write(folder / 'negative_controls.csv', controls)
    all_trials = []
    for rx in fitting:
        with (folder / ('trials_' + rx + '.csv')).open() as f:
            all_trials.extend([{k: float(v) for k, v in r.items() if k not in ('receiver', 'donor')} for r in csv.DictReader(f)])
    trials = {k: np.array([r[k] for r in all_trials]) for k in all_trials[0]}
    # Now and only now read profiles of source-held-out RX. Never fit or adjust responses with them.
    heldout_stats = {r: stats_for(out, r) for r in heldout}
    coverage = []
    for rx in heldout:
        target = heldout_stats[rx]['profiles'].mean(0)
        target = target - target.mean()
        clean_distances, augmented_distances = [], []
        for source in fitting:
            clean_profile = fitting_stats[source]['profiles'].mean(0)
            clean_profile -= clean_profile.mean()
            clean_distances.append(np.linalg.norm(target - clean_profile))
            for donor in fitting:
                if donor != source:
                    augmented_distances.append(np.linalg.norm(target - clean_profile - responses[source, donor]))
        original = min(clean_distances)
        changed = min(augmented_distances)
        gain = 1 - changed / max(original, 1e-30)
        coverage.append(dict(heldout_rx=rx, best_clean_distance=float(original), best_augmented_distance=float(changed),
                             improvement_fraction=float(gain), covered=bool(gain >= .10)))
    csv_write(folder / 'heldout_spectrum_coverage.csv', coverage)
    natural = []
    # Report class-conditional means from unpaired samples; do not pretend exact signal matches exist.
    both_stats = {**fitting_stats, **heldout_stats}
    names = fitting + heldout
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for cls in range(10):
                delta = np.sqrt(np.mean(((both_stats[a]['cis_means'][cls] - both_stats[b]['cis_means'][cls]) / sigma) ** 2))
                natural.append(dict(rx1=a, rx2=b, TX=cls + 31, delta_centroid=float(delta)))
    synthetic = []
    for (source, donor), means in synthetic_means.items():
        for cls in range(10):
            delta = np.sqrt(np.mean(((means[cls] - fitting_stats[source]['cis_means'][cls]) / sigma) ** 2))
            synthetic.append(dict(source_rx=source, donor_rx=donor, TX=cls + 31, delta_centroid=float(delta)))
    csv_write(folder / 'natural_CIS_centroid_distances.csv', natural)
    csv_write(folder / 'synthetic_CIS_centroid_distances.csv', synthetic)
    ratio = np.median([r['delta_centroid'] for r in synthetic]) / max(np.median([r['delta_centroid'] for r in natural]), 1e-30)
    limits = reg['v0']['hard_gates_each_fold']
    delta = np.array([r['delta'] for r in packets])
    core = np.array([r['delta_core'] for r in packets])
    observations = {
        'median_packet_delta_min': (np.median(delta), '>='),
        'packet_fraction_delta_ge_005_min': (np.mean(delta >= .05), '>='),
        'all_realizations_delta_q95_max': (np.quantile(trials['delta'], .95), '<='),
        'nonfinite_count_max': (0, '<='),
        'max_abs_response_db': (max(r['response_max_abs_db'] for r in pair_rows), '<='),
        'all_realizations_iq_correlation_min': (trials['iq_correlation'].min(), '>='),
        'negative_control_delta_max': (max(r['delta'] for r in controls), '<='),
        'median_packet_core_delta_min': (np.median(core), '>='),
        'low_energy_residual_share_q95_max': (np.quantile(trials['low_energy_change_share'], .95), '<='),
        'boundary_residual_share_q95_max': (np.quantile(trials['boundary_change_share'], .95), '<='),
        'extreme_cis_fraction_increase_max': (max(0., float(np.mean(trials['extreme_aug_fraction'] - trials['extreme_clean_fraction']))), '<='),
        'coherence_change_absolute_q95_max': (np.quantile(trials['coherence_change'], .95), '<='),
        'profile_half_split_stable_pair_fraction_min': (np.mean([r['half_stable'] for r in pair_rows]), '>='),
        'profile_loo_tx_relative_change_q95_max': (np.quantile([r['relative_change'] for r in dominance], .95), '<='),
        'profile_smooth_explained_fraction_median_min': (np.median([r['smooth_explained_fraction'] for r in pair_rows]), '>='),
        'heldout_spectral_improvement_fraction_min': (np.mean([r['covered'] for r in coverage]), '>='),
        'natural_cis_scale_ratio_min': (ratio, '>='), 'natural_cis_scale_ratio_max': (ratio, '<='),
    }
    # Floating arithmetic at an exact 3dB boundary is allowed 1e-12 roundoff, not a larger gain.
    gates = {k: check(v, rel, limits[k]) for k, (v, rel) in observations.items()}
    if observations['max_abs_response_db'][0] <= limits['max_abs_response_db'] + 1e-12:
        gates['max_abs_response_db']['passed'] = True
    per_rx_tx = []
    for rx in fitting:
        for cls in range(31, 41):
            subset = [r for r in packets if r['receiver'] == rx and r['TX'] == cls]
            ds = np.array([r['delta'] for r in subset])
            per_rx_tx.append(dict(receiver=rx, TX=cls, packets=len(subset), median_delta=float(np.median(ds)),
                coverage_delta_ge_005=float(np.mean(ds >= .05)), median_core_delta=float(np.median([r['delta_core'] for r in subset])),
                min_iq_correlation=float(min(r['iq_correlation_min'] for r in subset))))
    csv_write(folder / 'per_RX_TX.csv', per_rx_tx)
    result = dict(fold=fold, fitting_receivers=fitting, heldout_receivers=heldout,
        source_packets=len(packets), perturbation_realizations=len(all_trials), sigma=float(sigma),
        delta=summary(delta), delta_core=summary(core), correlation=summary(trials['iq_correlation']),
        natural_centroid_delta=summary([r['delta_centroid'] for r in natural]),
        synthetic_centroid_delta=summary([r['delta_centroid'] for r in synthetic]),
        negative_control_delta=summary([r['delta'] for r in controls]), gates=gates,
        failed_gates=[k for k, v in gates.items() if not v['passed']],
        go=all(v['passed'] for v in gates.values()))
    write_json(folder / 'decision.json', result)
    print(json.dumps(dict(fold=fold, delta=result['delta'], failed_gates=result['failed_gates']), ensure_ascii=False), flush=True)
    return result


def report(out, reg, decisions, seal):
    failed = [{'fold': d['fold'], 'gate': g} for d in decisions for g in d['failed_gates']]
    go = not failed
    decision = dict(experiment='VIRP-v1', stage='V0', V0_passed=go, V1_allowed=go,
        deep_training_runs=0, V1_started=False, V2_started=False, V3_started=False,
        developer_RX_accessed=False, historical_locked_RX_accessed=False, OSU_accessed=False,
        registration_sha256=seal['registration_sha256'], failed_gates=failed,
        status='GO_TO_V1' if go else 'STOP_VIRP_V1',
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(out / 'decision.json', decision)
    lines = ['# VIRP-v1 · V0 预登记审计', '',
        '**决策：' + ('V0 通过，允许进入 V1。' if go else 'V0 未通过，停止当前第一版；不进入 B0–B4。') + '**', '',
        '这是训练前的信号与表征可行性审计，不是分类性能或论文创新性验证。', '',
        '注册 SHA256：`' + seal['registration_sha256'] + '`。所有阈值、六维扰动族及审计程序均在新信号结果之前冻结。', '',
        '两折各仅用五台源 RX 估计响应及 CIS 标准差。覆盖核验使用另外五台源 RX；四开发 RX、六历史锁定 RX 和 OSU 均未访问。', '',
        '重新读取并验证十个源训练 HDF5 的 SHA256；沿用 G1 每 TX/RX 200 包的同一索引，原始 float64 重建 complex128。每源包评估其拟合折内全部四个 donor 响应，共 20,000 包、80,000 扰动实现；另有 40,000 个零扰动/全频增益控制。', '',
        '标准化 δ 对 `log10` 或 dB 表示相同。每包 δ 是四个固定响应 δ 的中位数；上界、相关性及伪影指标额外检查全部实现。', '',
        '|折|包数|median δ|δ≥0.05 包比例|全部实现 δ 的 q95|最小 IQ 相关性|核心 median δ|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for d in decisions:
        g = d['gates']
        lines.append(f"|{d['fold']}|{d['source_packets']}|{d['delta']['median']:.6f}|{g['packet_fraction_delta_ge_005_min']['value']:.2%}|{g['all_realizations_delta_q95_max']['value']:.6f}|{d['correlation']['min']:.6f}|{d['delta_core']['median']:.6f}|")
    lines += ['', '## 全部硬性关口', '', '|折|检查|观测|预登记要求|通过|', '|---|---|---:|---|---|']
    for d in decisions:
        for name, g in d['gates'].items():
            lines.append(f"|{d['fold']}|{name}|{g['value']:.8g}|{g['relation']} {g['threshold']}|{'是' if g['passed'] else '否'}|")
    lines += ['', '## 证据边界', '',
        '纯幅频、零相位全包 FFT 修改仅称 frequency-response-inspired spectral augmentation；不声称是实测、因果可实现的接收机硬件响应。', '',
        '同 TX 的自然 CIS 距离比较使用不同包的类别平均图，包含采集链路/信道/内容及采样误差，不能解释成同一发射波形的硬件因果效应。', '',
        '半包谱形稳定性、leave-one-TX-out、源留出谱形覆盖、低能量分母和边界残余均已单列。每 RX 只有一个源训练文件，无法从这些数据建立同一 RX 的跨独立采集文件稳定性；不将两折或 80,000 扰动当独立环境做显著性检验。', '',
        '未训练分类器，因而没有设备 Recall 或已证明的身份保护；per_RX_TX.csv 仅报告各设备扰动影响。未进行 OSU 新窗口审计或改写 G1 结果。', '',
        '## 复核和复现', '',
        '`configs/virp/v1_registration.json`、`.seal.json` 为冻结协议；`raw_source/*/manifest.json` 为原始数据出处；每折 `responses.frozen.npz`、`fit.frozen.json` 记录源拟合；`trials_*.csv` 保存逐包逐扰动；`packets.csv` 保存主指标；`negative_controls.csv` 保存控制结果。', '',
        '复现必须使用新的空输出目录：`/home/keyan/venvs/shen-stage1/bin/python scripts/audit_virp_v0.py --output outputs/virp/v1_v0_reproduction`。不修改本注册、阈值或结果目录。', '',
        'V0 失败后，不扩大幅频范围，不新增 CFO/IQ imbalance/时变扰动，不改 STFT，不启动 B0–B4。任何下一候选方法需独立注册并引用本次负结果。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registration', default='configs/virp/v1_registration.json')
    parser.add_argument('--output')
    args = parser.parse_args()
    config = ROOT / args.registration
    reg = json.loads(config.read_text())
    seal = json.loads(config.with_suffix('.seal.json').read_text())
    assert sha(config) == seal['registration_sha256'], 'Registration changed after freeze'
    for relative, expected in reg['immutable_code_sha256'].items():
        assert sha(ROOT / relative) == expected, 'Sealed code changed: ' + relative
    out = ROOT / (args.output or reg['v0_output'])
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    access = []
    write_json(out / 'registration.json', reg)
    write_json(out / 'seal.json', seal)
    write_json(out / 'status.json', dict(status='running', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    try:
        for item in reg['source_files']:
            prepare_source(item, reg, out, access)
            write_json(out / 'signal_access_log.json', access)
        decisions = []
        for fold, heldout in enumerate(reg['source_receiver_folds']):
            fitting = [r for r in reg['source_receivers'] if r not in heldout]
            decisions.append(fold_audit(fold, fitting, heldout, reg, out))
        decision = report(out, reg, decisions, seal)
        write_json(out / 'status.json', dict(status='completed', decision=decision['status'], seconds=time.perf_counter() - started))
        print(json.dumps(decision, ensure_ascii=False), flush=True)
    except Exception as e:
        write_json(out / 'signal_access_log.json', access)
        write_json(out / 'status.json', dict(status='failed', error=str(e), seconds=time.perf_counter() - started))
        write_json(out / 'decision.json', dict(status='STOP_AUDIT_ERROR', V0_passed=False, V1_allowed=False, deep_training_runs=0, error=str(e)))
        raise


if __name__ == '__main__':
    main()
