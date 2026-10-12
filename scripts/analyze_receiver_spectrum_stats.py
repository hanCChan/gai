#!/usr/bin/env python3
"""Task 2: per-receiver / per-device spectrum statistics.

For every RX1/RX2 DeviceX file (24 classes, exclude raw Device9), deterministically
sample 256 windows of 8192 samples (same stride logic as eval), compute spectral
descriptors on RAW IQ (no power normalization, to expose receiver gain / tilt / CFO),
then aggregate per file. sample_rate=1MHz, lora_bandwidth=125kHz.

Outputs:
  outputs/cross_receiver_analysis/receiver_spectrum_stats.csv          (per file)
  outputs/cross_receiver_analysis/receiver_spectrum_summary_by_rx.csv  (per receiver)
  outputs/cross_receiver_analysis/receiver_spectrum_summary_by_device.csv (per device)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path("/data1/hcc/llm4RF")
SUBSET = ROOT / "data/raw/osu_lora/Diff_Receivers_Setup/Diff_Receivers_Setup_Indoor_SameTx"
RECEIVERS = ["RX1", "RX2"]
RX_ID = {"RX1": 1, "RX2": 2}
EXCLUDED_RAW_DEVICES = {9}
WINDOW_SIZE = 8192
SAMPLES_PER_FILE = 256
SAMPLE_RATE = 1_000_000.0
LORA_BW = 125_000.0
EPS = 1e-12
OUTDIR = ROOT / "outputs/cross_receiver_analysis"

METRIC_KEYS = [
    "iq_amp_mean", "iq_amp_std",
    "iq_power_mean", "iq_power_std",
    "fft_mag_mean", "fft_mag_std",
    "inband_energy_mean", "inband_energy_std",
    "oob_energy_mean", "oob_energy_std",
    "oob_inband_ratio",
    "spectral_centroid",
    "spectral_flatness",
    "peak_bin",
    "peak_offset",
]


def experiment_device(raw_device: int) -> int:
    return raw_device - sum(1 for ex in EXCLUDED_RAW_DEVICES if ex < raw_device)


def deterministic_offsets(length: int) -> list[int]:
    max_offset = max(0, length - WINDOW_SIZE)
    if max_offset == 0:
        return [0] * SAMPLES_PER_FILE
    stride = max(1, max_offset // max(1, SAMPLES_PER_FILE - 1))
    return [min(i * stride, max_offset) for i in range(SAMPLES_PER_FILE)]


def file_stats(path: Path) -> dict[str, float]:
    mm = np.memmap(path, dtype=np.complex64, mode="r")
    length = mm.shape[0]
    offsets = deterministic_offsets(length)

    freq = np.fft.fftshift(np.fft.fftfreq(WINDOW_SIZE, d=1.0 / SAMPLE_RATE))
    in_band = np.abs(freq) <= (LORA_BW / 2.0)
    oob_mask = ~in_band

    amp_means, power_means = [], []
    fft_mag_means = []
    inband_energies, oob_energies = [], []
    ratios, centroids, flatnesses = [], [], []
    peak_bins, peak_offsets = [], []

    for off in offsets:
        win = np.asarray(mm[off:off + WINDOW_SIZE])
        if win.shape[0] < WINDOW_SIZE:
            continue
        i = win.real.astype(np.float64)
        q = win.imag.astype(np.float64)
        amp = np.sqrt(i * i + q * q)
        power = i * i + q * q
        amp_means.append(amp.mean())
        power_means.append(power.mean())

        spec = np.fft.fftshift(np.fft.fft(win.astype(np.complex128)))
        maglin = np.abs(spec)
        fft_mag_means.append(maglin.mean())
        psd = maglin * maglin
        tot = psd.sum() + EPS
        inb = psd[in_band].sum()
        oob = psd[oob_mask].sum()
        inband_energies.append(inb)
        oob_energies.append(oob)
        ratios.append(oob / (inb + EPS))
        centroids.append(float((freq * psd).sum() / tot))
        flatnesses.append(float(np.exp(np.mean(np.log(psd + EPS))) / (psd.mean() + EPS)))
        pk = int(np.argmax(psd))
        peak_bins.append(pk)
        peak_offsets.append(float(freq[pk]))

    def m(a):
        return float(np.mean(a)) if a else float("nan")

    def s(a):
        return float(np.std(a)) if a else float("nan")

    return {
        "iq_amp_mean": m(amp_means), "iq_amp_std": s(amp_means),
        "iq_power_mean": m(power_means), "iq_power_std": s(power_means),
        "fft_mag_mean": m(fft_mag_means), "fft_mag_std": s(fft_mag_means),
        "inband_energy_mean": m(inband_energies), "inband_energy_std": s(inband_energies),
        "oob_energy_mean": m(oob_energies), "oob_energy_std": s(oob_energies),
        "oob_inband_ratio": m(ratios),
        "spectral_centroid": m(centroids),
        "spectral_flatness": m(flatnesses),
        "peak_bin": m(peak_bins),
        "peak_offset": m(peak_offsets),
        "n_windows": len(amp_means),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for rx in RECEIVERS:
        for raw_device in range(1, 26):
            if raw_device in EXCLUDED_RAW_DEVICES:
                continue
            path = SUBSET / rx / f"Device{raw_device}_IQ.dat"
            if not path.exists():
                print(f"skip missing {path}")
                continue
            dev = experiment_device(raw_device)
            st = file_stats(path)
            row = {
                "receiver": rx,
                "receiver_id": RX_ID[rx],
                "raw_device": raw_device,
                "exp_device": dev,
                "label": dev - 1,
            }
            row.update(st)
            rows.append(row)
            print(f"{rx} raw{raw_device:2d} dev{dev:2d}: oob/inband={st['oob_inband_ratio']:.4f} "
                  f"centroid={st['spectral_centroid']:.0f}Hz peak_off={st['peak_offset']:.0f}Hz "
                  f"flat={st['spectral_flatness']:.4e} amp={st['iq_amp_mean']:.4f}")

    base_fields = ["receiver", "receiver_id", "raw_device", "exp_device", "label", "n_windows"]
    fields = base_fields + METRIC_KEYS
    stats_path = OUTDIR / "receiver_spectrum_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {stats_path} rows={len(rows)}")

    # summary by receiver
    by_rx_path = OUTDIR / "receiver_spectrum_summary_by_rx.csv"
    with by_rx_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["receiver", "n_files"] + METRIC_KEYS)
        w.writeheader()
        for rx in RECEIVERS:
            sub = [r for r in rows if r["receiver"] == rx]
            if not sub:
                continue
            out = {"receiver": rx, "n_files": len(sub)}
            for k in METRIC_KEYS:
                out[k] = float(np.mean([r[k] for r in sub]))
            w.writerow(out)
    print(f"wrote {by_rx_path}")

    # summary by device (average across receivers) + RX1-RX2 deltas for key shape metrics
    by_dev_path = OUTDIR / "receiver_spectrum_summary_by_device.csv"
    delta_keys = ["oob_inband_ratio", "spectral_centroid", "spectral_flatness", "peak_offset", "iq_amp_mean"]
    with by_dev_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["exp_device", "label"] + METRIC_KEYS + [f"rx2_minus_rx1_{k}" for k in delta_keys],
        )
        w.writeheader()
        devices = sorted({r["exp_device"] for r in rows})
        for dev in devices:
            sub = [r for r in rows if r["exp_device"] == dev]
            out = {"exp_device": dev, "label": dev - 1}
            for k in METRIC_KEYS:
                out[k] = float(np.mean([r[k] for r in sub]))
            rx1 = next((r for r in sub if r["receiver"] == "RX1"), None)
            rx2 = next((r for r in sub if r["receiver"] == "RX2"), None)
            for k in delta_keys:
                if rx1 is not None and rx2 is not None:
                    out[f"rx2_minus_rx1_{k}"] = float(rx2[k] - rx1[k])
                else:
                    out[f"rx2_minus_rx1_{k}"] = float("nan")
            w.writerow(out)
    print(f"wrote {by_dev_path}")


if __name__ == "__main__":
    main()
