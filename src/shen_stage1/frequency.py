"""Read-only phase-correlation diagnostics; never compensate an input signal.

Equivalent sign convention to Shen 2101.01668 Eq. (13), extended to sum
correlations across all neighboring assumed chirps. Estimates are modulo
Fs/lag, not a coarse/unwrapped CFO estimator. Correct periodicity and actual
sampling rate must be established before interpreting outputs physically.
"""
from __future__ import annotations

import numpy as np


def periodic_phase_offset(iq, lag: int, sample_rate_hz: float):
    x = np.asarray(iq, dtype=np.complex128)
    if x.ndim != 2 or lag < 1 or x.shape[1] < 2 * lag or x.shape[1] % lag:
        raise ValueError("Expected [packets, K*lag], with K >= 2")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Sampling rate must be finite and positive")
    if not np.isfinite(x).all():
        raise ValueError("Signal must be finite")
    blocks = x.reshape(len(x), -1, lag)
    previous, following = blocks[:, :-1], blocks[:, 1:]
    correlations = (previous.conj() * following).sum(axis=-1)
    previous_energy = (np.abs(previous) ** 2).sum(axis=-1)
    following_energy = (np.abs(following) ** 2).sum(axis=-1)
    total = correlations.sum(axis=-1)
    energy_product = previous_energy.sum(axis=-1) * following_energy.sum(axis=-1)
    valid = (energy_product > 0) & (np.abs(total) > 0)
    phase = np.angle(total)
    offsets = np.where(valid, phase * sample_rate_hz / (2 * np.pi * lag), np.nan)
    first_valid = (previous_energy[:, 0] * following_energy[:, 0] > 0) & (np.abs(correlations[:, 0]) > 0)
    first = np.where(first_valid, np.angle(correlations[:, 0]) * sample_rate_hz / (2 * np.pi * lag), np.nan)
    coherence = np.full(len(x), np.nan)
    coherence[valid] = np.clip(np.abs(total[valid]) / np.sqrt(energy_product[valid]), 0, 1)
    # Circular, rather than linear, dispersion avoids a false jump at +/-pi.
    phase_resultant = np.abs(np.exp(1j * np.angle(correlations)).mean(axis=-1))
    circular_std_hz = np.sqrt(-2 * np.log(np.clip(phase_resultant, 1e-15, 1))) * sample_rate_hz / (2 * np.pi * lag)
    circular_std_hz[~valid] = np.nan
    return {"all_pairs_offset_hz_under_assumed_fs": offsets,
            "first_pair_offset_hz_under_assumed_fs": first,
            "offset_cycles_per_sample_under_assumed_lag": offsets / sample_rate_hz,
            "coherence": coherence, "pair_phase_circular_std_hz_under_assumed_fs": circular_std_hz,
            "valid": valid}
