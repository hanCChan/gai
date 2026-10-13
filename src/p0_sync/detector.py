"""Conditional chirp-phase diagnostic, NOT a packet-start synchronizer.

Use normalized cycles/sample throughout: local Fs/BW/SF remain unverified.
Estimate a sawtooth phase-frequency pattern modulo its putative period.
An intercept accommodates CFO; positive/negative templates distinguish chirps.
Repeated symbols cannot establish which chirp starts a complete packet.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d


def distance(a, b, period):
    return np.abs((np.asarray(a) - b + period / 2) % period - period / 2)


def coherence(x, lag):
    a, b = x[:-lag], x[lag:]
    den = np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
    return float(abs(np.vdot(a, b)) / den) if den > 0 else 0.


def fit_profile(profile, smooth=8):
    n = len(profile)
    p = uniform_filter1d(np.asarray(profile, float), size=smooth, mode='wrap')
    p -= p.mean()
    ramp = (np.arange(n) + .5) / n - .5
    dot = np.fft.ifft(np.fft.fft(p) * np.conj(np.fft.fft(ramp))).real
    sign = 1 if dot.max() >= -dot.min() else -1
    dot *= sign
    idx = int(dot.argmax())
    norm = np.dot(ramp, ramp)
    variance = np.dot(p, p)
    r2 = np.square(np.maximum(dot, 0)) / max(norm * variance, 1e-30)
    r2 = np.clip(r2, 0, 1)
    width = float(abs(dot[idx] / norm))
    return dict(phase=idx, direction=sign, r2=float(r2[idx]), bandwidth_cycles=width), r2, p


def analyze(x, cfg):
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 1 or len(x) != 8192 or not np.isfinite(x).all():
        raise ValueError('Expected finite 8192 complex samples')
    n = cfg['nominal_period_samples']
    lags = cfg['candidate_periods_samples']
    cs = {lag: coherence(x, lag) for lag in lags}
    # Adjacent measured samples only; never append x[0] at the right boundary.
    f = np.angle(x[1:] * x[:-1].conj()) / (2 * np.pi)
    # Last within-record symbol lacks exactly one adjacent-sample observation.
    # Interpolate that frequency for shape diagnostics, NEVER wrap raw IQ.
    # This avoids omitting a possible downchirp in the final symbol.
    last = 2*f[-1] - f[-2]
    blocks = np.r_[f, last].reshape(-1, n)
    profile = np.median(blocks, axis=0)
    fitted, curve, folded = fit_profile(profile, cfg['profile_smoothing_samples'])
    candidates = np.flatnonzero(curve >= fitted['r2'] - cfg['candidate_r2_drop'])
    radius = float(distance(candidates, fitted['phase'], n).max())
    fits = [fit_profile(z, cfg['profile_smoothing_samples'])[0] for z in blocks]
    spread = float(np.quantile(distance([z['phase'] for z in fits], fitted['phase'], n), .95))
    good = (cs[n] >= cfg['coherence_min'] and fitted['r2'] >= cfg['chirp_r2_min']
            and cfg['bandwidth_cycles_range'][0] <= fitted['bandwidth_cycles'] <= cfg['bandwidth_cycles_range'][1]
            and radius <= cfg['phase_radius_max_samples']
            and spread <= cfg['inter_chirp_q95_max_samples'] and fitted['direction'] == 1)
    power = np.abs(x)**2
    central = float(np.median(power[n:-n]))
    head_ratio = float(np.mean(power[:128]) / max(central, 1e-30))
    tail_ratio = float(np.mean(power[-128:]) / max(central, 1e-30))
    pair = np.vdot(x[:-n], x[n:])
    residual = float(np.angle(pair) / (2*np.pi*n))
    out = dict(nominal_coherence=cs[n], chirp_phase_modulo_samples=fitted['phase'],
               chirp_direction=fitted['direction'], chirp_r2=fitted['r2'],
               bandwidth_cycles_per_sample=fitted['bandwidth_cycles'],
               candidate_count=int(len(candidates)), candidate_radius_samples=radius,
               inter_chirp_phase_q95_samples=spread, reliable_symbol_phase=bool(good),
               prefix_energy_ratio=head_ratio, suffix_energy_ratio=tail_ratio,
               left_quiet_candidate=bool(head_ratio < cfg['quiet_prefix_ratio_max']),
               up_chirp_blocks=sum(z['direction']==1 and z['r2']>=cfg['chirp_r2_min'] for z in fits),
               down_chirp_blocks=sum(z['direction']==-1 and z['r2']>=cfg['chirp_r2_min'] for z in fits),
               rms=float(np.sqrt(np.mean(power))),
               coarse_frequency_center_cycles_per_sample=float(np.mean(profile)),
               residual_frequency_modulo_cycles_per_sample=residual,
               packet_start_identified=False)
    out.update({f'coherence_lag_{lag}': v for lag,v in cs.items()})
    return out, folded, curve, fits


def synthetic_chirps(shift, cfo, snr, rng, direction=1):
    # A long analytic periodic waveform is cropped; no np.roll of stored IQ.
    t = np.arange(-2048, 10240, dtype=float)
    u = (t-shift) % 1024
    x = np.exp(2j*np.pi*(direction*.125*(u*u/(2*1024)-u/2) + cfo*t))
    y = x[2048:10240]
    noise = (rng.normal(size=8192) + 1j*rng.normal(size=8192)) * np.sqrt(10**(-snr/10)/2)
    return y+noise
