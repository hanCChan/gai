"""Model-free physical descriptors of WiSig ManyRx packets (non-equalized, 25 Msps, 2462 MHz).

At 25 Msps the 802.11 L-STF repeats every 20 samples (0.8 us) and its 12 occupied tones sit at
+-1.25 ... +-7.5 MHz, i.e. exactly bins +-1..+-6 of a 20-point FFT. Per packet we read:
  cfo_hz      observed relative carrier frequency offset (TX LO - RX LO, plus drift), lag-20
  snr_db      periodicity-based noise proxy |R| / (P - |R|); not a calibrated SNR
  tone_shape  12 STF tone log-powers, mean-removed: relative response shape of the received
              preamble (TX, RX and propagation combined; not a channel estimate). Invariant to
              circular start offset and packet gain; only approximately invariant to CFO, since a
              non-integer frequency offset changes leakage in the finite FFT window. The size of
              that leakage is measured on real packets by measured_change(), not assumed.
  periodicity |R|/P in [0,1]; low values flag windows that are not inside the L-STF
The packet start is only energy-detected upstream (WiSig paper Sec. IV), so the window skips the
first period. These are physical descriptors, not ground-truth channel measurements.
"""
import numpy as np

FS = 25e6
PERIOD = 20
TONE_BINS = np.array([-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])
TONE_HZ = TONE_BINS * FS / PERIOD


def to_complex(x):
    """(n,1,256,2) or (n,256,2) float -> (n,256) complex."""
    x = np.asarray(x)
    x = x[:, 0] if x.ndim == 4 else x
    return x[..., 0].astype(np.float64) + 1j * x[..., 1].astype(np.float64)


def to_cache(z):
    """(n,256) complex -> (n,1,256,2) float32 with the cache's per-packet unit power."""
    z = z / np.sqrt(np.mean(np.abs(z) ** 2, axis=1, keepdims=True))
    return np.stack((z.real, z.imag), axis=-1)[:, None].astype(np.float32)


def stf_descriptors(z, start=PERIOD, periods=8):
    seg = z[:, start:start + periods * PERIOD]
    r = np.mean(seg[:, PERIOD:] * np.conj(seg[:, :-PERIOD]), axis=1)
    p = np.mean(np.abs(seg) ** 2, axis=1)
    cfo = np.angle(r) * FS / (2 * np.pi * PERIOD)
    snr = np.abs(r) / np.maximum(p - np.abs(r), 1e-12 * p)
    spec = np.mean(np.abs(np.fft.fft(seg.reshape(len(z), periods, PERIOD), axis=2)) ** 2, axis=1)
    tones = np.log(spec[:, TONE_BINS % PERIOD] + 1e-30)
    # proxy is clipped: with only 8 periods, |R| ~ P at high SNR makes the estimate unbounded
    return dict(cfo_hz=cfo, snr_db=np.clip(10 * np.log10(snr), -10.0, 60.0), periodicity=np.abs(r) / p,
                tone_shape=tones - tones.mean(1, keepdims=True))


def shift_cfo(z, df):
    """Multiply by exp(j 2 pi df n / FS); df scalar or per-packet array."""
    n = np.arange(z.shape[1])
    return z * np.exp(2j * np.pi * np.asarray(df, dtype=float).reshape(-1, 1) * n / FS)


def reshape_tones(z, delta_log_power):
    """Zero-phase frequency-domain gain whose log-power equals `delta_log_power` [n,12] at the
    STF tones (linear interpolation between tones, edge values held outside them)."""
    n, L = z.shape
    f = np.fft.fftfreq(L, 1 / FS)
    order = np.argsort(TONE_HZ)
    gains = np.stack([np.interp(f, TONE_HZ[order], d[order]) for d in np.asarray(delta_log_power)])
    return np.fft.ifft(np.fft.fft(z, axis=1) * np.exp(gains / 2), axis=1)


def add_noise_to_snr(z, snr_now_db, snr_target_db, rng):
    """Add white noise so the periodicity SNR proxy drops from snr_now to snr_target (only lowers)."""
    p = np.mean(np.abs(z) ** 2, axis=1)
    s_now = 10 ** (np.asarray(snr_now_db) / 10)
    s_tgt = 10 ** (np.minimum(snr_target_db, snr_now_db) / 10)
    sig = p * s_now / (1 + s_now)
    extra = np.where(np.asarray(snr_target_db) < np.asarray(snr_now_db), np.maximum(sig / s_tgt - (p - sig), 0), 0.0)
    w = rng.normal(size=z.shape) + 1j * rng.normal(size=z.shape)
    return z + np.sqrt(extra / 2)[:, None] * w


def grid_distance(a, b):
    """ORBIT node names 'x-y' (1 m grid) -> Euclidean distance in metres."""
    (xa, ya), (xb, yb) = (tuple(map(int, s.split('-'))) for s in (a, b))
    return float(np.hypot(xa - xb, ya - yb))


def measured_change(z0, z1):
    """Re-extract descriptors after an intervention: what actually changed, packet by packet.

    Returns arrays: d_cfo_hz, d_tone (n,12), d_snr_db, periodicity_after, input_rel_change
    (squared difference of unit-power packets over unit power) and a finite mask.
    """
    d0, d1 = stf_descriptors(z0), stf_descriptors(z1)
    u0 = z0 / np.sqrt(np.mean(np.abs(z0) ** 2, axis=1, keepdims=True))
    u1 = z1 / np.sqrt(np.mean(np.abs(z1) ** 2, axis=1, keepdims=True))
    finite = np.isfinite(z1).all(1) & np.isfinite(d1['cfo_hz']) & np.isfinite(d1['tone_shape']).all(1)
    return dict(d_cfo_hz=d1['cfo_hz'] - d0['cfo_hz'], d_tone=d1['tone_shape'] - d0['tone_shape'],
                d_snr_db=d1['snr_db'] - d0['snr_db'], periodicity_after=d1['periodicity'],
                input_rel_change=np.mean(np.abs(u1 - u0) ** 2, axis=1), finite=finite)
