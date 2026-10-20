"""D1 physics: receiver tone-response templates, physically constructed spectral interventions, and
the measurements needed to tell a credible receiver-condition change from an intervention artefact.

Model-free, numpy only. Conventions follow physics.py: 25 Msps, L-STF period 20 samples (it fills
samples ~0..190 of the 256-sample WiSig packet; ~190..255 is the start of the L-LTF), 12 occupied
tones at bins +-1..+-6 of the 20-point DFT, `tone_shape` = per-packet mean-removed 12-tone natural
log-power. A shape delta therefore has no observable mean; every delta below is re-centred.

How the original L0 `tone_rx_swap` could mislead, and what each piece here does about it:
  * templates were fitted on the very packets they were applied to  -> `response_templates` on a
    disjoint fit pool; callers apply them to a disjoint evaluation pool;
  * the r1 `physics.reshape_tones` used by the L0 full run multiplies the 20-point DFT bins of the
    8-period descriptor segment only (samples 20..179), leaves samples 0..19 and 180..255 unfiltered
    and rescales the packet. Its measured tone change equals the request *by construction* (it is
    applied in the measurement domain), but the waveform is piecewise, not the output of any LTI
    receiver. The earlier v0 version was a zero-phase circular 256-point filter (the L-LTF wraps
    into the start). `apply_response` keeps both as reproduction arms and adds short linear-
    convolution FIRs (zero-phase, minimum-phase) on the whole packet, i.e. receivers that can exist;
  * only the requested delta was ever checked  -> `measure` re-measures the realised tone change,
    CFO, SNR proxy, periodicity and IQ change on the modified packets. For an LTI filter the
    realised 12-tone change differs from the request at deeply notched tones (about a third of
    WiSig STF tones sit >13 dB below the packet mean, where leakage and the STF end at ~180-190
    samples dominate), so `realisation` also reports the error on strong tones;
  * nothing said whether the counterfactual looks like any real source packet  -> `plausibility`
    scores the modified tone shape against the fit pool's real (TX, RX) cells.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'interaction_l0'))
from physics import FS, PERIOD, TONE_BINS, stf_descriptors  # noqa: E402  (r1 physics of the L0 full run)
import physics  # noqa: E402

NB = TONE_BINS % PERIOD                    # occupied bins on the 20-point grid
TONE_HZ = TONE_BINS * FS / PERIOD
POS_UNUSED = np.array([7, 8, 9])           # +8.75..+11.25 MHz
NEG_UNUSED = np.array([11, 12, 13])        # -11.25..-8.75 MHz
NYQ = 10
FIR_MIN_TAPS = 40


def centre_shape(delta):
    d = np.atleast_2d(np.asarray(delta, float))
    return d - d.mean(1, keepdims=True)


# ------------------------------------------------------------------------- templates (fit pool)

def cell_means(v, tx, rx, day, min_count=20):
    """Per date: complete TX x RX grid of cell means of v [n,k]."""
    per = {}
    for dd in np.unique(day):
        m = day == dd
        rxs = np.unique(rx[m])
        txs = np.array([t for t in np.unique(tx[m]) if all(((tx == t) & (rx == r) & m).sum() >= min_count for r in rxs)])
        M = np.stack([np.stack([v[m & (tx == t) & (rx == r)].mean(0) for r in rxs]) for t in txs])
        per[int(dd)] = dict(txs=txs, rxs=rxs, M=M)
    return per


def decompose(M):
    g = M.mean((0, 1))
    a = M.mean(1) - g
    b = M.mean(0) - g
    return g, a, b, M - a[:, None] - b[None] - g


def response_templates(shape, cfo, tx, rx, day, min_count=20):
    """Source-only templates from the fit pool, averaged over dates.

    shape [n,12] tone shapes, cfo [n] Hz. Returns RX main effect b [R,12], TX main effect a [T,12],
    link residual i [T,R,12], full cell means M [T,R,12], and the CFO cell means / RX main effect,
    plus the within-cell deviations (packet minus its cell mean) used by the natural-variation null.
    """
    S, F = cell_means(shape, tx, rx, day, min_count), cell_means(cfo[:, None], tx, rx, day, min_count)
    first = S[min(S)]
    for dd in S:
        assert np.array_equal(S[dd]['rxs'], first['rxs']) and np.array_equal(S[dd]['txs'], first['txs']), dd
    parts = [decompose(S[dd]['M']) for dd in sorted(S)]
    cparts = [decompose(F[dd]['M']) for dd in sorted(F)]
    txs, rxs = first['txs'], first['rxs']
    dev = np.full_like(shape, np.nan)
    for dd in S:
        for i, t in enumerate(txs):
            for j, r in enumerate(rxs):
                m = (day == dd) & (tx == t) & (rx == r)
                dev[m] = shape[m] - S[dd]['M'][i, j]
    return dict(rx_names=rxs, tx_names=txs,
                a=np.mean([p[1] for p in parts], 0), b=np.mean([p[2] for p in parts], 0),
                i=np.mean([p[3] for p in parts], 0), M=np.mean([S[dd]['M'] for dd in S], 0),
                cfo_M=np.mean([F[dd]['M'][..., 0] for dd in F], 0),
                cfo_b=np.mean([p[2][:, 0] for p in cparts], 0),
                dev=dev, dev_tx=tx, dev_rx=rx)


def split_half_rx_delta(shape, cfo, tx, rx, day, seed=20261019):
    """{rx: b_r(half A) - b_r(half B)}: two independent estimates of the same receiver's response
    from disjoint halves of every fit cell. Applying this delta is the estimation-noise null."""
    rng = np.random.default_rng(seed)
    half = np.zeros(len(tx), int)
    for key in set(zip(tx.tolist(), rx.tolist(), day.tolist())):
        idx = np.flatnonzero((tx == key[0]) & (rx == key[1]) & (day == key[2]))
        half[idx[rng.permutation(len(idx))[: len(idx) // 2]]] = 1
    tA = response_templates(shape[half == 0], cfo[half == 0], tx[half == 0], rx[half == 0], day[half == 0], 5)
    tB = response_templates(shape[half == 1], cfo[half == 1], tx[half == 1], rx[half == 1], day[half == 1], 5)
    assert np.array_equal(tA['rx_names'], tB['rx_names'])
    return {int(r): centre_shape(tA['b'][k] - tB['b'][k])[0] for k, r in enumerate(tA['rx_names'])}


# ------------------------------------------------------------------------------- the filter

def grid_gain(delta):
    """[n,12] centred log-power delta -> [n,20] complex amplitude gain on the 20-point grid.
    Occupied bins get exp(delta/2); DC exp(0)=1 (delta is centred); unused bins hold the nearest
    edge tone, Nyquist the mean of the two edges (the same edge-hold the original filter used)."""
    d = centre_shape(delta)
    lg = np.zeros((len(d), PERIOD))
    lg[:, NB] = d / 2
    lg[:, POS_UNUSED] = lg[:, [6]]
    lg[:, NEG_UNUSED] = lg[:, [14]]
    lg[:, NYQ] = (lg[:, 6] + lg[:, 14]) / 2
    return lg


def _periodic_prefix(z, k):
    """k samples before the window by periodic extension of the first STF period (the window
    starts inside the L-STF); avoids a filter start-up transient that no real receiver produces."""
    reps = int(np.ceil(k / PERIOD))
    return np.tile(z[:, :PERIOD], (1, reps))[:, -k:]


def _valid_conv(x, h):
    """Row-wise np.convolve(x, h, 'valid') via FFT (rows of x and h pair up)."""
    nx, nh = x.shape[1], h.shape[1]
    N = 1 << int(np.ceil(np.log2(nx + nh - 1)))
    full = np.fft.ifft(np.fft.fft(x, N, axis=1) * np.fft.fft(h, N, axis=1), axis=1)
    return full[:, nh - 1:nx]


def apply_response(z, delta, mode='fir_zero'):
    """Apply a 12-tone log-power shape change as a receiver-like filter; returns (z', realised).

    mode
      r1_segment  physics.reshape_tones of the L0 full run (descriptor-segment DFT gain) - reproduction
      v0_circular earlier full-packet circular zero-phase filter - reproduction of the critiqued code
      fir_zero   20-tap zero-phase FIR by frequency sampling on the STF grid, *linear* convolution
                 (exact tone gains in the periodic STF, no wrap-around, causal transients only at
                 the far end where the L-LTF starts)
      fir_min    minimum-phase FIR with the same magnitude on a dense grid (40 taps, causal): a
                 receiver response with group delay; tests whether damage depends on zero phase
    realised = measured tone_shape(z') - tone_shape(z) on the descriptor window.
    """
    z = np.asarray(z, complex)
    n, L = z.shape
    d = centre_shape(delta)
    if mode == 'r1_segment':
        out = physics.reshape_tones(z, d)
    elif mode == 'v0_circular':
        f = np.fft.fftfreq(L, 1 / FS)
        order = np.argsort(TONE_HZ)
        gains = np.stack([np.interp(f, TONE_HZ[order], row[order]) for row in d])
        out = np.fft.ifft(np.fft.fft(z, axis=1) * np.exp(gains / 2), axis=1)
    elif mode == 'fir_zero':
        H = np.exp(grid_gain(d))
        h = np.fft.ifft(H, axis=1)                      # h[k], k = 0..19 circular
        # valid convolution: out[m] = sum_j kern[j] ext[m+19-j]; with a 10-sample prefix and
        # kern[j] = h[(j-9) mod 20] this is sum_k h[k] z[m-k] over lags k = -9..10
        kern = np.roll(h, PERIOD // 2 - 1, axis=1)
        pre = _periodic_prefix(z, PERIOD // 2)
        post = np.zeros((n, PERIOD // 2 - 1), complex)   # zero-phase = non-causal: unknown future
        ext = np.concatenate([pre, z, post], axis=1)
        out = _valid_conv(ext, kern)
        assert out.shape == z.shape, out.shape
    elif mode == 'fir_min':
        N = 256
        f20 = np.arange(PERIOD) / PERIOD
        lg20 = grid_gain(d)                              # log amplitude at the 20 grid points
        fN = np.arange(N) / N
        # periodic linear interpolation of log amplitude onto the dense grid
        fx = np.r_[f20, 1.0]
        L_dense = np.stack([np.interp(fN, fx, np.r_[row, row[0]]) for row in lg20])
        c = np.fft.ifft(L_dense, axis=1)
        fold = np.zeros_like(c)
        fold[:, 0] = c[:, 0]
        fold[:, 1:N // 2] = 2 * c[:, 1:N // 2]
        fold[:, N // 2] = c[:, N // 2]
        Hmin = np.exp(np.fft.fft(fold, axis=1))
        h = np.fft.ifft(Hmin, axis=1)[:, :FIR_MIN_TAPS]
        pre = _periodic_prefix(z, FIR_MIN_TAPS - 1)
        ext = np.concatenate([pre, z], axis=1)
        out = _valid_conv(ext, h)
        assert out.shape == z.shape, out.shape
    else:
        raise ValueError(mode)
    realised = stf_descriptors(out)['tone_shape'] - stf_descriptors(z)['tone_shape']
    return out, realised


# ----------------------------------------------------------------------- matched controls

def match_norm(delta, target_norm):
    d = centre_shape(delta)
    cur = np.linalg.norm(d, axis=1, keepdims=True)
    t = np.broadcast_to(np.asarray(target_norm, float).reshape(-1, 1), cur.shape)
    return np.where(cur > 1e-12, d * t / np.maximum(cur, 1e-12), 0.0)


def gaussian_direction(n, norm, rng, covariance=None):
    if covariance is None:
        d = rng.normal(size=(n, 12))
    else:
        d = rng.normal(size=(n, 12)) @ np.linalg.cholesky(covariance + 1e-9 * np.eye(12)).T
    return match_norm(d, norm)


def smooth_tilt(n, norm, sign=1.0):
    """Quadratic-sign tilt over tone index (a tilted passband), norm-matched per packet."""
    i = np.linspace(-1, 1, 12)
    shape = centre_shape(np.sign(i) * i ** 2)[0]
    return match_norm(np.tile(shape * sign, (n, 1)), norm)


def shuffle_tones(delta, rng):
    """Same values, random tone order per packet: preserves the delta's amplitude distribution and
    norm, destroys its spectral arrangement (smoothness, symmetry)."""
    d = centre_shape(delta)
    idx = np.argsort(rng.random(d.shape), axis=1)
    return np.take_along_axis(d, idx, axis=1)


# ---------------------------------------------------------------------------- measurements

def measure(z0, z1):
    d0, d1 = stf_descriptors(z0), stf_descriptors(z1)
    dt = d1['tone_shape'] - d0['tone_shape']
    rough = np.abs(np.diff(dt[:, np.argsort(TONE_HZ)], axis=1)).sum(1)
    diff = z1 - z0
    nrm = lambda v: np.sqrt(np.sum(np.abs(v) ** 2, 1))
    return dict(
        tone_norm_mean=float(np.linalg.norm(dt, axis=1).mean()),
        tone_rms=float(np.sqrt((dt ** 2).mean())), tone_max=float(np.abs(dt).max()),
        tone_roughness_median=float(np.median(rough)),
        input_rel_change=float(np.median(nrm(diff) / np.maximum(nrm(z0), 1e-30))),
        stf_rel_change=float(np.median(nrm(diff[:, :190]) / np.maximum(nrm(z0[:, :190]), 1e-30))),
        ltf_rel_change=float(np.median(nrm(diff[:, 190:]) / np.maximum(nrm(z0[:, 190:]), 1e-30))),
        power_ratio=float(np.median(np.sum(np.abs(z1) ** 2, 1) / np.sum(np.abs(z0) ** 2, 1))),
        cfo_change_hz_median_abs=float(np.median(np.abs(d1['cfo_hz'] - d0['cfo_hz']))),
        snr_change_db_median=float(np.median(d1['snr_db'] - d0['snr_db'])),
        periodicity_before=float(np.median(d0['periodicity'])), periodicity_after=float(np.median(d1['periodicity'])),
        low_periodicity_after=float((d1['periodicity'] < 0.5).mean()),
        nonfinite=int((~np.isfinite(z1)).sum()))


def realisation(z0, realised, requested, strong=-1.5):
    """Requested vs realised tone change, overall and on tones whose natural-log power is within
    `strong` of the packet's mean tone power (notched tones are where an LTI filter and the
    12-tone descriptor disagree)."""
    seg = z0[:, PERIOD:PERIOD + 8 * PERIOD].reshape(len(z0), 8, PERIOD)
    P = np.mean(np.abs(np.fft.fft(seg, axis=2)) ** 2, 1)[:, NB]
    rel = np.log(P) - np.log(P.mean(1, keepdims=True))
    err = realised - requested
    ok = rel >= strong
    return dict(err_rms=float(np.sqrt((err ** 2).mean())),
                err_rms_strong=float(np.sqrt((err[ok] ** 2).mean())) if ok.any() else float('nan'),
                strong_tone_fraction=float(ok.mean()),
                realised_over_requested=float((realised * requested).sum() / max((requested ** 2).sum(), 1e-30)))


def plausibility(shape_after, tx, rx_target, tpl, ref_quantile=95):
    """Does the modified tone shape look like a real fit-pool packet of the same TX?

    d_target   distance to the real cell mean M[t, target RX] (for swaps; NaN target -> skipped)
    d_nearest  distance to the nearest real cell mean of that TX over all source RX
    The yardstick is the fit pool's own packet-to-cell-mean distance (95th percentile): a packet
    farther than that from every real cell of its TX is outside the observed source range.
    """
    tx_k = {int(t): k for k, t in enumerate(tpl['tx_names'])}
    rx_k = {int(r): k for k, r in enumerate(tpl['rx_names'])}
    M = tpl['M']                                          # [T,R,12]
    ref = np.nanpercentile(np.linalg.norm(tpl['dev'], axis=1), ref_quantile)
    ti = np.array([tx_k[int(t)] for t in tx])
    cells = M[ti]                                         # [n,R,12]
    dist = np.linalg.norm(cells - shape_after[:, None, :], axis=2)
    out = dict(ref_q95=float(ref), d_nearest_median=float(np.median(dist.min(1))),
               frac_within_source_range=float((dist.min(1) <= ref).mean()))
    if rx_target is not None:
        rj = np.array([rx_k[int(r)] for r in rx_target])
        d_t = dist[np.arange(len(ti)), rj]
        out.update(d_target_median=float(np.median(d_t)),
                   frac_nearest_is_target=float((dist.argmin(1) == rj).mean()))
    return out


def template_geometry(tpl):
    b = centre_shape(tpl['b'])
    D = np.linalg.norm(b[:, None] - b[None], axis=2)
    iu = np.triu_indices(len(b), 1)
    sv = np.linalg.svd(b - b.mean(0), compute_uv=False)
    a = centre_shape(tpl['a'])
    Qa = np.linalg.svd(a - a.mean(0), full_matrices=False)[2][:3].T
    Qb = np.linalg.svd(b - b.mean(0), full_matrices=False)[2][:3].T
    return dict(rx_pair_dist_median=float(np.median(D[iu])), rx_pair_dist_min=float(D[iu].min()),
                rx_pair_dist_max=float(D[iu].max()),
                rx_effective_rank=float(sv.sum() ** 2 / (sv ** 2).sum()),
                rx_top1_share=float(sv[0] ** 2 / (sv ** 2).sum()),
                rx_b_norm_median=float(np.median(np.linalg.norm(b, axis=1))),
                tx_a_norm_median=float(np.median(np.linalg.norm(a, axis=1))),
                link_i_norm_median=float(np.median(np.linalg.norm(tpl['i'], axis=2))),
                within_cell_dev_norm_median=float(np.nanmedian(np.linalg.norm(tpl['dev'], axis=1))),
                tx_rx_top3_principal_cosines=[float(v) for v in np.linalg.svd(Qa.T @ Qb, compute_uv=False)],
                cfo_rx_sd_hz=float(tpl['cfo_b'].std(ddof=1)))
