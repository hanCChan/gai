"""D1 source statistics: the full H4 parameter panel plus the shared PCA / pooled-moment estimators
needed by the fixed diagnostic matrix (configs/rffi_post_a/diagnostic_matrix.json, plan §6).

Everything here is fitted on SOURCE records only. The functions take features, labels, physical
receiver ids and (optionally) dates/positions of source records; none accepts target data.
Dates and positions are metadata for the stability block; they never become extra "receivers".

Reuses rffi_calibration.statistics so that D_CC is the H4 estimator, not a re-implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from rffi_calibration.statistics import (DTYPE, MissingClassError, SingleReceiverError, StatsConfig, _as64,
                                         _as_long, _check_conditions, cell_moments, fit_source_stats, psd_project)


class NotApplicable(ValueError):
    """A registered slot cannot be computed under its fixed definition (reported as N/A, never 0)."""


@dataclass
class SourceBundle:
    T: int
    d: int
    # diagonal panel (plan §6.2), all float64, shapes (T, d) unless stated
    mu: torch.Tensor
    cell_n: torch.Tensor           # (T, R)
    n_rx: torch.Tensor             # (T,)
    B: torch.Tensor
    Cbar: torch.Tensor
    K_raw: torch.Tensor
    K_pos: torch.Tensor
    K_pool: torch.Tensor           # (d,)
    K: torch.Tensor                # class-shrunk
    a_K: torch.Tensor              # (T, 1)
    sigma_class_raw: torch.Tensor
    sigma_class_shrunk: torch.Tensor
    sigma_pool: torch.Tensor       # (d,) floored
    a_S: torch.Tensor              # (T, 1)
    omega: torch.Tensor            # U_sampling = sum_r C[t,r] / R_t^2
    U_K: torch.Tensor              # K / R_t
    A: torch.Tensor                # K + U (H4)
    floor: float
    cfg: StatsConfig
    # shared-covariance blocks (H2L estimator), raw and PCA coordinates
    raw_mu: torch.Tensor = None    # (T, d) receiver-equal mean over complete cells
    raw_S: torch.Tensor = None     # (d, d) packet-pooled residual covariance, 10% isotropic shrinkage
    pca_m0: torch.Tensor = None
    pca_V: torch.Tensor = None     # (d, k)
    pca_mu: torch.Tensor = None    # (T, k)
    pca_S: torch.Tensor = None     # (k, k) == H2L covariance
    pooled_bayes: dict = None      # K0, Sigma0, U0, A0 (k x k) or None + reason
    extra: dict = field(default_factory=dict)


def _check_source(rx, held_rx):
    rx = np.asarray(rx)
    if held_rx is not None:
        bad = sorted(set(map(str, rx)) & set(map(str, held_rx)))
        if bad:
            raise ValueError(f'held RX {bad} present in source records: refused (no held RX in source stats)')
    if len(np.unique(rx)) < 2:
        raise SingleReceiverError('source diagnostics need >= 2 physical receivers')


def _shared_covariance(g, y, c, T, shrink):
    """H2L estimator: residuals around every (class, RX) cell mean pooled over packets,
    S = res^T res / N, then (1-shrink) S + shrink tr(S)/k I. Requires every cell (as EigenRX)."""
    R = int(c.max()) + 1
    n, u, _ = cell_moments(g, y, c, T, R)
    if bool((n == 0).any()):
        raise NotApplicable('shared-covariance block needs every (class, RX) source cell')
    res = g - u[y, c]
    S = res.T @ res / len(res)
    k = g.shape[1]
    S = (1 - shrink) * S + shrink * torch.trace(S) / k * torch.eye(k, dtype=DTYPE)
    return u.mean(1), S, n, u


def _pooled_bayes_moments(g, y, c, T):
    """Plan §6.5: K0 = mean_t PSD(B_t - mean_r C_t,r), U0 = K0/R + mean_t sum_r C_t,r / R^2 (full k x k).
    Same R for every class is required; otherwise N/A (no silent change of estimator)."""
    R = int(c.max()) + 1
    n, u, _ = cell_moments(g, y, c, T, R)
    if bool((n < 2).any()):
        return None, 'incomplete cells (n<2) for pooled Bayesian moments'
    k = g.shape[1]
    K_terms, Om_terms = [], []
    for t in range(T):
        ut = u[t]                                                     # (R, k)
        Bt = (ut - ut.mean(0)).T @ (ut - ut.mean(0)) / (R - 1)
        Cs = []
        for r in range(R):
            x = g[(y == t) & (c == r)] - u[t, r]
            Cs.append(x.T @ x / (len(x) - 1) / len(x))                # cell-mean sampling covariance
        Cs = torch.stack(Cs)
        K_terms.append(psd_project(Bt - Cs.mean(0)))
        Om_terms.append(Cs.sum(0) / R ** 2)
    K0 = torch.stack(K_terms).mean(0)
    U0 = K0 / R + torch.stack(Om_terms).mean(0)
    return dict(K0=K0, U0=U0, A0=K0 + U0, R=R, k=k), None


def _codes(rx):
    """Physical RX ids (int or str) -> 0..R-1 (dates never enter)."""
    return np.unique(np.asarray(rx).astype(str), return_inverse=True)[1]


def source_diagnostics(z_source, y_source, rx_source, metadata=None, config=None):
    """Only source records; never accepts target query labels.

    metadata: {'held_rx': [...]} (checked: none may appear among source RX).
    config:   {'stats': StatsConfig kwargs, 'pca_var': .995, 'pca_cap': 32, 'shrink': .1}.
    """
    metadata, config = metadata or {}, config or {}
    _check_source(rx_source, metadata.get('held_rx'))
    cfg = StatsConfig(**config.get('stats', {}))
    z = _as64(z_source)
    y = _as_long(y_source)
    c, _ = _check_conditions(_codes(rx_source))
    T = int(metadata.get('num_classes', int(y.max()) + 1))
    d = z.shape[1]
    st = fit_source_stats(z, y, c, T, cfg)                            # the H4 estimator itself
    n, u, v = cell_moments(z, y, c, T, int(c.max()) + 1)
    valid = n >= cfg.min_cell
    w = valid.to(DTYPE)[..., None]
    R_t = valid.sum(1).to(DTYPE)
    sig_cls = (v * w).sum(1) / R_t[:, None]
    df = ((n - 1.).clamp_min(0.) * valid.to(DTYPE)).sum(1)
    a_S = (df / (df + d))[:, None]
    C = v / n.clamp_min(1.)[..., None]
    omega = (C * w).sum(1) / (R_t ** 2)[:, None]
    K = st.diag['K']
    b = SourceBundle(T=T, d=d, mu=st.mu, cell_n=n, n_rx=R_t, B=st.diag['B'], Cbar=st.diag['Cbar'],
                     K_raw=st.diag['K_raw'], K_pos=st.diag['K_raw'].clamp_min(0.), K_pool=st.diag['K_pool'],
                     K=K, a_K=((R_t - 1.) / (R_t - 1. + cfg.kappa_K))[:, None], sigma_class_raw=sig_cls,
                     sigma_class_shrunk=st.sigma, sigma_pool=st.sigma_pool, a_S=a_S, omega=omega,
                     U_K=K / R_t[:, None], A=st.A, floor=st.floor, cfg=cfg)
    # independent identity check: our panel must reproduce the H4 estimator exactly
    if not torch.allclose(b.U_K + b.omega + K, st.A, rtol=0, atol=1e-12):
        raise AssertionError('panel does not reproduce H4 A = K + K/R + Omega')
    shrink = config.get('shrink', .1)
    try:
        b.raw_mu, b.raw_S, _, _ = _shared_covariance(z, y, c, T, shrink)
        b.pca_m0 = z.mean(0)
        X = z - b.pca_m0
        _, sv, Vt = torch.linalg.svd(X, full_matrices=False)
        ev = sv ** 2
        k = int(torch.searchsorted(torch.cumsum(ev, 0) / ev.sum(),
                                   torch.tensor(config.get('pca_var', .995), dtype=DTYPE))) + 1
        k = max(2, min(config.get('pca_cap', 32), k))
        b.pca_V = Vt[:k].T
        g = X @ b.pca_V
        b.pca_mu, b.pca_S, _, _ = _shared_covariance(g, y, c, T, shrink)
        pb, why = _pooled_bayes_moments(g, y, c, T)
        if pb is not None:
            pb['Sigma0'] = b.pca_S
        b.pooled_bayes = pb if pb is not None else dict(not_applicable=why)
    except NotApplicable as e:
        b.extra['shared_block_not_applicable'] = str(e)
    return b


def panel_summary(b: SourceBundle, budgets=(1, 2, 5, 10)):
    """Quantile summary of the H4 parameter panel and of G per budget (plan §6.2), in source pooled
    variance units where a scale is involved. All classes are included."""
    q = lambda x: {f'q{p}': float(torch.quantile(x.flatten(), p / 100.)) for p in (10, 50, 90)}
    sp = b.sigma_pool[None]
    out = dict(T=b.T, d=b.d, n_rx=b.n_rx.tolist(), a_K=b.a_K.flatten().tolist(), a_S=b.a_S.flatten().tolist(),
               frac_K_raw_clamped=float((b.K_raw < 0).to(DTYPE).mean()),
               frac_sigma_floor_hit=float((b.sigma_class_shrunk <= b.floor * (1 + 1e-12)).to(DTYPE).mean()),
               K_over_sigma_pool=q(b.K / sp), K_raw_over_sigma_pool=q(b.K_raw / sp),
               sigma_class_over_pool=q(b.sigma_class_shrunk / sp), U_K_over_sigma_pool=q(b.U_K / sp),
               omega_over_sigma_pool=q(b.omega / sp), A_over_sigma_pool=q(b.A / sp),
               logdet_sigma_class=q(torch.log(b.sigma_class_shrunk).sum(1)),
               per_class_K_mean=(b.K / sp).mean(1).tolist(),
               per_class_sigma_ratio_mean=(b.sigma_class_shrunk / sp).mean(1).tolist())
    G = {}
    for m in budgets:
        g = b.A / (b.A + b.sigma_class_shrunk / m)
        G[m] = dict(q(g), frac_lt_005=float((g < .05).to(DTYPE).mean()), frac_gt_095=float((g > .95).to(DTYPE).mean()))
        Q = b.sigma_class_shrunk + b.A * (b.sigma_class_shrunk / m) / (b.A + b.sigma_class_shrunk / m)
        G[m]['logdet_Q'] = q(torch.log(Q).sum(1))
    out['G'] = G
    if b.pca_V is not None:
        out['pca_k'] = int(b.pca_V.shape[1])
    if b.pooled_bayes is not None and 'K0' in b.pooled_bayes:
        ev = torch.linalg.eigvalsh(b.pooled_bayes['K0'])
        out['K0_eig'] = dict(min=float(ev.min()), max=float(ev.max()), rank=int((ev > 1e-12 * ev.max()).sum()))
    elif b.pooled_bayes is not None:
        out['pooled_bayes'] = b.pooled_bayes
    return out


# --------------------------------------------------------------------------- stability block §6.6
def _rel(a, b):
    return float((a - b).abs().sum() / b.abs().sum().clamp_min(1e-300))


def _spearman(a, b):
    ra = torch.argsort(torch.argsort(a.flatten())).to(DTYPE)
    rb = torch.argsort(torch.argsort(b.flatten())).to(DTYPE)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-300))


def loro_stability(z, y, rx, num_classes, cfg=StatsConfig(), budgets=(1, 5)):
    """Leave one SOURCE physical RX out and re-estimate K / Sigma / G. Internal statistical stability
    only: the encoder was trained on these RX, so this is not encoder-held generalisation."""
    rx = np.asarray(rx).astype(str)
    full = fit_source_stats(z, y, _codes(rx), num_classes, cfg)
    rows = []
    for r in np.unique(rx):
        keep = rx != r
        if len(np.unique(rx[keep])) < 2:
            rows.append(dict(left_out=r, status='N/A single remaining RX'))
            continue
        st = fit_source_stats(_as64(z)[torch.as_tensor(keep)], np.asarray(y)[keep], _codes(rx[keep]), num_classes, cfg)
        row = dict(left_out=r, rel_K=_rel(st.diag['K'], full.diag['K']), rel_sigma=_rel(st.sigma, full.sigma),
                   spearman_K=_spearman(st.diag['K'], full.diag['K']),
                   clamp_rate=st.diag['frac_K_clamped'], clamp_rate_full=full.diag['frac_K_clamped'])
        for m in budgets:
            g0 = full.A / (full.A + full.sigma / m)
            g1 = st.A / (st.A + st.sigma / m)
            row[f'rel_G_m{m}'] = _rel(g1, g0)
        rows.append(row)
    return rows


def half_split_stability(z, y, rx, position, num_classes, cfg=StatsConfig()):
    """Split every (class, RX) cell by original position order into first/second half and compare
    Sigma, cell-mean noise and K. Halves are row-order halves, not verified time."""
    rx, y, position = np.asarray(rx).astype(str), np.asarray(y), np.asarray(position)
    first = np.zeros(len(y), bool)
    for t in np.unique(y):
        for r in np.unique(rx):
            ix = np.flatnonzero((y == t) & (rx == r))
            o = ix[np.argsort(position[ix], kind='stable')]
            first[o[:len(o) // 2]] = True
    z = _as64(z)
    a = fit_source_stats(z[torch.as_tensor(first)], y[first], _codes(rx[first]), num_classes, cfg)
    b = fit_source_stats(z[torch.as_tensor(~first)], y[~first], _codes(rx[~first]), num_classes, cfg)
    sp = a.sigma_pool
    return dict(rel_sigma=_rel(a.sigma, b.sigma), rel_K=_rel(a.diag['K'], b.diag['K']),
                spearman_K=_spearman(a.diag['K'], b.diag['K']),
                cbar_ratio=float((a.diag['Cbar'] / b.diag['Cbar'].clamp_min(1e-300)).median()),
                half_mean_shift_over_sigma_pool=float((((a.mu - b.mu) ** 2) / sp).mean()))


def date_decomposition(z, y, rx, date):
    """WiSig only: per (TX, RX, date) means; share of the within-RX variance that is between dates.
    (RX, date) pairs are NOT treated as independent receivers."""
    z = _as64(z)
    y, rx, date = np.asarray(y), np.asarray(rx).astype(str), np.asarray(date)
    between, within = [], []
    for t in np.unique(y):
        for r in np.unique(rx):
            m = (y == t) & (rx == r)
            ds = np.unique(date[m])
            if len(ds) < 2:
                continue
            means = torch.stack([z[torch.as_tensor(m & (date == dd))].mean(0) for dd in ds])
            between.append(means.var(0, unbiased=True))
            within.append(torch.stack([z[torch.as_tensor(m & (date == dd))].var(0, unbiased=True) for dd in ds]).mean(0))
    if not between:
        return dict(status='N/A no cell with two dates')
    Bd, Wd = torch.stack(between), torch.stack(within)
    share = Bd / (Bd + Wd)
    return dict(cross_date_share_median=float(share.median()), cross_date_share_q90=float(torch.quantile(share, .9)),
                cells=len(between))


def lag_correlation(z, y, rx, position, lags=(1, 2, 4)):
    """Residual (cell-mean removed) correlation between rows `lag` apart in original position order.
    Row-order correlation only; no effective-sample-size model is fitted from it."""
    z = _as64(z)
    y, rx, position = np.asarray(y), np.asarray(rx).astype(str), np.asarray(position)
    out = {}
    for lag in lags:
        num = torch.zeros(z.shape[1], dtype=DTYPE)
        den = torch.zeros(z.shape[1], dtype=DTYPE)
        for t in np.unique(y):
            for r in np.unique(rx):
                ix = np.flatnonzero((y == t) & (rx == r))
                if len(ix) <= lag + 1:
                    continue
                o = ix[np.argsort(position[ix], kind='stable')]
                e = z[torch.as_tensor(o)] - z[torch.as_tensor(o)].mean(0)
                num += (e[lag:] * e[:-lag]).sum(0)
                den += (e ** 2).sum(0)
        rho = num / den.clamp_min(1e-300)
        out[f'lag{lag}'] = dict(median=float(rho.median()), q90=float(torch.quantile(rho, .9)))
    return out
