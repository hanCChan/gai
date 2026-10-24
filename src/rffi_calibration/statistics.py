"""Source statistics: receiver-equal class means and the between/within-condition decomposition.

Working model (a hypothesis under test, not a claim that RF features are Gaussian or that
receiver shifts are independent):

    z[t, r, i] = mu[t] + d[t, r] + eps[t, r, i]
    d[t, r]   ~ N(0, K[t])        between-condition (receiver) shift of class t
    eps       ~ N(0, Sigma[t])    within-condition packet scatter

V1 keeps every covariance DIAGONAL (80/128-D features, 4-14 source RX). Everything is float64 and
written in torch so that T3 can back-propagate through the same code that the frozen heads use.

Estimators (EXPERIMENT_PROTOCOL.md §4.2; constants fixed before any data run):
    u[t,r]    cell mean, v[t,r] cell variance (ddof=1), n[t,r] cell size; cells with n < min_cell
              are not used. R[t] = number of usable cells of class t.
    mu[t]     = mean_r u[t,r]                        (receivers weighted equally, not packets)
    C[t,r]    = v[t,r] / n[t,r]                      (sampling noise of a cell mean)
    B[t]      = var_r(u[t,r], ddof=1)
    K_raw[t]  = B[t] - mean_r C[t,r]                 (cell-mean noise removed from B)
    K_pos     = max(K_raw, 0);  K_pool = mean_t K_pos;  a_K = (R-1)/(R-1+kappa_K)
    K[t]      = a_K K_pos[t] + (1-a_K) K_pool
    Sigma_cls = mean_r v[t,r];  Sigma_pool = mean_t Sigma_cls
    a_S[t]    = df[t]/(df[t]+dim),  df[t] = sum_r (n[t,r]-1)
    Sigma[t]  = max(a_S Sigma_cls + (1-a_S) Sigma_pool, floor),  floor = abs + rel*mean(Sigma_pool)
    U[t]      = K[t]/R[t] + sum_r C[t,r]/R[t]^2      (finite-source-centre correction)
    A[t]      = K[t] + U[t]   (or K[t] in the fixed-centre ablation)

A is the prior covariance of the target receiver's class centre around mu. It is an empirical-Bayes
plug-in: the estimation uncertainty of K and Sigma themselves is NOT integrated out.

Single source receiver: K is not identifiable. fit_source_stats raises SingleReceiverError instead of
silently setting K = 0; the registered transfer variant (transfer_prior) is a separate, named model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch

DTYPE = torch.float64


class SingleReceiverError(ValueError):
    """Between-receiver covariance requested with fewer than two usable source receivers."""


class MissingClassError(ValueError):
    """A registered class has no usable source cell (or no support where support is required)."""


@dataclass(frozen=True)
class StatsConfig:
    kappa_K: float = 8.0            # shrinkage strength of K[t] towards K_pool; V1 constant, not tuned
    floor_abs: float = 1e-8
    floor_rel: float = 1e-5         # relative to the source variance scale mean(Sigma_pool)
    min_cell: int = 2               # packets needed for a (class, condition) cell to count
    finite_center: bool = True      # A = K + U (True) or A = K (fixed-centre ablation)
    pooled: bool = False            # ablation: K_pool / Sigma_pool for every class

    def tag(self):
        return 'H4' + ('' if self.finite_center else '-AK') + ('-pooled' if self.pooled else '')


@dataclass
class SourceStats:
    """Everything a calibration head may know about the source. No target data inside."""
    mu: torch.Tensor                # (T, d) prior centre of each class on a new receiver
    sigma: torch.Tensor             # (T, d) within-condition variance (floored)
    A: torch.Tensor                 # (T, d) prior variance of the new receiver's class centre
    sigma_pool: torch.Tensor        # (d,)
    floor: float
    mode: str                       # 'multi_rx' | 'transfer_pooled_prior'
    diag: dict = field(default_factory=dict)   # K, U, K_raw, n_rx, ... (reporting only)

    @property
    def num_classes(self):
        return self.mu.shape[0]

    @property
    def dim(self):
        return self.mu.shape[1]


def _as64(x):
    return torch.as_tensor(x).to(DTYPE)


def _as_long(x):
    return torch.as_tensor(x).long()


def cell_moments(z, y, c, num_classes, num_conditions):
    """Per (class, condition) cell: size n (T, R), mean u (T, R, d), variance v (ddof=1, T, R, d).

    Two-pass (mean, then squared residuals) for numerical stability; differentiable in z.
    Cells with n < 2 get v = 0 (they are masked out by the callers)."""
    z, y, c = _as64(z), _as_long(y), _as_long(c)
    if z.ndim != 2 or len(z) != len(y) or len(z) != len(c):
        raise ValueError('cell_moments: z must be (N, d) with matching y and c')
    if len(y) and (int(y.min()) < 0 or int(y.max()) >= num_classes):
        raise MissingClassError(f'labels outside [0, {num_classes})')
    if len(c) and (int(c.min()) < 0 or int(c.max()) >= num_conditions):
        raise ValueError(f'condition index outside [0, {num_conditions})')
    T, R, d = num_classes, num_conditions, z.shape[1]
    idx = y * R + c
    n = torch.bincount(idx, minlength=T * R).to(DTYPE)
    s1 = torch.zeros(T * R, d, dtype=DTYPE, device=z.device).index_add(0, idx, z)
    u = s1 / n.clamp_min(1.)[:, None]
    r2 = (z - u.index_select(0, idx)) ** 2
    s2 = torch.zeros(T * R, d, dtype=DTYPE, device=z.device).index_add(0, idx, r2)
    v = s2 / (n - 1.).clamp_min(1.)[:, None]
    v = torch.where((n >= 2)[:, None], v, torch.zeros_like(v))
    return n.view(T, R), u.view(T, R, d), v.view(T, R, d)


def _check_conditions(c):
    """Map arbitrary condition ids to 0..R-1."""
    c = _as_long(c)
    uniq, inv = torch.unique(c, return_inverse=True)
    return inv, uniq


def estimate_between_rx_covariance(n, u, v, valid):
    """B, mean cell-mean noise and K_raw per class; requires >= 2 usable cells for EVERY class."""
    R_t = valid.sum(1).to(DTYPE)                                     # (T,)
    if bool((R_t < 2).any()):
        bad = torch.nonzero(R_t < 2).flatten().tolist()
        raise SingleReceiverError(
            f'between-receiver covariance needs >= 2 usable source receivers per class; classes {bad} '
            f'have {R_t[R_t < 2].long().tolist()}. Use the registered transfer variant instead of K = 0.')
    w = valid.to(DTYPE)[..., None]                                   # (T, R, 1)
    mu = (u * w).sum(1) / R_t[:, None]
    B = (((u - mu[:, None]) ** 2) * w).sum(1) / (R_t - 1.)[:, None]
    C = v / n.clamp_min(1.)[..., None]                               # (T, R, d)
    Cbar = (C * w).sum(1) / R_t[:, None]
    return mu, B, C, Cbar, R_t


def fit_source_stats(z, y, condition, num_classes, cfg: StatsConfig = StatsConfig()):
    """Fit the decomposition on SOURCE fitting records only (multi-receiver mode)."""
    z = _as64(z)
    c, cond_ids = _check_conditions(condition)
    T, d = num_classes, z.shape[1]
    n, u, v = cell_moments(z, y, c, T, len(cond_ids))
    valid = n >= cfg.min_cell
    if bool((valid.sum(1) == 0).any()):
        raise MissingClassError(f'classes without usable source cells: '
                                f'{torch.nonzero(valid.sum(1) == 0).flatten().tolist()}')
    mu, B, C, Cbar, R_t = estimate_between_rx_covariance(n, u, v, valid)
    w = valid.to(DTYPE)[..., None]
    K_raw = B - Cbar
    K_pos = K_raw.clamp_min(0.)
    K_pool = K_pos.mean(0)
    a_K = ((R_t - 1.) / (R_t - 1. + cfg.kappa_K))[:, None]
    K = a_K * K_pos + (1. - a_K) * K_pool
    sig_cls = (v * w).sum(1) / R_t[:, None]
    sig_pool = sig_cls.mean(0)
    floor = float(cfg.floor_abs + cfg.floor_rel * sig_pool.detach().mean())
    df = ((n - 1.).clamp_min(0.) * valid.to(DTYPE)).sum(1)
    a_S = (df / (df + d))[:, None]
    sigma = (a_S * sig_cls + (1. - a_S) * sig_pool).clamp_min(floor)
    sig_pool = sig_pool.clamp_min(floor)
    inv_n = (valid.to(DTYPE) / n.clamp_min(1.)).sum(1)               # sum_r 1/n[t,r]
    if cfg.pooled:
        K = K_pool.expand(T, d)
        sigma = sig_pool.expand(T, d)
        U = K / R_t[:, None] + sig_pool * (inv_n / R_t ** 2)[:, None]
    else:
        U = K / R_t[:, None] + (C * w).sum(1) / (R_t ** 2)[:, None]
    A = K + U if cfg.finite_center else K
    diag = dict(K=K, K_raw=K_raw, K_pool=K_pool, U=U, B=B, Cbar=Cbar, n_rx=R_t, cell_n=n,
                conditions=cond_ids, frac_K_clamped=float((K_raw.detach() < 0).to(DTYPE).mean()),
                cfg=cfg)
    return SourceStats(mu=mu, sigma=sigma, A=A, sigma_pool=sig_pool, floor=floor,
                       mode='multi_rx', diag=diag)


def transfer_prior(pool: SourceStats, z_reg, y_reg, num_classes, min_count=1):
    """TRANSFER_POOLED_PRIOR (single source receiver, e.g. OSU RX1 -> RX2; protocol §4.5).

    pool:  stats from a DIFFERENT dataset's multi-RX fitting data in the same frozen representation
           (only its class-agnostic K_pool and Sigma_pool are used).
    z_reg: registration packets of the new identities on the one source receiver.
    Model: both receivers shift independently with covariance K_pool, so the target centre differs
    from the registration mean u[t] by 2 K_pool plus the registration-mean noise Sigma_pool/n_src[t].
    The factor 2 is a modelling assumption (independent, identically distributed RX shifts)."""
    z = _as64(z_reg)
    y = _as_long(y_reg)
    T, d = num_classes, z.shape[1]
    if z.shape[1] != pool.dim:
        raise ValueError('transfer_prior: representation dimension differs from the pooled prior')
    n = torch.bincount(y, minlength=T).to(DTYPE)
    if bool((n < min_count).any()):
        raise MissingClassError(f'classes without registration packets: '
                                f'{torch.nonzero(n < min_count).flatten().tolist()}')
    u = torch.zeros(T, d, dtype=DTYPE, device=z.device).index_add(0, y, z) / n[:, None]
    K_pool = pool.diag['K_pool'].to(DTYPE)
    sig = pool.sigma_pool.expand(T, d)
    A = 2. * K_pool + sig / n[:, None]
    return SourceStats(mu=u, sigma=sig, A=A, sigma_pool=pool.sigma_pool, floor=pool.floor,
                       mode='transfer_pooled_prior', diag=dict(n_src=n, K_pool=K_pool))


def support_means(z_support, y_support, num_classes):
    """Per-class support mean s (T, d), count m (T,), and sample variance (ddof=1; 0 when m < 2)."""
    z, y = _as64(z_support), _as_long(y_support)
    T = num_classes
    if len(y) and (int(y.min()) < 0 or int(y.max()) >= T):
        raise MissingClassError(f'support labels outside [0, {T})')
    d = z.shape[1]
    m = torch.bincount(y, minlength=T).to(DTYPE)
    s = torch.zeros(T, d, dtype=DTYPE, device=z.device).index_add(0, y, z) / m.clamp_min(1.)[:, None]
    r2 = (z - s[y]) ** 2
    var = torch.zeros(T, d, dtype=DTYPE, device=z.device).index_add(0, y, r2) / (m - 1.).clamp_min(1.)[:, None]
    var = torch.where((m >= 2)[:, None], var, torch.zeros_like(var))
    return s, m, var


def posterior_predictive(stats: SourceStats, s, m):
    """Diagonal Gaussian conditioning of each class centre on its support mean.

    s (T, d) support means (any finite value where m == 0), m (T,) support counts.
        R_noise = Sigma / m,  G = A / (A + R_noise)
        theta   = mu + G (s - mu),  V = A R_noise / (A + R_noise)        (m > 0)
        theta   = mu,               V = A                                (m = 0)
        Q       = Sigma + V
    Returns (theta, Q). Elementwise because all covariances are diagonal."""
    m = _as64(m).to(stats.mu.device)
    has = (m > 0)[:, None]
    Rn = stats.sigma / m.clamp_min(1.)[:, None]
    denom = stats.A + Rn
    G = stats.A / denom
    s = torch.where(has, _as64(s), stats.mu)
    theta = torch.where(has, stats.mu + G * (s - stats.mu), stats.mu)
    V = torch.where(has, stats.A * Rn / denom, stats.A)
    return theta, stats.sigma + V


def diag_gaussian_logpdf(z, center, var, log_prior=None):
    """log N(z | center[t], diag var[t]) for all classes t: (n, T). Full score incl. logdet."""
    z = _as64(z)
    iv = 1. / var
    maha = (z ** 2) @ iv.T - 2. * z @ (center * iv).T + ((center ** 2) * iv).sum(1)[None]
    ll = -0.5 * (maha + torch.log(var).sum(1)[None] + z.shape[1] * math.log(2 * math.pi))
    if log_prior is not None:
        ll = ll + log_prior[None]
    return ll


# ---------------------------------------------------------------- full-covariance diagnostics
def psd_project(S, eps=0.):
    """Symmetric PSD projection (eigenvalues clipped at eps)."""
    S = 0.5 * (S + S.transpose(-1, -2))
    w, Q = torch.linalg.eigh(S)
    return (Q * w.clamp_min(eps)[..., None, :]) @ Q.transpose(-1, -2)


def safe_cholesky(S, jitter=0., max_tries=6):
    """Cholesky with recorded jitter escalation; raises instead of returning NaN."""
    eye = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device)
    scale = float(torch.diagonal(S, dim1=-2, dim2=-1).abs().mean().clamp_min(1e-300))
    j = jitter
    for k in range(max_tries):
        L, info = torch.linalg.cholesky_ex(S + j * eye)
        if int(info.max()) == 0:
            return L, j
        j = scale * 10. ** (-10 + 2 * k) if j == 0 else j * 100.
    raise torch.linalg.LinAlgError(f'cholesky failed after jitter {j:.3g}')


def full_gaussian_logpdf(z, center, chol):
    """log N(z | center[t], L_t L_t^T) for chol (T, d, d) or shared (d, d)."""
    z = _as64(z)
    T, d = center.shape
    if chol.ndim == 2:
        chol = chol.expand(T, d, d)
    diff = (z[None] - center[:, None]).transpose(1, 2)            # (T, d, n)
    sol = torch.linalg.solve_triangular(chol, diff, upper=False)  # L^-1 (z - c)
    maha = (sol ** 2).sum(1).T                                    # (n, T)
    logdet = 2. * torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    return -0.5 * (maha + logdet[None] + d * math.log(2 * math.pi))


def gaussian_condition_full(mu, A, Sigma, s, m):
    """Reference (full-covariance) conditioning for tests: centre ~ N(mu, A), support mean
    s | centre ~ N(centre, Sigma/m). Returns posterior mean and predictive covariance Sigma + V."""
    if m == 0:
        return mu, Sigma + A
    Rn = Sigma / m
    L, _ = safe_cholesky(A + Rn)
    G = torch.cholesky_solve(A, L).T                              # A (A+Rn)^-1  (symmetric matrices)
    theta = mu + G @ (s - mu)
    V = A - G @ A
    return theta, Sigma + 0.5 * (V + V.T)


def with_config(stats: SourceStats, **kw):
    return replace(stats, **kw)
