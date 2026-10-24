"""Calibration heads H0-H5 on frozen features (EXPERIMENT_PROTOCOL.md §5).

API (identical for every head; the query label never appears in it):
    head = SomeHead(...).fit(z_source, y_source, condition_source, num_classes)   # source only
    cal  = head.adapt(z_support, y_support)                                       # trusted target labels
    p    = cal.predict_proba(z_query)                                             # (n, T), rows sum to 1

`fit` sees only source fitting records; `adapt` sees only the m labelled support packets per TX of the
target receiver; `predict_*` sees only query features. Every CalibratedHead scores ALL classes with a
complete score (Mahalanobis + log det + log prior) and is inductive: the prediction for one query does
not depend on any other query in the batch.

Heads and attribution (details and RF-specific adaptation in NOVELTY_AND_BASELINE_AUDIT.md):
    H0   NativeHead          the trained linear classifier (0-shot cost reference)
    H0S  SourceMeanLDA       source class means + pooled within-condition variance (0-shot reference)
    H1   TargetPrototype     target support mean, pooled diag LDA (metric='pooled') or Euclidean
                             (Snell et al. 2017 ProtoNet decision rule on frozen features)
    H2   ScalarShrink        (1-a) mu + a s; a fixed, selected on source_selection, or 'legacy' = m/(m+2)
    H2L  LegacyP2EProto      exact re-implementation of P2E `p2e_fewshot.calibrate(mode='proto')`
                             (PCA .995/cap 32, pooled full covariance shrunk .1, m/(m+2)); the verified baseline
    H3   GaussianShrink      ordinary per-class Gaussian with covariance shrinkage; no between/within split
    H4   PosteriorPredictive receiver/noise decomposition + diagonal posterior predictive (statistics.py)
    H5a  SimpleShotCL2N      Wang et al. 2019: centre by source mean, L2-normalise, nearest centroid
    H5b  SimpleCNAPSFrozen   Bateni et al. 2020 Mahalanobis head (class/task covariance blend + beta I)
                             WITHOUT its FiLM feature adaptation -> a frozen-head port, not a reproduction
    H5c  NIWPredictive       MetaQDA-style (Zhang et al. 2021) Normal-inverse-Wishart predictive Student-t,
                             prior set from source moments with a registered kappa0 grid, not meta-learned
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .statistics import (DTYPE, MissingClassError, SourceStats, StatsConfig, cell_moments,
                         diag_gaussian_logpdf, fit_source_stats, full_gaussian_logpdf, posterior_predictive,
                         psd_project, safe_cholesky, support_means, transfer_prior, _as64, _as_long,
                         _check_conditions)


# ---------------------------------------------------------------------------------- calibrated heads
class CalibratedHead:
    """A frozen decision rule. Only query features enter predict_*."""
    temperature: float = 1.0

    def log_scores(self, z):
        raise NotImplementedError

    def predict_logits(self, z):
        return self.log_scores(z) / self.temperature

    def predict_proba(self, z):
        return torch.softmax(self.predict_logits(z), dim=-1)

    def with_temperature(self, t):
        self.temperature = float(t)
        return self


@dataclass(eq=False)
class LinearCalibrated(CalibratedHead):
    W: torch.Tensor
    b: torch.Tensor
    temperature: float = 1.0

    def log_scores(self, z):
        return _as64(z) @ self.W.T + self.b


@dataclass(eq=False)
class DiagGaussianCalibrated(CalibratedHead):
    center: torch.Tensor          # (T, d)
    var: torch.Tensor             # (T, d)
    log_prior: torch.Tensor       # (T,)
    temperature: float = 1.0

    def log_scores(self, z):
        return diag_gaussian_logpdf(z, self.center, self.var, self.log_prior)


@dataclass(eq=False)
class FullGaussianCalibrated(CalibratedHead):
    center: torch.Tensor          # (T, d) in the head's own coordinates
    chol: torch.Tensor            # (d, d) shared or (T, d, d)
    log_prior: torch.Tensor
    transform: object = None      # optional callable z -> head coordinates (source-fitted)
    temperature: float = 1.0

    def log_scores(self, z):
        z = _as64(z)
        if self.transform is not None:
            z = self.transform(z)
        return full_gaussian_logpdf(z, self.center, self.chol) + self.log_prior[None]


@dataclass(eq=False)
class StudentTCalibrated(CalibratedHead):
    loc: torch.Tensor             # (T, d)
    chol: torch.Tensor            # (T, d, d) Cholesky of the Student-t scale matrix
    dof: torch.Tensor             # (T,)
    log_prior: torch.Tensor
    temperature: float = 1.0

    def log_scores(self, z):
        z = _as64(z)
        T, d = self.loc.shape
        diff = (z[None] - self.loc[:, None]).transpose(1, 2)
        sol = torch.linalg.solve_triangular(self.chol, diff, upper=False)
        maha = (sol ** 2).sum(1).T                                           # (n, T)
        logdet = 2. * torch.log(torch.diagonal(self.chol, dim1=-2, dim2=-1)).sum(-1)
        nu = self.dof
        const = torch.lgamma((nu + d) / 2) - torch.lgamma(nu / 2) - d / 2 * torch.log(nu * math.pi) - logdet / 2
        return const[None] - (nu[None] + d) / 2 * torch.log1p(maha / nu[None]) + self.log_prior[None]


@dataclass(eq=False)
class CosineCalibrated(CalibratedHead):
    center: torch.Tensor          # (T, d) prototypes in the normalised space
    shift: torch.Tensor           # (d,) source mean used for centring
    temperature: float = 1.0

    def log_scores(self, z):
        x = _as64(z) - self.shift
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return -((x[:, None, :] - self.center[None]) ** 2).sum(-1)


# ---------------------------------------------------------------------------------- head families
def _uniform_log_prior(T, device=None):
    return torch.full((T,), -math.log(T), dtype=DTYPE, device=device)


def _require_support(m, name):
    if bool((m <= 0).any()):
        missing = torch.nonzero(m <= 0).flatten().tolist()
        raise MissingClassError(f'{name}: no support for classes {missing}; target prototypes are undefined '
                                f'(use the 0-shot reference head instead)')


class Head:
    """Base class: fit on source, adapt on support. Subclasses set `name`."""
    name = 'head'
    needs_support = True

    def fit(self, z, y, condition, num_classes):
        self.T = int(num_classes)
        self.z_dim = _as64(z).shape[1]
        return self

    def adapt(self, z_support, y_support):
        raise NotImplementedError

    def _support(self, z_support, y_support):
        if z_support is None or len(y_support) == 0:
            s = torch.zeros(self.T, self.z_dim, dtype=DTYPE)
            return s, torch.zeros(self.T, dtype=DTYPE), torch.zeros_like(s)
        return support_means(z_support, y_support, self.T)


class NativeHead(Head):
    """H0: the encoder's own linear classifier. Undefined for new identities (external transfer)."""
    name = 'H0'
    needs_support = False

    def __init__(self, W, b):
        self.W, self.b = _as64(W), _as64(b)

    def fit(self, z, y, condition, num_classes):
        super().fit(z, y, condition, num_classes)
        if self.W.shape[0] != num_classes:
            raise ValueError(f'native head has {self.W.shape[0]} classes, task has {num_classes}: '
                             f'not applicable (report N/A, never 0)')
        return self

    def adapt(self, z_support=None, y_support=None):
        return LinearCalibrated(self.W, self.b)


class _PooledVarianceMixin:
    """Source moments shared by H0S/H1/H2: receiver-equal class means + pooled within-cell variance."""
    stats_cfg = StatsConfig()

    def _fit_moments(self, z, y, condition, num_classes):
        z = _as64(z)
        c, _ = _check_conditions(condition)
        n, u, v = cell_moments(z, y, c, num_classes, int(c.max()) + 1)
        valid = n >= self.stats_cfg.min_cell
        R_t = valid.sum(1).to(DTYPE)
        if bool((R_t == 0).any()):
            raise MissingClassError('class without usable source cells')
        w = valid.to(DTYPE)[..., None]
        self.mu = (u * w).sum(1) / R_t[:, None]
        sig_pool = ((v * w).sum(1) / R_t[:, None]).mean(0)
        self.floor = float(self.stats_cfg.floor_abs + self.stats_cfg.floor_rel * sig_pool.mean())
        self.sigma_pool = sig_pool.clamp_min(self.floor)


class SourceMeanLDA(_PooledVarianceMixin, Head):
    """H0S: source means, pooled diag variance; the information-matched 0-shot reference."""
    name = 'H0S'
    needs_support = False

    def fit(self, z, y, condition, num_classes):
        super().fit(z, y, condition, num_classes)
        self._fit_moments(z, y, condition, num_classes)
        return self

    def adapt(self, z_support=None, y_support=None):
        return DiagGaussianCalibrated(self.mu, self.sigma_pool.expand(self.T, -1), _uniform_log_prior(self.T))


class TargetPrototype(_PooledVarianceMixin, Head):
    """H1: target support mean; metric 'pooled' (diag LDA with source within variance) or 'euclid'."""

    def __init__(self, metric='pooled'):
        assert metric in ('pooled', 'euclid')
        self.metric = metric
        self.name = 'H1' if metric == 'pooled' else 'H1e'

    def fit(self, z, y, condition, num_classes):
        super().fit(z, y, condition, num_classes)
        self._fit_moments(z, y, condition, num_classes)
        return self

    def adapt(self, z_support, y_support):
        s, m, _ = self._support(z_support, y_support)
        _require_support(m, self.name)
        var = self.sigma_pool if self.metric == 'pooled' else torch.ones(self.z_dim, dtype=DTYPE)
        return DiagGaussianCalibrated(s, var.expand(self.T, -1), _uniform_log_prior(self.T))


class ScalarShrink(_PooledVarianceMixin, Head):
    """H2: centre = (1-a) mu + a s with one scalar a shared by all classes.
    alpha: float in [0, 1] (selected on source_selection only) or 'legacy' -> a = m/(m+2) per class."""

    def __init__(self, alpha):
        if alpha != 'legacy' and not (0. <= float(alpha) <= 1.):
            raise ValueError('alpha must be in [0, 1] or "legacy"')
        self.alpha = alpha
        self.name = 'H2fix' if alpha == 'legacy' else f'H2[a={float(alpha):g}]'

    def fit(self, z, y, condition, num_classes):
        super().fit(z, y, condition, num_classes)
        self._fit_moments(z, y, condition, num_classes)
        return self

    def adapt(self, z_support, y_support):
        s, m, _ = self._support(z_support, y_support)
        _require_support(m, self.name)
        a = (m / (m + 2.))[:, None] if self.alpha == 'legacy' else float(self.alpha)
        center = (1. - a) * self.mu + a * s
        return DiagGaussianCalibrated(center, self.sigma_pool.expand(self.T, -1), _uniform_log_prior(self.T))


class LegacyP2EProto(Head):
    """H2L: byte-for-byte the P2E few-shot `proto` rule (scripts/receiver_p2/p2e_fewshot.py with
    p2e_eigenrx.EigenRX): source PCA keeping 99.5% variance (2 <= k <= 32), receiver-equal class means in
    PCA space, packet-pooled within-(class,RX) full covariance shrunk 10% to isotropic, decision by
    shared-covariance Gaussian, centres mu + m/(m+2) (gbar - mu). Cholesky replaces the explicit inverse
    (identical argmax up to rounding; checked against EigenRX in tests)."""
    name = 'H2L'

    def __init__(self, var=.995, cap=32, shrink=.1):
        self.var, self.cap, self.shrink = var, cap, shrink

    def fit(self, z, y, condition, num_classes):
        super().fit(z, y, condition, num_classes)
        hs = _as64(z)
        y = _as_long(y)
        c, _ = _check_conditions(condition)
        self.m0 = hs.mean(0)
        X = hs - self.m0
        _, sv, Vt = torch.linalg.svd(X, full_matrices=False)
        ev = sv ** 2
        k = int(torch.searchsorted(torch.cumsum(ev, 0) / ev.sum(), torch.tensor(self.var, dtype=DTYPE))) + 1
        k = max(2, min(self.cap, k))
        self.V = Vt[:k].T
        g = X @ self.V
        R = int(c.max()) + 1
        n, u, _ = cell_moments(g, y, c, self.T, R)
        if bool((n == 0).any()):
            raise MissingClassError('LegacyP2EProto needs every (class, RX) source cell (as in EigenRX)')
        self.mu = u.mean(1)
        res = g - u[y, c]
        S = res.T @ res / len(res)
        S = (1 - self.shrink) * S + self.shrink * torch.trace(S) / k * torch.eye(k, dtype=DTYPE)
        self.L, _ = safe_cholesky(S)
        return self

    def proj(self, z):
        return (_as64(z) - self.m0) @ self.V

    def adapt(self, z_support, y_support):
        y = _as_long(y_support)
        m = torch.bincount(y, minlength=self.T).to(DTYPE)
        _require_support(m, self.name)
        g = self.proj(z_support)
        gbar = torch.zeros(self.T, g.shape[1], dtype=DTYPE).index_add(0, y, g) / m[:, None]
        means = self.mu + (m / (m + 2.))[:, None] * (gbar - self.mu)
        return FullGaussianCalibrated(means, self.L, _uniform_log_prior(self.T), transform=self.proj)


class GaussianShrink(Head):
    """H3: ordinary Gaussian covariance shrinkage with NO between/within decomposition.
        M[t]       variance of ALL source features of class t (includes between-RX spread)
        a_M[t]     = (N_t - 1)/(N_t - 1 + d);  M_shrunk = a_M M + (1 - a_M) mean_t M
        b[t]       = (m_t - 1)/(m_t - 1 + kappa_b)   (0 when m_t = 1)
        W[t]       = (1 - b) M_shrunk + b * support sample variance
        centre     = support mean;  Q = (1 + 1/m) W   (plug-in predictive for an unknown mean)
    Source records are receiver-balanced before M is computed (equal weight per RX within each class)."""
    name = 'H3'

    def __init__(self, kappa_b=8.0, cfg: StatsConfig = StatsConfig()):
        self.kappa_b, self.cfg = kappa_b, cfg

    def fit(self, z, y, condition, num_classes):
        super().fit(z, y, condition, num_classes)
        z = _as64(z)
        y = _as_long(y)
        c, _ = _check_conditions(condition)
        R = int(c.max()) + 1
        n, u, v = cell_moments(z, y, c, self.T, R)
        valid = n >= 1
        w = valid.to(DTYPE)
        Rt = w.sum(1)
        if bool((Rt == 0).any()):
            raise MissingClassError('class without source records')
        # receiver-balanced class mixture: mean of cell means; variance = mean within + spread of means
        mu = (u * w[..., None]).sum(1) / Rt[:, None]
        within = (v * ((n - 1).clamp_min(0) / n.clamp_min(1))[..., None] * w[..., None]).sum(1) / Rt[:, None]
        between = (((u - mu[:, None]) ** 2) * w[..., None]).sum(1) / Rt[:, None]
        M = within + between
        N = torch.bincount(y, minlength=self.T).to(DTYPE)
        a = ((N - 1) / (N - 1 + self.z_dim))[:, None]
        floor = float(self.cfg.floor_abs + self.cfg.floor_rel * M.mean())
        self.M = (a * M + (1 - a) * M.mean(0)).clamp_min(floor)
        self.mu, self.floor = mu, floor
        return self

    def adapt(self, z_support, y_support):
        s, m, sv = self._support(z_support, y_support)
        _require_support(m, self.name)
        b = ((m - 1) / (m - 1 + self.kappa_b))[:, None]
        W = ((1 - b) * self.M + b * sv).clamp_min(self.floor)
        return DiagGaussianCalibrated(s, (1 + 1 / m)[:, None] * W, _uniform_log_prior(self.T))


class PosteriorPredictive(Head):
    """H4: receiver/noise decomposition + diagonal Gaussian posterior predictive (statistics.py).
    Classes without support fall back to their prior (theta = mu, Q = Sigma + A); m = 0 is legal."""
    needs_support = False

    def __init__(self, cfg: StatsConfig = StatsConfig(), stats: SourceStats | None = None):
        self.cfg = cfg
        self.stats = stats
        self.name = cfg.tag() if stats is None else 'H4-transfer'

    def fit(self, z, y, condition, num_classes):
        super().fit(z, y, condition, num_classes)
        if self.stats is None:
            self.stats = fit_source_stats(z, y, condition, num_classes, self.cfg)
        return self

    @classmethod
    def transfer(cls, pool: SourceStats, z_reg, y_reg, num_classes):
        """OSU single-source-RX variant (TRANSFER_POOLED_PRIOR), see statistics.transfer_prior."""
        st = transfer_prior(pool, z_reg, y_reg, num_classes)
        h = cls(stats=st)
        h.T, h.z_dim = num_classes, st.dim
        return h

    def adapt(self, z_support, y_support):
        s, m, _ = self._support(z_support, y_support)
        theta, Q = posterior_predictive(self.stats, s, m)
        return DiagGaussianCalibrated(theta, Q, _uniform_log_prior(self.T))


class SimpleShotCL2N(Head):
    """H5a: SimpleShot CL2N (Wang et al. 2019) — subtract the SOURCE feature mean, L2-normalise, nearest
    support centroid (Euclidean in the normalised space)."""
    name = 'H5a'

    def fit(self, z, y, condition, num_classes):
        super().fit(z, y, condition, num_classes)
        self.shift = _as64(z).mean(0)
        return self

    def adapt(self, z_support, y_support):
        x = _as64(z_support) - self.shift
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        s, m, _ = support_means(x, y_support, self.T)
        _require_support(m, self.name)
        return CosineCalibrated(s, self.shift)


class SimpleCNAPSFrozen(Head):
    """H5b: the Simple CNAPS Mahalanobis classifier (Bateni et al. 2020, eq. 4-5) on frozen features:
        Q_k = lam_k S_k + (1 - lam_k) S_task + beta I,  lam_k = n_k / (n_k + 1),  beta = 1
    S_k / S_task are the class / whole-support covariances of the TARGET support only (no source
    statistics, as in the original); score = -0.5 Mahalanobis^2 to the support mean (no log det, as in
    the original). The FiLM-adapted feature extractor of Simple CNAPS is NOT reproduced."""
    name = 'H5b'
    needs_support = True

    def __init__(self, beta=1.0):
        self.beta = beta

    def fit(self, z, y, condition, num_classes):
        """The original classifier needs no source data; fit only records the task's class count."""
        self.T, self.z_dim = int(num_classes), _as64(z).shape[1]
        return self

    def adapt(self, z_support, y_support):
        z = _as64(z_support)
        y = _as_long(y_support)
        s, m, _ = support_means(z, y, self.T)
        _require_support(m, self.name)
        d = z.shape[1]
        eye = torch.eye(d, dtype=DTYPE)
        zt = z - z.mean(0)
        S_task = zt.T @ zt / max(len(z) - 1, 1)
        chols = []
        for t in range(self.T):
            zk = z[y == t] - s[t]
            S_k = zk.T @ zk / max(int(m[t]) - 1, 1) if int(m[t]) > 1 else torch.zeros(d, d, dtype=DTYPE)
            lam = float(m[t] / (m[t] + 1))
            Q = lam * S_k + (1 - lam) * S_task + self.beta * eye
            chols.append(safe_cholesky(0.5 * (Q + Q.T))[0])
        L = torch.stack(chols)
        return _NoLogdetGaussian(s, L, torch.zeros(self.T, dtype=DTYPE))


@dataclass(eq=False)
class _NoLogdetGaussian(CalibratedHead):
    center: torch.Tensor
    chol: torch.Tensor
    log_prior: torch.Tensor
    temperature: float = 1.0

    def log_scores(self, z):
        z = _as64(z)
        diff = (z[None] - self.center[:, None]).transpose(1, 2)
        sol = torch.linalg.solve_triangular(self.chol, diff, upper=False)
        return -0.5 * (sol ** 2).sum(1).T + self.log_prior[None]


class NIWPredictive(Head):
    """H5c: MetaQDA-style Bayesian QDA (Zhang et al. 2021) with a FIXED source-moment prior.
    Prior per class t: NIW(m0 = mu[t], kappa0, nu0 = d + 2, Psi0 = (nu0 - d - 1) S_within) where S_within is
    the receiver-equal pooled within-(class,RX) full covariance (PSD-projected, 10% shrunk to isotropic).
    Posterior predictive: multivariate Student-t. kappa0 comes from a registered grid selected on
    source_selection; MetaQDA meta-learns its prior, which is NOT done here."""
    needs_support = False

    def __init__(self, kappa0=1.0, shrink=.1):
        self.kappa0, self.shrink = float(kappa0), shrink
        self.name = f'H5c[k0={self.kappa0:g}]'

    def fit(self, z, y, condition, num_classes):
        super().fit(z, y, condition, num_classes)
        z = _as64(z)
        y = _as_long(y)
        c, _ = _check_conditions(condition)
        R = int(c.max()) + 1
        n, u, _ = cell_moments(z, y, c, self.T, R)
        valid = n >= 2
        if bool((valid.sum(1) == 0).any()):
            raise MissingClassError('class without usable source cells')
        d = z.shape[1]
        S = torch.zeros(d, d, dtype=DTYPE)
        cnt = 0
        for t in range(self.T):
            for r in range(R):
                if bool(valid[t, r]):
                    x = z[(y == t) & (c == r)] - u[t, r]
                    S = S + x.T @ x / (len(x) - 1)
                    cnt += 1
        S = psd_project(S / cnt)
        S = (1 - self.shrink) * S + self.shrink * torch.trace(S) / d * torch.eye(d, dtype=DTYPE)
        w = valid.to(DTYPE)[..., None]
        self.mu = (u * w).sum(1) / valid.sum(1).to(DTYPE)[:, None]
        self.nu0 = d + 2.
        self.Psi0 = (self.nu0 - d - 1) * S
        return self

    def adapt(self, z_support, y_support):
        z = _as64(z_support) if z_support is not None else torch.zeros(0, self.z_dim, dtype=DTYPE)
        y = _as_long(y_support) if y_support is not None else torch.zeros(0, dtype=torch.long)
        d = self.z_dim
        locs, chols, dofs = [], [], []
        for t in range(self.T):
            x = z[y == t]
            n = len(x)
            k0, nu0, m0 = self.kappa0, self.nu0, self.mu[t]
            xbar = x.mean(0) if n else m0
            Sx = (x - xbar).T @ (x - xbar) if n > 1 else torch.zeros(d, d, dtype=DTYPE)
            kn, nun = k0 + n, nu0 + n
            mn = (k0 * m0 + n * xbar) / kn
            dm = (xbar - m0)[:, None]
            Psin = self.Psi0 + Sx + (k0 * n / kn) * (dm @ dm.T)
            dof = nun - d + 1
            scale = Psin * (kn + 1) / (kn * dof)
            locs.append(mn)
            chols.append(safe_cholesky(0.5 * (scale + scale.T))[0])
            dofs.append(dof)
        return StudentTCalibrated(torch.stack(locs), torch.stack(chols), torch.tensor(dofs, dtype=DTYPE),
                                  _uniform_log_prior(self.T))


def default_heads(native=None, alpha_grid=(0., .25, .5, .75, 1.), kappa0_grid=(1., 4., 16.)):
    """The registered Stage-A head set. Parametrised families are expanded over their grid; the grid
    member actually reported is chosen on source_selection RX only (evaluator.select_parameters)."""
    heads = []
    if native is not None:
        heads.append(NativeHead(*native))
    heads += [SourceMeanLDA(), TargetPrototype('pooled'), TargetPrototype('euclid'), ScalarShrink('legacy'),
              LegacyP2EProto()]
    heads += [ScalarShrink(a) for a in alpha_grid]
    heads += [GaussianShrink(), PosteriorPredictive(StatsConfig()),
              PosteriorPredictive(StatsConfig(finite_center=False)),
              PosteriorPredictive(StatsConfig(pooled=True)),
              SimpleShotCL2N(), SimpleCNAPSFrozen()]
    heads += [NIWPredictive(k) for k in kappa0_grid]
    return heads


FAMILIES = {'H2': lambda name: name.startswith('H2['), 'H5c': lambda name: name.startswith('H5c[')}
