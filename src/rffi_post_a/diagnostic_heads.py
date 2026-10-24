"""Fixed D1 diagnostic slots (plan §6.3-6.5). Each slot is a pure function of a SourceBundle; it is
adapted with the support set only and returns a scorer that sees query FEATURES only.

    variant = SLOT_BUILDERS[name](bundle)
    scorer  = adapt_variant(variant, z_support, y_support)     # raises NotApplicable when undefined
    terms   = score_terms(scorer, z_query)                     # {'score', 'maha', 'logdet'}

Gaussian slots use a fixed uniform class prior and keep the complete log-density unless their
definition removes a term (M_NOLOGDET, which is a scoring diagnostic, not a posterior).
No slot is created or changed from scores; the list is fixed in configs/rffi_post_a/diagnostic_matrix.json.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from rffi_calibration.statistics import (DTYPE, SourceStats, _as64, _as_long, gaussian_condition_full,
                                         posterior_predictive, safe_cholesky, support_means)

from .diagnostic_stats import NotApplicable, SourceBundle

LOG2PI = math.log(2 * math.pi)


# ------------------------------------------------------------------------------------------- scorers
@dataclass(eq=False)
class DiagScorer:
    center: torch.Tensor     # (T, d)
    var: torch.Tensor        # (T, d)
    use_logdet: bool = True
    transform: object = None

    def terms(self, z):
        z = _as64(z)
        if self.transform is not None:
            z = self.transform(z)
        iv = 1. / self.var
        maha = (z ** 2) @ iv.T - 2. * z @ (self.center * iv).T + ((self.center ** 2) * iv).sum(1)[None]
        logdet = torch.log(self.var).sum(1)
        T, d = self.center.shape
        score = -0.5 * maha - math.log(T)
        if self.use_logdet:
            score = score - 0.5 * (logdet[None] + d * LOG2PI)
        return dict(score=score, maha=maha, logdet=logdet)


@dataclass(eq=False)
class FullScorer:
    center: torch.Tensor     # (T, k)
    chol: torch.Tensor       # (T, k, k)
    jitter: list
    transform: object = None
    use_logdet: bool = True

    def terms(self, z):
        z = _as64(z)
        if self.transform is not None:
            z = self.transform(z)
        T, k = self.center.shape
        diff = (z[None] - self.center[:, None]).transpose(1, 2)
        sol = torch.linalg.solve_triangular(self.chol, diff, upper=False)
        maha = (sol ** 2).sum(1).T
        logdet = 2. * torch.log(torch.diagonal(self.chol, dim1=-2, dim2=-1)).sum(-1)
        score = -0.5 * maha - math.log(T)
        if self.use_logdet:
            score = score - 0.5 * (logdet[None] + k * LOG2PI)
        return dict(score=score, maha=maha, logdet=logdet)


def score_terms(scorer, z_query):
    """Scores, distance terms and volume terms; no query IDs/labels."""
    if not torch.is_tensor(z_query) or z_query.dtype not in (torch.float32, torch.float64):
        raise TypeError('score_terms accepts a float feature tensor only')
    t = scorer.terms(z_query)
    if not bool(torch.isfinite(t['score']).all()):
        raise FloatingPointError('non-finite diagnostic score')
    return t


# ------------------------------------------------------------------------------------------ helpers
def _support(b, z_support, y_support):
    if z_support is None or len(y_support) == 0:
        return torch.zeros(b.T, b.d, dtype=DTYPE), torch.zeros(b.T, dtype=DTYPE)
    s, m, _ = support_means(z_support, y_support, b.T)
    return s, m


def _require(m, name):
    if bool((m <= 0).any()):
        raise NotApplicable(f'{name}: support-dependent slot is undefined without support for every class')


def _stats(b: SourceBundle, K, sigma, omega=None):
    """SourceStats with A = K + K/R + Omega (Omega fixed to the class sampling term, plan §6.3)."""
    om = b.omega if omega is None else omega
    A = K + K / b.n_rx[:, None] + om
    return SourceStats(mu=b.mu, sigma=sigma, A=A, sigma_pool=b.sigma_pool, floor=b.floor, mode='diagnostic')


def _posterior(b, K, sigma, s, m):
    return posterior_predictive(_stats(b, K, sigma), s, m)


# ---------------------------------------------------------------------------------- slot families
class Variant:
    needs_support = False

    def __init__(self, name, b, fn, needs_support=False):
        self.name, self.b, self.fn, self.needs_support = name, b, fn, needs_support

    def adapt(self, z_support, y_support):
        s, m = _support(self.b, z_support, y_support)
        if self.needs_support:
            _require(m, self.name)
        return self.fn(self.b, s, m, z_support, y_support)


def adapt_variant(variant, z_support, y_support):
    """Returns a scorer that accepts query feature tensors only."""
    return variant.adapt(z_support, y_support)


def _Kc(b):
    return b.K


def _Kp(b):
    return b.K_pool.expand(b.T, b.d)


def _Sc(b):
    return b.sigma_class_shrunk


def _Sp(b):
    return b.sigma_pool.expand(b.T, b.d)


def _block1(Kf, Sf):
    def fn(b, s, m, *_):
        theta, Q = _posterior(b, Kf(b), Sf(b), s, m)
        return DiagScorer(theta, Q)
    return fn


def _block2(center, cov, logdet=True):
    """center in {'s','theta'}, cov in {'pool','Q','sigma'}; theta/Q/Sigma from D_CC."""
    def fn(b, s, m, *_):
        theta, Q = _posterior(b, _Kc(b), _Sc(b), s, m)
        c = s if center == 's' else theta
        v = {'pool': _Sp(b), 'Q': Q, 'sigma': _Sc(b)}[cov]
        return DiagScorer(c, v, use_logdet=logdet)
    return fn


def _proj(b):
    return lambda z: (_as64(z) - b.pca_m0) @ b.pca_V


def _block3(coord, cov, centre):
    """coord raw|pca, cov diag|full, centre S (support mean) | L (legacy m/(m+2)) | B (pooled Bayes)."""
    def fn(b, s, m, z_support, y_support):
        if b.raw_S is None:
            raise NotApplicable(b.extra.get('shared_block_not_applicable', 'shared block unavailable'))
        if coord == 'raw':
            tf, mu, S = None, b.raw_mu, b.raw_S
            sc = s
        else:
            tf, mu, S = _proj(b), b.pca_mu, b.pca_S
            sc = support_means(tf(z_support), y_support, b.T)[0] if (z_support is not None and len(y_support)) \
                else torch.zeros_like(mu)
        k = mu.shape[1]
        if centre == 'S':
            center = sc
        elif centre == 'L':
            center = mu + (m / (m + 2.))[:, None] * (sc - mu)
        if centre in ('S', 'L'):
            if cov == 'diag':
                return DiagScorer(center, torch.diagonal(S).expand(b.T, k), transform=tf)
            L, j = safe_cholesky(S)
            return FullScorer(center, L.expand(b.T, k, k), [j], transform=tf)
        pb = b.pooled_bayes
        if pb is None or 'K0' not in pb:
            raise NotApplicable(f'pooled Bayesian slot: {pb.get("not_applicable") if pb else "no PCA block"}')
        A0, S0 = pb['A0'], pb['Sigma0']
        if cov == 'diag':
            a, sg = torch.diagonal(A0).expand(b.T, k), torch.diagonal(S0).expand(b.T, k)
            st = SourceStats(mu=mu, sigma=sg, A=a, sigma_pool=torch.diagonal(S0), floor=0., mode='diagnostic')
            theta, Q = posterior_predictive(st, sc, m)
            return DiagScorer(theta, Q, transform=tf)
        centers, chols, jit = [], [], []
        for t in range(b.T):
            th, Q = gaussian_condition_full(mu[t], A0, S0, sc[t], float(m[t]))
            L, j = safe_cholesky(0.5 * (Q + Q.T))
            centers.append(th)
            chols.append(L)
            jit.append(j)
        return FullScorer(torch.stack(centers), torch.stack(chols), jit, transform=tf)
    return fn


SLOT_DEFS = {
    # block 1: K / Sigma 2x2 with common sampling-U
    'D_CC': (_block1(_Kc, _Sc), False),
    'D_PC': (_block1(_Kp, _Sc), False),
    'D_CP': (_block1(_Kc, _Sp), False),
    'D_PP': (_block1(_Kp, _Sp), False),
    # block 2: centre / density
    'M_SP': (_block2('s', 'pool'), True),
    'M_TP': (_block2('theta', 'pool'), False),
    'M_SQ': (_block2('s', 'Q'), True),
    'M_TQ': (_block2('theta', 'Q'), False),
    'M_TSIG': (_block2('theta', 'sigma'), False),
    'M_NOLOGDET': (_block2('theta', 'Q', logdet=False), False),
    # block 3: coordinates / correlation / shared covariance
    'C_RD_S': (_block3('raw', 'diag', 'S'), True), 'C_RD_L': (_block3('raw', 'diag', 'L'), True),
    'C_RF_S': (_block3('raw', 'full', 'S'), True), 'C_RF_L': (_block3('raw', 'full', 'L'), True),
    'C_PD_S': (_block3('pca', 'diag', 'S'), True), 'C_PD_L': (_block3('pca', 'diag', 'L'), True),
    'C_PF_S': (_block3('pca', 'full', 'S'), True), 'C_PF_L': (_block3('pca', 'full', 'L'), True),
    'C_PD_B': (_block3('pca', 'diag', 'B'), False), 'C_PF_B': (_block3('pca', 'full', 'B'), False),
}


def diagnostic_variants(source_bundle, matrix):
    """Registered fixed variants; no score-driven candidate creation."""
    names = [s['slot'] for s in matrix['slots']]
    unknown = set(names) - set(SLOT_DEFS)
    if unknown:
        raise KeyError(f'unregistered diagnostic slots {sorted(unknown)}')
    if len(names) != len(set(names)):
        raise ValueError('duplicate slot names in the diagnostic matrix')
    return {n: Variant(n, source_bundle, *SLOT_DEFS[n]) for n in names}
