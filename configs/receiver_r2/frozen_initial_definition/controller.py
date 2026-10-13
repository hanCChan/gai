"""Minimal source-only finite Adam scaling policy; no controlled fitting in R2-A.

This policy is frozen before the aggregate C2 analysis. Its noise tolerances
are recomputed from C1 paired-packet estimation errors only, never from C2.
"""
import math

SCALES=(1.,.5,.25,0.)
NOISE_MULTIPLIER=2.


def choose_scale(c1_by_scale,multi_rx=True):
    """Input: each nonzero scale -> per-RX (delta, paired_SE, resolution, n).

    First guard: mean risk <= mean2SE tolerance. Multi-RX adds a harm-budget
    guard: each RX's excess over its2SE noise margin may not exceed the mean
    measured improvement plus the mean noise allowance. Positive individual
    RX changes are permitted. All coefficients are fixed; no fitted threshold.
    """
    for scale in SCALES[:-1]:
        cells=c1_by_scale.get(scale,[])
        if not cells or any(c['n']<2 or not all(math.isfinite(c[k]) for k in ['delta','paired_SE','resolution']) for c in cells):continue
        means=[c['delta'] for c in cells];noise=[NOISE_MULTIPLIER*c['paired_SE']+c['resolution'] for c in cells]
        mean=sum(means)/len(cells);mean_noise=sum(noise)/len(cells)
        if mean>mean_noise:continue
        budget=max(0.,-mean)+mean_noise
        if multi_rx and max(d-e for d,e in zip(means,noise))>budget:continue
        return scale
    return 0.
