# D phase protocol (frozen before any D model output was computed) — 2026-10-20

Scope approved by the reviewer: D1, D2, D2+ only. Inference on frozen R0/R1 checkpoints; the only
trained objects are the documented source-only refit heads (run_l0 recipe). No backbone training,
no E0/E1, no CRR fits. WiSig final 8 RX, date 3 and the equalized variant stay sealed; Shen
development/locked RX boundaries unchanged; Shen raw IQ not read (Strict CIS cache only).
Budget cap: 0.5 reserved GPUh for all of D (stop and ask if the projection exceeds it).

## Pre-run code finding that changes the D1 question

The L0 full run (`scripts/interaction_l0/run_l0.py` sha 9b4f88…, `physics.py` sha 5c42d2…) did **not**
use the full-band circular filter quoted in the review (that was the v0 `link_l0/physics.py`). r1
`reshape_tones` multiplies the 20-point DFT bins of the descriptor segment only (samples 20..179),
leaves samples 0..19 and 180..255 unfiltered and rescales the packet. Its "tone error 0.0" holds by
construction (applied in the measurement domain); the waveform is piecewise and is not the output
of any LTI receiver. A true LTI filter on real packets realises the 12-tone change with ~0.2–0.34
RMS error, concentrated on notched tones (≈⅓ of tones sit >13 dB below the packet mean; the STF ends
at ~180–190 samples). D1 therefore compares four implementations of the same requested change:
`r1_segment` (L0), `v0_circular` (critiqued code), `fir_zero` and `fir_min` (whole-packet linear
convolution, zero / minimum phase). `fir_zero` is the primary physical arm.

## D1 — WiSig receiver-response interventions (21 frozen models: 18 official + 3 resnet)

Pools. Templates: source 14 RX, dates 0–1, positions 0–119 (fit pool). Evaluation: same RX/dates,
positions 160–199 (the L0 `source` pool; positions 120–159 were checkpoint selection). Internal
(development, already viewed) RX, positions 0–199, dates 0–1 and 2: only for diagnostics R7.
One partner RX / partner TX / random draw per packet, seed 20261019, identical for all models.

Arms (12-tone natural-log-power shape deltas, centred).
- `L0_*`: exact reproduction via r1 `wisig_physics` (in-sample templates) — sanity check against L0.
- RX swap `b[r']-b[r]` held-out, in all four implementations. Link swap `i[t,r']-i[t,r]`, TX swap
  `a[t']-a[t]` (r1 and fir), cell swap `M[t,r']-M[t,r]`, cell swap + RX CFO difference.
- Matched-norm controls (per packet ‖·‖ = ‖b[r']-b[r]‖, fir_zero): link/TX direction, 4 isotropic
  random, 4 random with the covariance of real source-RX differences, smooth tilt ±, tone-shuffled
  RX swap, negated RX swap.
- Natural-size same-RX nulls (fir_zero): split-half estimate difference of the same RX; same RX
  other date `b_r^{d'}-b_r^{d}`; same cell other date `M^{d'}[t,r]-M^{d}[t,r]`.
- Dose: α ∈ {0.25,0.5,1,1.5,2} × ‖b[r']-b[r]‖ for RX, random-cov, tilt+, link directions.

Statistic. F(arm) = fraction of eval predictions changed vs the null arm; also accuracy drop,
true-class probability, feature change ‖Δf‖/‖f‖. Aggregation: per seed = median over the 6
official methods; overall = median over 18 official models; resnet reported separately; 95%
bootstrap intervals over models (2000 draws) and per-RX breakdowns are reported, not gated.

Rules (applied in this order; thresholds frozen here).
- R1 credibility (model-free, held-out `fir_zero` RX swap): nonfinite = 0; median periodicity
  after ≥ 0.95; median |CFO side effect| ≤ 500 Hz; strong-tone realisation RMS ≤ 0.2 and
  realised/requested slope ≥ 0.8; fraction of packets within the source range (nearest real fit
  cell of the same TX within the fit pool's 95th-percentile packet-to-cell distance) ≥ 0.8.
  Fail → `INVALID_INTERVENTION`.
- R2 artefact: `ARTEFACT` if overall F(fir_zero RX) < 0.5 × F(r1_segment RX) (held-out templates).
- R3 original exploratory ratios, now on held-out `fir_zero`: F(RX) ≥ 2 × max(F split-half,
  F same-RX-other-date, F same-cell-other-date) and F(RX) ≥ 1.5 × max(F tilt+, F tilt−).
- R4 direction specificity (decides CRR's prior): Q = F(RX) / median_k F(random-cov_k).
  `RX_SPECIFIC` if overall Q ≥ 1.5 and per-seed Q ≥ 1.5 in 3/3 seeds; `GENERIC` if overall
  Q ≤ 1.2; otherwise `PARTIAL`. Q_iso, Q_tilt, Q_shuffle, Q_negated reported alongside.
- R5 seed robustness: R3/R4 verdicts are stated per seed; a verdict not holding in 3/3 seeds is
  reported as seed-dependent.
- R6–R8 diagnostics (no gate): dose curves (norm at F=0.2 per direction); Spearman of F(RX),
  F(random-cov), F(tilt) with internal unseen-date accuracy across 18 official models, overall
  and within seed; confusion-structure cosine between each arm and the internal unseen-RX
  confusion; transplant recovery (cell swap, cell swap + CFO vs RX swap).

Decision for CRR. `INVALID_INTERVENTION`/`ARTEFACT` → no CRR training, fix the intervention.
Otherwise the V0–V4 WiSig plan may be proposed (separate approval), with prior V2 > V1 under
`RX_SPECIFIC`, V1 ≈ V2 under `GENERIC` (then a strong random spectral randomisation baseline is
the priority), no prior under `PARTIAL`.

## D2 — Shen frozen representations (30 models: 2 folds × 5 methods × 3 seeds)

Checks: checkpoint status hash; checkpoint scaler equals the scaler recomputed from the fold's
training keys (first 160 per TX); Strict CIS file hashes equal the registration's input assets.
Fixes vs L0: Shen jobs only; the source selection pool (positions 160–199 per TX) is split into
two halves by within-cell order; erasure directions estimated on one half, source identity and
energies evaluated on the other, both ways, averaged. Refit heads on the training packets (first
160), recipe unchanged. Reported per model: TX/RX/link cross-fit energies and shares (source;
outer RX as diagnostic), effective rank and condition number, link/RX/cond/random erasure
(k = 4, 8, 16) with frozen and refit heads, feature change, per-outer-RX accuracy, per-TX recall
damage; paired (fold, seed) differences SWAD−CE, PCGrad−CE. No WiSig physical CFO/tone criteria.
No verdict thresholds: D2 is descriptive and feeds the report.

## D2+ — Strict CIS input atlas (model-free + saved predictions)

Rows by STFT bin (7.8125 kHz): in-band |bin| ≤ 8 (17 rows, inside the |f| ≤ 62.5 kHz strict
filter), transition 9–12 (8 rows), out-of-band ≥ 13 (27 rows). Per band and fold: cross-fit
TX/RX/link energies. With source-only grand/TX means (leave-one-RX-out for source RX):
d_RX(r) = ‖b_r‖ / median source ‖b‖; d_class(r) = ‖i_{·,r}‖_F / ‖a‖_F; pair separation ratios for
the pairs SWAD confuses at Pluto. Join with saved R0/R1 predictions: CE/SWAD accuracy, Δ_r,
per-TX recall, pair confusions. Spearman over the 10 RX (each outer once; exact permutation p,
exploratory, n = 10). Counterexamples: outer RX with d_RX ≥ d_RX(Pluto) and Δ_r ≥ 0. Hardware
family recorded (Pluto, B200mini, B210: AD936x; RTL: R820T/RTL2832U). No conclusion about
"frequency-response extrapolation" in general from one RX.
