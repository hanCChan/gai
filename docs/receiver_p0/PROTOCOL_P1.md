# P1 protocol: P0 review fixes + mechanism confirmation (frozen 2026-10-21, before any P1 search fit)

Base: c79a59c (P0). Branch `experiment/p0-pair-margin`. Shen Strict CIS only; WiSig untouched this round
(no final 8 RX / date 3 / equalized); Shen outer folds are repeatedly-developed data, not blind tests.

## 0. Review findings adopted (code-verified)

1. P0 `pair` branch minimised logsumexp_r(batch-min margin): sign error, gamma unused
   (test_p1_losses.py: larger margins -> larger loss). Its λ=1 collapse and λ=0.1 "gain" are not evidence
   about margin protection. Kept only as `pair_p0_buggy` for replay.
2. P0 `drift` = softplus(γ+Δ), Σ_r Δ = 0 on balanced batches -> softplus(γ) + κ·Var_r(Δ) + O(Δ³)
   (test verifies to <5%); it penalises the margin-ENLARGING direction and is lowered by shrinking
   ||w_t−w_u||. Renamed: decision-projection dispersion regulariser. Measured gradient ≈5–10% of CE.
3. P0-C risk averaged 5 methods while select_pairs used CE. Re-analysis (p1_stats.py, TX-relabel
   permutation keeping shared-TX dependence): predictability holds (all p=5e-5), but R_conf ≥ R_q
   (receiver level R_q better on 2/10 RX CE-only, 1/10 all-5). Margin risk has no shown advantage as a
   selector; Rconf and Rq sets coincide (fold 1 identical, fold 0 one pair differs) so no Rconf training arm.
4. "Collapse count" in P0 = number of (outer RX × P pair) cells with conf>0.3, not number of pairs.
5. Buggy pair λ=0.1 still gave +2.18 pp on CV1 -> the CV1 gain may be generic regularisation; a
   label-smoothing control (ls, ε=0.1) is mandatory.

## 1. Arms (same backbone/data/seeds/schedule/optimizer/epochs/checkpoint rule)

| arm | loss | question |
|---|---|---|
| A0 ce | CE (P0 anchors, bitwise = frozen R0/R1) | baseline |
| A6 ls | CE with label smoothing 0.1 | generic confidence regulariser control |
| A1 drift | CE + λ·drift_v0 (P0 code, bitwise replay verified) | does the P0 gain replicate under source-only λ |
| A2 var | CE + λ·κ(γ)·mean Var_r Δ | is drift just its 2nd-order variance term |
| A3 margin | CE + λ·mean softplus((γ_m−m)/τ_m), m=(ℓ_t−ℓ_u)/‖w_t−w_u‖ | plain hard-pair margin |
| A4 tail | CE + λ·τ_r·log mean_r exp(L_r/τ_r), τ_r=0.5 | worst-source-RX aggregation beyond A3 |
| A5 tail_drift | A4(λ*) + drift(λ*) | complementarity |
| set control | drift(λ*), tail(λ*) on a random 9-pair set | is pair selection needed |

γ_m, τ_m: median / IQR·0.7413 of normalized margins of the CE checkpoints on source packets
(sets_p1.json; formal: 5 source RX; search: 4 inner RX). γ_drift=0.5, τ_r=0.5 fixed a priori.
Pair sets frozen in sets_p1.json (source only; search uses Rq_inner excluding the inner validation RX).

## 2. λ selection (source only)

Search stage = R0/R1 inner split: train inner_train RX (first160), validate inner_validation RX
(all packets; fold 0 rtl_6, fold 1 pluto, both source RX of their fold), seed 20261016, 20 epochs.
λ ∈ {0.1, 0.3, 1.0} for drift/var/margin/tail; ce and ls one run each. Rule: per method, λ with minimum
held-RX CE (tie -> smaller λ); formal uses that λ even if it does not beat ce on held-RX CE (reported).
Outer RX never read in search.

## 3. Formal and metrics

2 folds × seeds {20261013,14,15}. Outer evaluation only after all formal checkpoints exist.
Primary: RX-equal outer accuracy. Secondary: RX-equal macro-F1, worst outer RX accuracy, Pluto,
P-set (Rq formal) mean outer confusion and mean/q10 normalized margin, non-P mean confusion,
per-RX×TX recall, all 90 ordered-pair confusions, gradient-norm/‖w_tu‖ diagnostics, seconds per fit.

"X beats Y": seed-mean Δ > 0 in BOTH folds and ≥ 5/6 paired (fold, seed) Δ > 0 (primary metric).

## 4. Decision (pre-registered)

- GO (method line, tail or tail_drift as contribution): it beats ce AND ls AND margin AND its random-set
  version; P-set confusion lower than ce in both folds; non-P confusion not up > 0.005.
- MODIFY: some auxiliary arm beats ce but fails against ls or against its simpler counterpart
  (tail≈margin, drift≈var, Rq≈rand) -> keep the simplest effective one, no module claim for the
  redundant part, novelty reassessed before any new training.
- STOP (method line): no auxiliary arm beats both ce and ls -> the P0 CV1 gain is generic
  regularisation; paper framing becomes diagnosis + predictability + boundary.
- drift ≡ var declared if |seed-mean Δ| < 0.5 pp in both folds without a consistent sign.
Whatever the outcome: Pluto reported separately; no claim of information-theoretic non-identifiability.

## 5. Budget / stop

≤ 28 search + ≤ 48 formal fits, cap 4.5 GPUh. No added seeds, no λ changes after outer evaluation.
After the report: stop; WiSig transfer only as a written proposal.
