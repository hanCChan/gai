# P2 protocol: RX-nullspace exposure of class-mean displacements (frozen 2026-10-21, before any P2 fit)

Base: 8461a5a (P1). Branch `experiment/p2-rx-null`. Shen Strict CIS only; WiSig untouched this round.
Shen outer folds are repeatedly-developed data, not blind tests. This protocol, the loss code,
its unit tests and the gate script are committed **before any P2 training fit**.

## 0. Mechanism adopted from P0/P1/D (evidence, not hypothesis)

1. Unseen-RX failure is driven by **class-mean displacements along decision directions**:
   refit-head on frozen features reproduces the Pluto collapse (P0, encoder-level geometry);
   source-RX risk statistics predict outer pair failures (rho 0.72–0.73, TX-permutation p=5e-5,
   P1 fair re-analysis); the sensitivity is GENERIC/spectral, not RX-direction-specific (D phase).
2. Margin-type losses (P1 A3/A4) raised the protected margins but moved errors to non-P pairs and
   lost accuracy: in the rank-16 feature space (effective rank ~3.5 of the 512-D projection),
   pushing selected pairs' margins apart *re-allocates* limited discriminative capacity.
3. The only P1 survivor, `drift`, is a **decision-projection dispersion regulariser**:
   ≈ kappa·Var_r(w_tu^T D_{t,r}) over a 9-pair set (asymmetric softplus, ||w||-cheatable,
   common-component blind). Its gain over label smoothing is only +0.07/+0.88 pp.
4. Label smoothing 0.1 is the strongest generic control (+0.28/+1.75 vs CE; best arm on Pluto).

**P2 hypothesis.** For a linear head, a feature-space displacement δ of class t's mean perturbs
every normalized margin by the projection of δ onto the (centered) classifier row space — the
9-dimensional margin subspace span{w_t − w_u}. Source-RX displacements D_{t,r} = mu_{t,r} − mu_t
are the observable instances of the perturbations that break unseen-RX accuracy. The mechanism-
aligned target is therefore: **minimise the fraction of each D_{t,r} that lies in the margin
subspace, over every class and every source RX** — the exact quantity that P1's margin arms tried
to protect at the level, but on the sensitivity (curvature) side, with two degeneracies removed:
scale-invariance in ‖w‖ (Rayleigh-style normalization ⇒ cannot be cheated by shrinking the head)
and coverage of all 90 ordered pairs / all classes (no capacity re-allocation target list).

Design-history note: an early draft of this protocol minimised the *complement* projection
(P_perp), i.e. pushed displacements INTO the decision subspace. That direction is wrong: the
margin perturbation under displacement δ is (A δ) for centered A = W − mean(W), so margin damage
is zero exactly when δ is orthogonal to row(A). The corrected form below was fixed before any fit.

## 1. Losses (p2_losses.py; unit-tested before any fit)

Notation: `pre` = 128-D pre-head features (ReLU(tx_input(feat512(x)))), W = live classifier
weight (10×128), A = W − mean_rows(W) (rank ≤ 9), B = orthonormal basis of row(A) (from SVD,
tolerance 1e-5·σ_max), mu_{t,r} = batch mean of pre over (y=t, RX=r), mu_t = mean over source RX
in batch, D_{t,r} = mu_{t,r} − mu_t (E_r D = 0 on balanced batches; 2 packets per (t,r) per batch
as in P1). P_marg = BᵀB never materialised; ‖P_marg D‖² = ‖B D‖².

| arm | aux loss added to CE | question |
|---|---|---|
| B0 ce | CE (frozen R0/R1 checkpoints reused, no refit) | baseline |
| B6 ls | CE + label smoothing 0.1 (P1 checkpoints reused, no refit) | generic control |
| B1 rxnull | λ · mean_{t,r} ‖B D_{t,r}‖² / (‖D_{t,r}‖² + eps), eps=1e-4 | main arm: null the decision-subspace exposure of every RX displacement |
| B2 cmean | λ · mean_{t,r} ‖B D_{t,r}‖² (unnormalized) | is the ‖D‖-normalization (anti-collapse) needed |
| B3 rxsel | λ · mean_{(t,u)∈Rq} mean_r (ŵ_tuᵀ D_{t,r})² with ŵ = (w_t−w_u)/‖w_t−w_u‖, each term / (‖D_{t,r}‖² + eps). CLI name `rxsel`; the loss is `rxnull9` in p2_losses.py (run_id parsing requires the arm tag to end in the λ value) | is all-pairs coverage needed (else the P1 capacity reallocation returns) |
| B4 rxnull_ls | CE(LS 0.1) + λ · rxnull term | additivity with the generic control |
| B5 drift_ls | CE(LS 0.1) + λ · drift_v0(P0 code, Rq set, γ=0.5) | is the P1 survivor additive with LS |
| ctrl rxnull_rand | λ · rxnull term with D_{t,r} ← same-norm random direction (fold-seeded rng, redrawn per batch) | is the actual displacement geometry needed, or does any isotropic push do |

CE part of aux arms: plain F.cross_entropy (no weight-decay term), exactly as P1 aux arms; LS arms
use label_smoothing=0.1 as P1 ls. rxnull_rand uses λ*(rxnull) from the search.

Properties (unit-tested): ρ := ‖B D‖²/‖D‖² ∈ [0,1]; ρ=0 ⇔ every raw and normalized margin
perturbation A·D vanishes; invariant to scaling W or D (ρ(cW, c'D) = ρ(W, D)); relabel-invariant;
gradcheck in float64; B from rank-deficient A never spans a spurious direction (SVD tolerance);
rxnull9 equals the per-pair normalized-margin damage; ‖A D‖ ≤ σ_max(A)·‖B D‖ (damage vanishes
with ρ). Random-direction control has E[ρ] = rank(A)/128 under isotropic g (reported by the gate).

## 2. Pre-training mechanism gates (gate_rxnull.py; inference-only; must run before any fit)

On frozen CE (R0/R1) formal checkpoints, source_train packets of each fold's 5 source RX:

- **G1 (gate, must pass)**: ρ_row := mean_{t,r} ‖B D_{t,r}‖²/(‖D_{t,r}‖²+eps) per fold.
  Pass = ρ_row > 0.15 in BOTH folds (≈ 2× the isotropic 9/128 ≈ 0.070). Fail ⇒ STOP before any
  fit; the negative gate is itself the deliverable (displacements are not decision-visible, the
  P1 mechanism claim loses its basis).
- **G2 (report)**: Spearman across the 90 ordered pairs between per-pair displacement visibility
  v_{t,u} = mean_r (ŵ_tuᵀ D_{t,r})²/(‖D_{t,r}‖²+eps) and P0's frozen pair risk R_q; per fold.
- **G4 (report)**: ρ_row recomputed under the P1 formal checkpoints of ls / drift / var / margin /
  tail (all source RX). If LS lowers ρ_row materially, "LS is a weak rxnull" becomes the working
  explanation of the LS gain (incl. Pluto), testable by arm B4; report regardless of outcome.
- **G3 (report)**: E[ρ] of the random-direction control on the same D norms; per fold.

The gate is inference-only (minutes, 1 GPU), run after committing protocol+code, before queueing
any fit. Nothing in this section reads outer-RX data.

## 3. λ selection (source only; P1 recipe verbatim)

Search stage = P1 search splits verbatim: train inner_train RX (first160), validate on the
inner_validation RX (all packets), seed 20261016, 20 epochs, same schedule/optimizer/clip.
λ ∈ {0.3, 1.0, 3.0} for rxnull / cmean / rxsel / rxnull_ls / drift_ls (2 folds × 5 methods ×
3 λ = 30 search fits; ce/ls reused from P1 search). Rule: per method, λ with minimum held-RX CE
(tie → smaller λ); formal uses that λ even if it does not beat ce (reported). Outer RX never read
in search. If the selected λ sits at the grid boundary 3.0 for an arm entering the decision,
report boundary-limited; no post-hoc λ extension (pre-registered).

## 4. Formal, metrics, decision (pre-registered)

Formal = all 5 source RX (first160), 2 folds × seeds {20261013, 20261014, 20261015}, checkpoint
by last-40 receiver-equal val CE, same recipe as P1. Arms: rxnull, cmean, rxsel, rxnull_ls,
drift_ls at λ*; rxnull_rand at λ*(rxnull); ce/ls reused from frozen R0/R1 and P1 formal.
36 new formal fits.

Primary: RX-equal outer accuracy. "X beats Y": seed-mean Δacc > 0 in BOTH folds AND ≥ 5/6 paired
(fold, seed) Δ > 0. Secondary: Rq-set mean outer confusion, non-P mean confusion, P normalized
margin mean/q10, per-RX×TX recall, worst outer RX, Pluto, all 90 ordered-pair confusions,
per-arm ρ_row at the end of training (diagnostic), seconds per fit.

Decision:
- **GO (method line)**: rxnull beats ce AND ls AND rxnull_rand AND rxsel in the pre-registered
  rule, P-set confusion not above ce in either fold, and non-P confusion ≤ ce + 0.005.
- **MODIFY**: rxnull beats ce and ls but fails against rxsel or rxnull_rand → keep the simplest
  effective form; no module claim for the redundant part (all-pairs coverage or displacement
  geometry), novelty reassessed.
- **STOP (method line)**: no P2 arm beats both ce and ls → the sensitivity family is exhausted;
  paper = diagnosis + negative mechanism + reporting standard (P1 §4 framing) + gate numbers.
- Normalization ablation (reported, not gated): if cmean beats rxnull, the unnormalized form is
  kept and the anti-collapse claim dropped; if neither dominates (|Δ|<0.5 pp both folds), the
  normalized form is kept a priori.
- Complementarity: rxnull_ls must beat ls AND rxnull (both by the rule) for any additivity claim;
  drift_ls vs ls and vs drift(P1) reported the same way.
- Pluto reported separately, always. No non-identifiability claim. WiSig transfer proposed in
  writing only (CE/LS/rxnull/drift_ls/rxnull_ls, source-only λ), executed after approval.

## 5. Budget / stop

30 search + 36 formal fits; gate and eval inference-only. Cap **9 GPUh** (job-hour accounting as
P1; P1 measured ~2.8 min/search fit, ~7 min/formal fit, 2 jobs per V100 ⇒ device occupancy ≈ half
the job-hour sum). No added seeds, no λ changes after outer evaluation, no new arms after the
gate. After the report: stop; nothing else runs without approval.

## 6. What would make a paper (my judgement, non-binding)

Contributions, best case: (i) cross-RX failure concentrates on predictable directional pairs
(P0/P1, both folds, WiSig mirror); (ii) direct margin protection re-allocates errors — a
mechanism-level negative result with the capacity explanation; (iii) G1/G4: a structural measure
(decision-subspace exposure ρ_row of class-mean displacements) that explains both the failure and
why label smoothing helps — first mechanistic account of the strongest generic control in RFFI;
(iv) if B1 survives: the exposure-minimising regulariser as the first mechanism-aligned control
that beats LS under a fair protocol. If (iv) fails: (i)+(ii)+(iii) plus a reporting-standard claim
(LS-level controls absent from most RFFI-DG papers) is the honest paper.

## 7. Artifacts

`scripts/receiver_p2/`: p2_losses.py, test_p2_losses.py, gate_rxnull.py, train_p2.py,
select_lambda_p2.py, queue_p2.py, eval_p2.py, decide_p2.py, phase.sh.
`reports/receiver_p2/`: gate_rxnull.json, lambda_p2.json, search_table.csv, jobs_formal.txt,
eval/{runs,per_rx,per_rx_tx,pairs,pairwise_beats,summary_by_fold_arm}.csv, decision.json,
REPORT.md. Outputs under `outputs/receiver_p2/{search,formal}` (local only).
