# P2A protocol: mathematical / implementation audit of rxnull (frozen 2026-10-21, before the audit runs)

Base: 4f00c7f (P2 GO). Inference only on existing checkpoints; no new training in this step.
Trigger: external review of P2. Accepted corrections to the P2 report (errata, applied in the audit
report, P2 pre-registration left byte-unchanged):

- E1. 9/128 is the expectation of ρ for an isotropic random displacement when rank(A)=9, NOT a lower
  bound; ρ=0 for any displacement in null(A). "8.8× the isotropic floor" → "8.8× the isotropic
  reference"; "rxnull reaches the floor" → "rxnull brings ρ to about the isotropic reference".
- E2. ρ is a ratio in a trainable feature metric. It is NOT invariant to function-preserving
  reparameterisations: h ↦ P_row h + s·P_null h changes logits only by a common-mode term
  (A·P_null = 0, so W·P_null = 1·w̄ᵀP_null), so predictions are identical, while ρ changes with s.
  "ρ equals all of the argmax damage" is withdrawn. What holds:
  σ_min⁺(A)²‖P_row D‖² ≤ ‖AD‖² ≤ σ_max(A)²‖P_row D‖² = σ_max(A)² ρ ‖D‖²,
  so ρ bounds the logit displacement only together with ‖D‖ and the conditioning of A.
- E3. ρ squares the projection and does not separate margin-enlarging from margin-shrinking
  displacements. (Target-free argument kept for the paper: source displacements are centred,
  E_r D = 0, so each direction appears with both signs across source RX; with no target information
  the minimax choice over the unknown sign is the magnitude.) Harmful vs helpful parts are measured
  on outer RX below as a diagnostic.
- E4. "LS works because it lowers ρ" is downgraded to an association (CE 0.62, LS 0.39); no causal
  claim without an intervention.

## 1. Quantities (per checkpoint; source_train packets = first160/TX of the 5 source RX, model scaler)

h = 128-D pre-head features; z = Wh+b; z̃ = z − mean_classes(z) (= A h + const); μ_{t,r}, μ_t = mean_r
μ_{t,r}, μ̄ = mean_t μ_t; D_{t,r} = μ_{t,r} − μ_t; C_t = μ_t − μ̄; P_row/P_null as in p2_losses.

| name | definition | level |
|---|---|---|
| ρ | mean ‖P_row D‖²/(‖D‖²+1e-4) | feature metric (P2 objective) |
| E_row, E_null, E_D | mean ‖P_row D‖², ‖P_null D‖², ‖D‖² | feature metric |
| n_row, n_null | E_row / B_row, E_null / B_row, B_row = mean_t ‖P_row C_t‖² | relative to decision-space class separation |
| **κ** | mean_{t,r} ‖A D_{t,r}‖² / mean_t ‖A C_t‖² (computed from z̃ directly) | **function level**: invariant to null inflation and to scaling W or h |
| MD | mean_{t,u} Var_r(m_{t,u,r}) / mean_{t,u} (mean_r m_{t,u,r})², m = class-t mean of z_t − z_u on RX r | function level, all 90 ordered pairs |
| S_w | mean packet ‖z̃ − mean z̃_{y,r}‖² / mean_t ‖A C_t‖² | function level, within-cell logit scatter |
| A spectrum | σ_1..σ_9, ‖A‖_op, ‖A‖_F, effective rank (Σσ²)²/Σσ⁴, σ_1/σ_9 | head |
| src_val_acc, src_min_recall | on source last-40 packets | identity retention |

Outer RX (repeatedly-developed data; diagnostic only, never used for selection): D_out = μ_{t,r'} −
μ_t^src; κ_out = mean ‖A D_out‖² / mean_t ‖A C_t^src‖²; ρ_out; harmful shrinkage
H = mean_{t,u,r'} max(0, −Δm)/mean_{t,u} m^src with Δm = (w_t−w_u)ᵀD_out; fraction of Δm<0; outer acc.

Models: ce (frozen R0/R1, 6), all P1 formal (48), all P2 formal (36) = 90.

## 2. Pre-declared verdict

**MATHEMATICAL_GUARD_PASSED** iff all, comparing seed means within each fold:
- G1 κ(rxnull) < κ(ce) and κ(rxnull) < κ(ls), both folds — the function-level decision-relevant RX
  displacement is actually reduced.
- G2 n_row(rxnull) < n_row(ce), both folds — decision-space displacement shrinks relative to
  decision-space class separation.
- G3 share_row := [log n_row(rx) − log n_row(ce)] / [log ρ(rx) − log ρ(ce)] ≥ 0.5, both folds — at
  least half of the log-ρ reduction comes from less decision-space displacement, not from more
  null-space energy (log ρ = log n_row − log(n_row + n_null)).
- G4 effective rank of A (rxnull) ≥ 0.8 × ce and src_val_acc (rxnull) ≥ ce − 1 pp, both folds — no
  head-rank collapse, identity retained.
- G5 (correctness of the audit itself) on one ce and one rxnull checkpoint per fold, float64,
  s ∈ {0.25, 4, 16}: predictions and κ unchanged (|Δsoftmax| < 1e-6, |Δκ|/κ < 1e-6), ρ changes.
  Failure = audit bug, fix before any verdict.

Otherwise **MODIFY_OBJECTIVE**. Minimal fix fixed now (no other variant may be introduced):
rxlogit = CE + λ·[mean_{t,r} ‖A D_{t,r}‖² / (mean_t ‖A C_t‖² + 1e-6)] per batch (batch estimate of κ),
then Shen search (λ ∈ {0.3, 1, 3}) + formal (3 seeds × 2 folds) under the P2 rules, compared with
rxnull, ce, ls before any WiSig fit.

Reported, not gating: within-fold Spearman over the ~45 models of {κ, ρ, n_row, MD, src_val_acc}
against outer accuracy, and rank-partial correlation controlling for src_val_acc; Spearman(κ, κ_out);
per-RX gains, macro-F1, worst RX; static check that search/formal never load outer RX; fairness notes.
Association across arms is not causation.
