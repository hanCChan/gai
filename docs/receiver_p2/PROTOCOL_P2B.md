# P2B amendment: function-level objective and class-conditional alignment control (frozen 2026-10-21, before any P2B fit)

Trigger: P2A verdict MODIFY_OBJECTIVE (b5229c3). Shen only, same recipe/splits/seeds/schedule/selection
as P2. Existing ce (R0/R1), ls (P1) and rxnull (P2) checkpoints are reused, not retrained.

## 1. Arms (p2_losses.py; tests test_p2b_losses.py 17/17 before any fit)

| arm | aux added to plain CE | role |
|---|---|---|
| rxlogit | λ · mean_{t,r}‖A D_{t,r}‖² / (mean_t ‖A C_t‖² + 1e-6), live batch estimate of κ | minimal fix pre-fixed in P2A; function-level, invariant to null-space rescaling and to scaling W or h |
| ccalign | λ · mean_{t,r}‖D_{t,r}‖² / (mean_t ‖C_t‖² + 1e-6) in the full 128-D feature space | first-moment class-conditional alignment (CIDG-style) control: aligns all RX displacement, ignores the classifier |

D_{t,r} = μ_{t,r} − μ_t, C_t = μ_t − mean_t μ_t on balanced batch cells; A = row-centred live W. Only
these two arms are added; no other variant may be introduced in P2B.

## 2. Search and formal

λ ∈ {0.3, 1, 3}, search split and seed 20261016 as P2, min held-RX CE (tie → smaller λ), boundary
reported, no extension: 2 folds × 2 arms × 3 λ = 12 search fits. Formal: λ*, 2 folds × seeds
{20261013, 14, 15} = 12 fits. Budget: ≈2.6 job-h; P2 cumulative stays under the 9 job-h cap.
Outer evaluation only after all 12 formal checkpoints exist.

## 3. Decision (P2 "beats" rule on RX-equal outer accuracy)

- R1: rxlogit beats ce AND ls, and rxnull does not beat rxlogit → **rxlogit becomes the main
  objective** (κ is the mechanism quantity; theory and objective consistent).
- R2: rxnull beats rxlogit → rxnull stays the main empirical arm, flagged "lead not explained by κ";
  no κ-based causal claim for its extra gain; both go to WiSig.
- R3: otherwise (mixed) → both go to WiSig, no main-objective claim yet.
- Novelty vs class-conditional alignment: the main objective(s) must beat ccalign; otherwise "targeting
  the decision subspace is not shown to beat plain class-conditional alignment on Shen".
- Reported, not gating: P-set / non-P confusion, worst RX, Pluto, and a P2A re-audit of rxlogit and
  ccalign (κ, n_null, E_D, effective rank) to confirm rxlogit has no null-inflation path.
