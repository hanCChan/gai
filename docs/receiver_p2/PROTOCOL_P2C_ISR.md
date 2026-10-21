# P2C: post-hoc ISR-Multiclass control (frozen 2026-10-21, before running)

Purpose: rule out the strongest post-hoc competitor found in the novelty check
(ISR-Multiclass, arXiv 2311.00966: per-class per-environment means → per-class PCA spanning
span{D_{t,e}} → SVD of stacked bases → fit a linear classifier in Null(P′)). Inference + head refit
only, on frozen features of existing checkpoints; no encoder training.

## Arms evaluated (Shen, both folds, seeds 20261013-15 of the formal checkpoints)

Base checkpoints: ce (R0/R1), rxlogit (P2B), rxnull (P2) — the three that matter.
For each base checkpoint:

1. `refit` = linear head refit on frozen pre-head features, **no projection** (the P0 `refit_head`
   recipe, 500 steps, lr 3e-3, l2 1e-4, seed 20261019) — isolates the head from the trained head.
2. `isr` = same refit on features projected onto Null(P′). P′ built from source_train packets only:
   per class t, per source RX e, μ_{t,e}; D_{t,e} = μ_{t,e} − mean_e; per-class PCA basis of the D's,
   keeping every direction with singular value > 1e-6·max (faithful: the exact span, ≤ E−1 dims per
   class); stack all class bases, take the left singular vectors of the stacked matrix with a
   relative threshold 1e-3 of the top singular value as P′.
3. `isr_normfilt` = same with directions kept only if the corresponding displacement norm exceeds
   10% of the per-class between-mean norm (guards against fitting noise directions on arms whose
   displacements are tiny, e.g. rxlogit).

Evaluation: RX-equal outer accuracy, worst outer RX, Pluto; also report source-val accuracy of the refit
and the number of removed directions.

## Interpretation (fixed before running)

- If `isr` on **ce** features reaches the trained arms' accuracy (within the P2 rule margin, i.e. does
  not lose by the beats rule), then "training the encoder to reduce exposure" is not established as
  better than a post-hoc linear fix, and the paper must say so; the method claim rests on WiSig only.
- If `isr` on **rxlogit** features loses to trained rxlogit (base), the frozen features already carry
  almost no decision-visible displacement (consistent with κ ≈ 0.002–0.004), i.e. the training did the
  work that the post-hoc projection cannot replicate from what is left.
- `refit` vs frozen-head numbers act as a sanity check that our refit recipe is not the confound.
- No Shen hyper-parameter or arm is changed on the basis of this control.
