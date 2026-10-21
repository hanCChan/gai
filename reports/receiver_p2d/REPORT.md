# P2D: margin-tail audit + auto-chain candidate search — 2026-10-22

Branch `experiment/p2-rx-null` after P2C (57d26a5). Everything here runs the sealed Shen
protocol (5 source RX; outer RX read only at eval). Losses unit-tested (16/16, `test_p2d_losses.py`)
before any fit; each candidate family was registered in `run_p2d_chain.py CANDIDATES` only
AFTER the previous round's post-mortem. Artifacts: `reports/receiver_p2d/{tail_audit,
eval_r1..r3, decision_p2d_r*.json, search_table_p2d_r*.csv, lambda_p2d_r*.json}`;
checkpoints in `outputs/receiver_p2d/` (not in git).

## 0. Executive summary

1. **Tail audit (no training, 102 checkpoints)**: outer-RX accuracy is summarised far better by
   outer-RX margin-tail statistics than by source κ. **Caveat that limits this result**: the tail
   statistics are computed on the OUTER RX themselves, while κ is computed on SOURCE RX. The mean
   pair-flip rate `P(M_tu<=0)` is close to a restatement of the outer error rate (Spearman −0.90/−0.93
   is near-tautological and is not mechanism evidence). The informative part is narrower: a
   mean+variance summary of outer margins (Cantelli) explains outer accuracy better than source
   κ does, so margin *spread*, not only mean displacement, matters. **Whether a source-computable
   tail statistic predicts unseen-RX error is untested**, and r3 below is evidence against the
   naive version.
2. **Three chain rounds ran end-to-end** (search → λ-select → formal → eval → decide).
   No new arm beat ccalign. Best: **rxccp_soft** (log-sum-exp relaxation of worst-source-shift CE)
   is the only P2D arm that beats BOTH ce and ls (5/6, 5/6), but it trails ccalign by −1.24/−1.07 pp
   and has a worse worst-RX.
3. **Post-mortems**: r1's hard inner max collapses onto one receiver per class (b210 picked as
   worst for 7/10 classes on fold 0), which is the P1 fixed-hard-set failure again. r3's Cantelli
   loss *improves* source held-RX CE in search (0.284 vs ce 0.297 / 0.245 vs 0.285) yet loses on
   outer RX (fold 0 −0.78 pp vs ce): lowering source-RX margin spread does not transfer.
4. Audit fixes from the external review: the P2C ISR control now has a fair refit baseline, and
   the WiSig probe uses an 8-class head, the TX0-7 filter, and the same base regulariser in both
   arms. Measured cost is 2.2 s/epoch, so the full WiSig plan (30 search + 36 formal) is ≈3.1 job-h.

## 1. Tail audit (`tail_audit/`)

n=102 checkpoints (ce R0/R1, all P1/P2/P2B formal). Spearman with outer accuracy; κ on source RX,
all other columns on outer RX:

| fold | κ (source) | mean P(M<=0) (outer) | mean Cantelli (outer) | mean q10 (outer) |
|---|---:|---:|---:|---:|
| 0 | −0.57 | −0.90 | −0.60 | +0.29 |
| 1 | −0.46 | −0.93 | −0.65 | +0.47 |

Partial Spearman (per fold): q10 given κ = +0.11/+0.36; κ given q10 = −0.52/−0.35. Pooled with the
r1 arms (n=120): Cantelli given κ = −0.76. The Cantelli bound tracks realised cell-level flip rates
at ρ≈0.73–0.75. Read these as descriptions of where outer errors sit, not as source-side predictors.
Files: `tail_models.csv`, `tail_cells.csv`, `tail_verdict.json`, `tail_models_r1.csv`.

## 2. Chain rounds

r1: 18 search + 18 formal fits; r2 and r3: 12 + 12 each. A round takes ~25–35 min wall on 4×V100.

| round | arm | beats ce | beats ls | beats ccalign | worst-RX f0/f1 | verdict |
|---|---|---|---|---|---|---|
| r1 | rxccp (hard max) | **6/6** +1.19/+2.72 pp | 4/6 | 0/6 −1.05/−0.83 | .782/.819 | — |
| r1 | rxgeo (geometric ablation) | 2/6 | 2/6 | 0/6 | .765/.803 | — |
| r1 | rxccp_ls | 4/6 | 3/6 | 0/6 | .736/.824 | — |
| r2 | rxccp_soft (LSE β=1) | **5/6** +0.99/+2.48 | **5/6** +0.71/+0.73 | 1/6 −1.24/−1.07 | .749/.806 | ITERATE |
| r2 | rxccp_soft_ls | 3/6 | 2/6 | 0/6 | .683/.810 | — |
| r3 | rxcant (Cantelli loss) | 4/6 | 2/6 | 1/6 | .732/.802 | — |
| r3 | rxcant_ls | 5/6 | 4/6 | 0/6 | .740/.800 | STOP_TAIL |

Reference arms (from P2/P2B): ccalign beats ce and ls 6/6; rxnull 6/6; rxlogit 5/6.
Decisions: `decision_p2d_r{1,2,3}.json`. r1's decision file says STOP_TAIL (no arm beat ls); it was
recorded before the chain's auto-continue logic was in place, and r2 was then launched by hand.
`chain_state.json` holds only the last invocation (r3).

### 2.1 Post-mortems

- **r1, hard max.** The per-class worst-shift selection is degenerate. On the trained fold-0 rxccp
  checkpoint the selected receiver is b210 for 7/10 classes and rtl_6 for the other 3. With 5 source
  RX there are only 4–5 shift directions per class and they sum to zero, so the minimax becomes a
  fixed hard-receiver reweighting, the same failure as P1.
- **r2, soft max.** Keeping all RX in play clears the ls bar, but fold-0 worst-RX (Pluto) drops from
  .782 (r1) to .749: spreading weight away from the hard receiver hurts the receiver that needed it.
- **r3, Cantelli loss.** Source-side selection says it works (held-RX CE beats ce at λ*), and the
  outer RX say it does not. Source pooled margin variance is not the variance that unseen receivers
  induce. This is the gap the Cantelli argument in the review warned about: the bound holds on the
  true target-RX distribution, and source estimates do not constrain it without a shift assumption.

**Conclusion of P2D.** Neither mean-displacement control (κ family) nor worst-shifted CE (rxccp
family) nor source-tail minimisation separates from plain class-conditional alignment on Shen.
The open question has moved to: is there a *source-computable* statistic that predicts unseen-RX
tail risk? Without one, a tail-targeting loss has nothing valid to optimise.

## 3. Audit fixes (external review items)

- `isr_posthoc.py`: new `refit` arm (no projection, seed fixed before head instantiation, CE + L2=1e-4
  per `C.REFIT`); the ISR arms now use the identical recipe. Result (`../receiver_p2c/isr_posthoc_fixed/`),
  outer acc mean over folds and seeds:

  | base | frozen head | fair refit | isr | isr_normfilt |
  |---|---:|---:|---:|---:|
  | ce | .8406 | **.8544** | .8460 | .8484 |
  | rxlogit | .8680 | **.8736** | .8632 | .8626 |
  | rxnull | .8734 | **.8759** | .8681 | .8676 |

  ISR is below the fair refit on all three bases. P2C compared ISR only with the frozen head
  (−0.9/+1.9 pp on CE features); part of that apparent gain was the refit itself, not the projection.
- `wisig_cost_probe.py`: `OfficialCNN()` (8 classes, matching W0 history and the TX0-7 protocol)
  instead of `OfficialCNN(10)`; the shared `wisig()` loader does not filter TX, so `y<8` is applied in
  the probe; `model.regularization()` in both the ce and rxlogit branches; `CUBLAS_WORKSPACE_CONFIG`
  set; single-epoch mean no longer NaN. Measured (3 epochs): ce 0.4 s/epoch, rxlogit 2.2 s/epoch.
  Note: a 3-epoch rxlogit run at λ=1 kept val CE flat at ~2.088 while ce fell to 1.57, so λ=1 may be
  too strong on WiSig; the λ search in the protocol covers this but it is worth watching.
- Not fixed here: the P2C report's "on both datasets-to-date" should read "on Shen" (WiSig was not
  run). The published P2C record is left as is; this note is the correction.

## 4. Next step (not run)

`run_p2d_chain.py` is resumable and skips completed fits. To run a new family, add it to
`CANDIDATES` and call `python3 run_p2d_chain.py --families rN --max-rounds 1`.

Given §2.1, the next useful step is an analysis, not a new loss: test whether any SOURCE-side
statistic (source-RX leave-one-out margin variance, κ, source Cantelli, source pflip) predicts
outer-RX pflip across the 120 checkpoints. If none does, a tail-targeting objective has no valid
source signal and the method line should stop at "class-conditional alignment is the strong
baseline". If one does, that statistic defines the r4 constraint.

## 5. Budget

84 fits (42 search + 42 formal), 7.43 job-h of fit time on 4×V100-SXM2. All fits use the 20-epoch
Shen recipe, deterministic, seeds 20261013–16 (search 20261016), λ grid {0.3, 1, 3}.
