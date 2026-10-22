# P2E errata and code audit (2026-10-22)

An independent read-only audit of `scripts/receiver_p2/` (P2–P2E) found no data leakage into training,
epoch selection, λ selection or receiver statistics, and nothing that changes the E4 STOP or E6
STOP_NOISE_LINE decisions. It found the problems below. "Fixed" = code changed in this commit;
committed CSVs from before the fix are kept and marked.

## Affects interpretation

**A1 `cc` / `ccproj` / `ccrand` / `ccw` put the test label into the test features** (`p2e_audit.py`).
Each test packet was moved by the correction of its *own true class* (`h − D[y_true]`). On features that
carry no class information this decode gives 100% (regression test: `p2e_generality.py --dataset selftest`
→ leaky 1.000, label-free 0.101). Every "recoverable shift" number and every "share of cc gain" in the
first P2E REPORT used this denominator. **Fixed / replaced**: `p2e_generality.decompose` keeps the old
decode only as `*_leak`, adds `*_lf` (per-class bias correction, label never used) and the proper
Gaussian version `acc_lda_*` (source pooled covariance, class means from the even half, scored on the odd
half; the decision directions move with the means). Corrected numbers: REPORT §1, §8.
Consequence: on Shen noisy RTL receivers, the earlier `cc` 89–90 was the label; the label-free class-mean
fix adds only +0.1–0.4 pp there. On WiSig, the earlier "≈40 pp class-mean shift" (leaky cc 94.5) is
88.4 with the label-free LDA decode. The conclusion survives, at a smaller size.

**A2 Fidelity "source" rows compare against in-domain accuracy** (`p2e_fidelity.py`). The formal
checkpoints were trained on the receiver being imitated, and the source span U included its own signature
(so `sigproj == sig` on every source row). On the truly unseen (outer) rows the drop correlations are
noise .38, signature .37, **noise+signature .78**. The first report's "only the noise model is valid as a
counterfactual; the signature over-damages clean RX by 6–14 pp" is withdrawn. On unseen RX the signature
over-damage is 2.5–10.7 pp and noise+signature fits best. Not re-run. A valid version would use LORO
checkpoints with held = q and build U without q.

**A3 The search-stage "held" RX selected λ and the epoch.** Relabelled "selection RX" (rtl_6 / Pluto on
Shen, inner_held on WiSig). They are not unseen receivers.

**A4 `ccanoise` / `ccancons` ccalign weight ≠ the ccalign anchor in fold 1** (`train_p2e.py` CCA_LAM =
P2B {0: 3, 1: 1} vs LORO-selected 3 in both folds). Fold-1 comparisons of the combined arms carry a 3×
weaker alignment term. The likely size of the effect is small (P2B ccalign λ=1 vs P2E λ=3: 89.0 vs 89.4).
**Fixed**: `--cca-lam` flag. Combined arms not re-run, because the noise line is closed for a
dataset-level reason (§8: the noise floor does not exist on WiSig).

**A5 Noise vs isotropic control was not energy-matched**: λ* noise 2/4 vs noiseiso 1/2; ncons 3 vs
nconsiso 0.3. "Beats its isotropic control" confounds profile shape with noise energy. Several λ* sit on
a grid edge. The claim is downgraded to "calibrated noise at its selected scale beats isotropic noise at
its selected scale".

**A6 REPORT statements the CSVs did not support (corrected in REPORT)**:
- ccalign also raises noisy-RX raw accuracy (+2.9 pp over ce) and the noisy-RX oracle (83.4 → 85.6), so
  "alignment acts only on the shift component" was wrong.
- The noise arm's clean-RX cost is relative to ccalign. Against ce it is −1.19 / +0.66 pp (5/10).
- The LORO table omitted rtl_6, rtl_9 and b200 (where noise is weakest).
- Several ranges were wrong: oracle gap 1.23–2.82 (not 1.3–2.3); random-span share 0.10–0.56; oracle
  head 93.5–96.1.

**A7 Two analyses had no committed script.** **Fixed**: `p2e_analyses.py predictors` (regenerates
`audit/source_predictors.csv` byte-identically) and `p2e_analyses.py input` (per-RX input stats, SNR,
TX-independent share, offset coverage). The input coverage now has a fair control: a random subspace
inside the source packet PCA span gives 0.07–0.11, against 0.49–0.99 for the source offset span.

**A8 Protocol**: the noise idea and the noisy/clean grouping came from labelled diagnostics on the outer
RX, before E1. Outer RX are therefore development data for the noise arms, not a clean test. No code
path feeds outer data into training or selection.

## Minor bugs (fixed in code; affected committed numbers noted)

- B1 `eval_p2.py`: per-RX pair statistics pooled over all outer RX (`mt = y == t` without `& (rx == r)`).
  Every per-`rx_key` row of P2/P2B `pairs.csv` holds the pooled value. P_conf / P_nm_mean are unaffected
  (balanced design); P_nm_q10 is a pooled quantile. `tail_audit.py` and `gate_rxnull.py` do not read this file.
- B2 `train_p2e.receiver_stats`: float32 SVD kept a null direction (`sig_dims` 4/5 instead of 3/4; its
  σ_a ≈ 1e-8, negligible). Now float64.
- B3 `p2d_losses.rxccp_soft`: `logsumexp(βc)/β − log n` should be `(logsumexp(βc) − log n)/β`. Identical
  at β = 1, the only value used.
- B4 `p2d_losses.tailrisk`: softplus not squared (diagnostic only); now matches `tail_audit`.
- B5 `isr_posthoc.py`: the normfilt threshold divided by √d (≈0.9% of class separation, not 10%), so the
  committed `isr_normfilt` rows used the wrong threshold. `src_val_acc` was the refit-packet accuracy;
  renamed `src_fit_acc`. The plain `isr` arm is unaffected.
- B6 `p2e_audit.py`: formal-checkpoint source metrics use the epoch-selection val packets (mild
  optimism); `ccrand` leaves span(V) when pca_k < dim U (4 collapsed search models).
- B7 `run_p2e_chain.py`: GENERIC also caught arms failing G1 (docstring fixed); `beats()` now asserts no
  duplicate (fold, seed, arm); `select()` filters to the registered grid.
- `p2e_eigenrx.py` (written in this round): a duplicated `acc_ca` key crashed the first WiSig run.
  Fixed before any result was used.
