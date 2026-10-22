# P2E: where unseen-receiver errors come from, and a receiver-calibrated test of the answer — 2026-10-22

Base: `experiment/p2-rx-null` @ 23dbc03. Shen Strict CIS, the 10 registered RX only (no sealed data).
Errata for P2D: `ERRATA_P2D.md`. All P2E fits use `train_p2e.py`, which reproduces the registered
R0/R1 CE checkpoints **bitwise (6/6)**, so every P2E arm differs from CE only by its own term.

## 0. Summary

1. **Two failure types, not one (257 checkpoints, inference only).** On unseen RX, every method's error
   splits into (a) *noisy RTL receivers* (rtl_1/2/3/5): features are degraded. A linear head refit on the
   target RX's own labels gains only 1.3–2.3 pp for every arm, so neither alignment nor a better head can
   help. (b) *USRP/Pluto*: features stay separable (oracle head 94–96) but class means shift. For
   ccalign 9.6 pp is recoverable, and 91% of that is recoverable using only the source-RX shift span.
   Mean-alignment methods (ccalign, rxnull, rxlogit) act only on (b).
2. **The input explains the split.** Between-class CIS variance is identical on all 10 RX (0.0022–0.0024);
   within-class variance varies 3.3× (USRP/Pluto 0.0064 → rtl_5 0.0213), concentrated in the in-band rows.
   Unseen-RX TX-independent input offsets lie largely in the ≤4-dim span of source offsets
   (coverage 0.49–0.98 vs ~0.001 for random subspaces of equal dimension).
3. **Counterfactual fidelity (no training).** Transplanting the source-estimated noise profile onto clean
   source packets reproduces the noisy receivers' accuracy drops (source-only corr 0.82). Transplanting
   the TX-independent signature does not (it over-damages clean RX by 6–14 pp). Only the noise model is
   valid as a training counterfactual.
4. **Pre-registered chain E1–E4 → STOP.** Receiver-calibrated noise beats CE on the noisy RX in 10/10 paired
   runs (+3.1/+4.2 pp), beats the isotropic-noise control (9/10 noisy, 8/10 overall), and raises the
   noisy-RX oracle ceiling 83.6 → 87.6, i.e. the encoder extracts more information. It costs 1.6–3.9 pp
   on clean RX, so overall it does not beat ccalign. Signature extrapolation does nothing beyond its
   random-direction control.
5. **Source-only evidence is weak in this setting.** Single-held-RX selection (used in P2–P2D) is
   anti-correlated with outer accuracy in fold 1 (ρ = −0.17). Nested LORO over all 5 source RX ranks the
   8 arms at ρ = 0.43 / −0.10 against outer. Pluto as a held source RX is not predicted by any source
   statistic.
6. **E6 (registered after E4)** tests one hypothesis from (4): the clean-RX cost comes from training CE on
   noisy inputs, not from noise invariance. See §5.

## 1. Error composition (`audit/`, `audit_p2e/`; `p2e_audit.py`)

For each checkpoint × unseen RX, accuracy with the frozen head after successive corrections of the target
features (statistics from the target RX, accuracy on the odd half of each class):
`ca` label-free RX centering; `cc` oracle class-mean fix; `ccproj` the same restricted to the source
shift span; `ccrand` random span of equal dimension; `oracle` linear head refit on the target's even half.
`cc` uses labels per class and also re-expands noise-compressed class means, so it can exceed `oracle`;
read it as "first moment removed", not as an attainable bound.

Formal checkpoints, outer RX, grouped by input noise (acc ×100):

| group | arm | raw | ca | cc | ccproj share of cc gain | oracle | oracle − raw |
|---|---|---:|---:|---:|---:|---:|---:|
| noisy RTL | ce | 81.1 | 81.3 | 89.1 | 0.71 | 83.4 | 2.3 |
| | ccalign | 84.0 | 84.1 | 90.2 | 0.79 | 85.6 | **1.5** |
| | rxnull | 83.9 | 84.0 | 90.7 | 0.41 | 85.3 | 1.4 |
| USRP/Pluto | ce | 84.0 | 88.1 | 95.7 | 0.84 | 94.4 | 10.4 |
| | ccalign | 86.1 | 90.3 | 95.7 | **0.91** | 95.7 | **9.6** |
| | rxnull | 86.4 | 89.6 | 97.3 | 0.74 | 96.1 | 9.7 |
| clean RTL | ccalign | 91.4 | 91.7 | 95.0 | 0.96 | 94.4 | 3.1 |

Random-span control recovers 0.17–0.56 of the cc gain, versus 0.41–0.96 for the source span.
Truly held source RX (search checkpoints, 4 fitting RX): rtl_6 raw 86.8 / oracle 90.7;
Pluto raw 85.0 / oracle 90.5. For held Pluto, `ccproj` *lowers* accuracy (83.1): with one USRP-type
fitting receiver the source span does not cover it.

## 2. Input physics (`rx_physics.py`)

- Strict CIS = log10 |S(f,n+1)/S(f,n)|² over adjacent STFT frames. A receiver's LTI response cancels,
  so what is left is additive packet-level fluctuation and non-LTI log-domain patterns. The torch port
  reproduces the cache to 5e-6.
- Per RX: between-class variance 0.0022–0.0024 everywhere; within-class variance b200mini .0064,
  b210 .0066, pluto .0064, rtl_9 .0073, rtl_6/7 .0077, rtl_1 .0110, rtl_2 .0160, rtl_3 .0180, rtl_5 .0213.
  Noisy group fixed from these numbers before any P2E fit: rtl_1/2/3/5 (> 0.010).
- Estimated in-band SNR is 41–44 dB on all RX and does not order like the within-class variance, so the
  RTL excess is not thermal noise. A per-packet CFO estimator was tried and rejected as unreliable.
- Pluto's input offset is 88% TX-independent and the largest of all RX (2× source RMS).

## 3. Source-only predictors (`audit/source_predictors.csv`)

Spearman within fold; "within" = family-demeaned with a within-family permutation p (2000 perms).

| stage / target | src val acc | src q10 | κ | src CE variance (REx) | held acc → outer |
|---|---|---|---|---|---|
| search f0, outer | .55 (within .67, p<.001) | .69 (partial .50) | −.37 (ns) | −.41 (ns) | **.66** |
| search f1, outer | .91 | .58 (partial .10) | −.14 | −.09 | **−.17** |
| search f1, held = Pluto | −.14 | −.01 | .22 | .41 (p=.08) | — |
| formal f0 / f1, outer | .63 / .51 | .59 / .45 (partial .22/.25) | −.59/−.44 (within −.43/−.42, p<.001) | ns | — |

In-domain accuracy and the closest-competitor q10 margin are the useful source signals; κ adds
within-family information only for formal checkpoints; the REx quantity adds nothing.

## 4. Chain E1–E4 (`run_p2e_chain.py`, decision rules in its docstring, fixed before any fit)

E1: 200 LORO fits (5 held source RX × 2 folds × 8 arms × grid), epoch chosen on fitting-RX val packets,
held RX evaluated only at that epoch. E2: 80 formal fits, 5 seeds. 280 fits, 10.8 job-h in total.
Noise profile = pixel-wise max over fitting RX of (within-class variance − cleanest fitting RX), scale λ.

LORO held accuracy at λ* (source only, single seed):

| held | ce | ccalign | noise | noiseiso | ccanoise | sig | sigrand |
|---|---:|---:|---:|---:|---:|---:|---:|
| rtl_3 (noisy) | 74.8 | 81.6 | **82.0** | 80.4 | 80.6 | 79.4 | 79.0 |
| rtl_1 (noisy) | 83.9 | 86.4 | 85.6 | 83.9 | **87.4** | 82.0 | 85.0 |
| rtl_2 (noisy) | 80.4 | 79.6 | **83.6** | 81.2 | 83.5 | 79.3 | 80.8 |
| rtl_5 (noisy) | 77.5 | 78.8 | **82.4** | 74.9 | 82.2 | 74.5 | 78.1 |
| b210 | **80.8** | 80.6 | 76.4 | 74.9 | 78.2 | 73.4 | 78.3 |
| pluto | **80.2** | 71.2 | 71.4 | 75.6 | 62.4 | 66.2 | 76.0 |
| rtl_7 | **88.4** | 85.7 | 81.4 | 82.6 | 87.6 | 84.9 | 84.6 |

G1 (beat ccalign's LORO mean in both folds): no arm. ccalign itself is below ce in fold 1 LORO (79.4 vs 80.7).

Outer RX, 5 seeds (acc ×100, fold 0 / fold 1):

| arm | acc | noisy RX | clean RX | worst RX |
|---|---|---|---|---|
| ce | 82.70 / 86.02 | 81.08 / 81.96 | 83.78 / 88.73 | 73.3 / 79.4 |
| ls | 83.31 / 87.21 | 80.42 / 82.50 | 85.24 / 90.34 | 77.1 / 79.8 |
| ccalign | **85.24 / 88.71** | 83.41 / 85.23 | **86.45 / 91.03** | 78.4 / 83.6 |
| noise | 83.23 / 88.10 | 84.18 / **86.17** | 82.59 / 89.39 | 73.3 / **84.8** |
| noiseiso | 82.72 / 86.05 | 81.62 / 83.61 | 83.46 / 87.67 | 73.6 / 82.7 |
| ccanoise | **85.27** / 86.63 | **84.95** / 85.18 | 85.48 / 87.59 | 78.5 / 83.7 |
| sig | 81.55 / 86.68 | 79.87 / 82.16 | 82.67 / 89.69 | 70.7 / 79.8 |
| sigrand | 82.38 / 86.42 | 81.34 / 82.98 | 83.07 / 88.72 | 72.9 / 81.2 |

Pairwise (8/10 rule, `eval/pairwise.csv`): noise > ce on noisy 10/10, on acc 8/10; noise > noiseiso on
acc 8/10 and on noisy 9/10; ccanoise > ce on noisy 10/10. Neither noise nor ccanoise beats ccalign
(acc 3/10, 4/10); against ccalign their clean-RX deltas are −3.86/−1.64 and −0.97/−3.44 pp.
sig vs sigrand: 5/10. **Decision: STOP** (`decision_e4.json`).

Oracle ceiling on the noisy RX (`audit_p2e/`): ce 83.6, ccalign 85.8, noiseiso 84.9, ccanoise 87.0,
**noise 87.6**. The noise arm's noisy-RX gain is in the features, not in the head.

**Label-free calibration** (`tta/`, a deployment setting, not DG): centring pre-head features with N
unlabeled target packets gives ccalign +2.1 / 0.0 pp at N = 50 (balanced). With a Dirichlet(1) class prior
it gives +0.1 / −1.8 pp, so it is not robust to class imbalance in the calibration packets.

## 5. E6: consistency instead of replacement (registered before any E6 fit)

Hypothesis: training CE on noisy inputs replaces the training distribution, and that, not noise
invariance, costs clean-RX accuracy. Arms: `ncons` = CE on clean packets + λ·KL(p_clean.detach() ‖
p_noisy), noisy view = calibrated noise at fixed scale 2 on every packet; `ccancons` = the same plus the
ccalign term; control `nconsiso` (flat profile). λ ∈ {0.3, 1, 3} by LORO; 5 seeds. GO_METHOD iff ncons or
ccancons beats ccalign and nconsiso (8/10) with clean-RX seed-mean Δ vs ccalign ≥ −0.5 pp in both folds;
otherwise STOP_NOISE_LINE. Results: `decision_e6.json` (pending at the time of writing).

## 6. WiSig protocol revision (no fit run)

- Head: 8 classes (TX0–7), done in `wisig_cost_probe.py`; CE and every aux arm carry
  `model.regularization()`.
- Selection: nested leave-one-source-RX-out inside the 14 source RX (grouped 7 folds of 2 RX), epoch on
  fitting-RX validation positions [120,160); internal 10 RX for confirmation only; final 8 RX, date 3 and
  the equalised variant stay sealed.
- λ: before any search, report ‖∇(λ·L_aux)‖/‖∇L_CE‖ separately for encoder and head on 20 source batches
  and extend the grid down by a decade if the ratio exceeds 1 at λ = 0.3 (the 3-epoch probe at λ = 1 left
  rxlogit's val CE flat at 2.09 while CE reached 1.57).
- The noise mechanism must be re-derived for WiSig's raw-IQ input (no CIS); the Shen profile does not transfer.
- Budget from the measured probe: CE 0.43 s/epoch, rxlogit 2.22 s/epoch. Nested LORO for 4 arms × 3 λ ×
  7 folds × 100 epochs at ≤2.22 s/epoch ≈ 5.2 job-h, plus formal 4 arms × 5 seeds × 100 epochs ≈ 1.2
  job-h, ≈ 6.4 job-h (≈1.6 device-h on 4 GPUs at 2 jobs/GPU). Hard stop at 2× (13 job-h).

## 7. Files

`scripts/receiver_p2/`: `p2e_audit.py`, `p2e_fidelity.py`, `p2e_tta.py`, `rx_physics.py`, `train_p2e.py`,
`run_p2e_chain.py`, `eval_p2e.py`, `run_sharded.py`. Reports here: `audit/` (257 ckpts),
`audit_p2e/` (80 P2E ckpts), `fidelity/`, `tta/`, `eval/`, `loro_runs.csv`, `loro_grid.csv`,
`lambda_p2e.json`, `decision_e4.json`. Checkpoints in `outputs/receiver_p2e/` (not in git).
