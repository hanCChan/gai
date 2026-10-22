# P2E: why cross-receiver methods keep plateauing — a two-dataset, label-free analysis — 2026-10-22

Base: `experiment/p2-rx-null` @ c68798b. Datasets: Shen (10 registered RX) and WiSig (14 source RX,
10 internal RX for diagnosis only; final 8 RX, date 2/3 and the equalised variant never read).
`train_p2e.py` reproduces the registered R0/R1 CE checkpoints bitwise (6/6).
**Read with**: `ERRATA_P2E.md` (independent code audit; it corrects several statements of the first
version of this report, most importantly A1) and `ERRATA_P2D.md`.

## 0. Summary

1. **Method search on Shen: STOP (E1–E4 and E6, 400 fits).** No arm beats class-conditional alignment
   (ccalign) under the pre-registered rule (5 seeds, 8/10 paired, both folds).
   - Calibrated input noise improves the noisy RTL receivers (+3.1/+4.2 pp vs ce, 10/10) and nothing
     else.
   - E6 tested whether its clean-RX cost came from training CE on noisy inputs. Making it a
     consistency term instead (CE on clean packets) kept the noisy-RX gain and made the clean-RX cost
     larger, so the hypothesis is refuted.
2. **Corrected error decomposition (label never used at test).** On both datasets, the part of
   unseen-receiver error that can be removed without changing the features is a class-conditional
   mean shift. It is removed by re-estimating class means in a Gaussian (LDA) head:
   - WiSig: LDA with source means 53.1% → LDA with target class means 88.4% (oracle head 94.2%).
   - Shen USRP/Pluto: 86.6 → 94.7 for ccalign.
   - Shen noisy RTL: +0.4 pp only. That error is information loss (oracle head 85.8), not shift.
3. **What generalises and what does not.**
   - General (both datasets): the shift direction lies mostly in the per-class span of source-RX shifts
     (share of the recoverable gain: WiSig 0.66–0.80 vs 0.18–0.41 for random spans of equal dimension;
     Shen USRP/Pluto 0.80–0.98 vs 0.16–0.53).
   - General: the coefficients inside that span are class-specific. A single receiver vector shared by
     all classes (eigen-receiver) recovers little (WiSig 53 → 56–58).
   - **Shen-only**: the noise floor. WiSig raw-IQ within-class variance is identical across receivers
     (CV 0.00). The calibrated-noise mechanism is therefore dataset-specific and is not a general method.
4. **Why target-free methods keep plateauing.** The unseen receiver moves each class mean by an amount
   that is class-specific, unknown, and not predictable from 4–14 source receivers.
   - Without target labels the shift is not identifiable: unlabeled EM, eigen-receiver EM and centring
     all fail on WiSig (≤ +1.4 pp, often negative), and help only ~2 pp on Shen.
   - The only target-free lever left is invariance (shrinking the shift span), and that span overlaps
     the identity directions. ccalign sits at that trade-off; every further invariance term (noise,
     consistency, null-space, margins) paid for itself with identity (§4).
5. **What works on both datasets: a few labelled packets per transmitter at the new receiver.**
   - WiSig official CNN: 1 / 5 / 10 shots → 67.8 / 82.0 / 85.1 % (raw 53.5).
   - Shen ccalign: 87.0 → 88.6 / 89.8 / 90.1.
   - Done with source covariance + target class prototypes. Source-span priors add ≤ 1.3 pp at 1 shot
     and nothing beyond. Head fine-tuning is worse than prototypes.

## 1. Method search (Shen)

Chain `run_p2e_chain.py`, rules fixed in its docstring before any fit; nested leave-one-source-RX-out
(LORO) selection with epochs chosen on fitting-RX val packets; 5 seeds; 280 (E1–E4) + 120 (E6) fits.

Outer RX, acc ×100 (fold 0 / fold 1); noisy = rtl_1/2/3/5:

| arm | acc | noisy RX | clean RX | vs ccalign acc (n/10) |
|---|---|---|---|---|
| ce | 82.70 / 86.02 | 81.08 / 81.96 | 83.78 / 88.73 | 1 |
| ls | 83.31 / 87.21 | 80.42 / 82.50 | 85.24 / 90.34 | 1 |
| **ccalign** | **85.24 / 88.71** | 83.41 / 85.23 | **86.45 / 91.03** | — |
| noise | 83.23 / 88.10 | 84.18 / **86.17** | 82.59 / 89.39 | 3 |
| noiseiso | 82.72 / 86.05 | 81.62 / 83.61 | 83.46 / 87.67 | 0 |
| ccanoise | **85.27** / 86.63 | **84.95** / 85.18 | 85.48 / 87.59 | 4 |
| sig / sigrand | 81.55 / 86.68 · 82.38 / 86.42 | | | 1 · 0 |
| ncons (E6) | 82.56 / 87.01 | 83.86 / 86.00 | 81.69 / 87.67 | 1 |
| ccancons (E6) | 82.60 / 85.23 | 84.09 / 84.37 | 81.61 / 85.80 | 1 |
| nconsiso (E6) | 82.25 / 86.03 | 80.53 / 82.72 | 83.40 / 88.24 | 1 |

- noise vs ce on noisy RX: 10/10, +3.11/+4.21. ncons vs ce on noisy RX: 9/10, +2.78/+4.05.
- Clean RX vs ce: noise −1.19/+0.66 (5/10); ncons −2.09/−1.05 (3/10). Against ccalign the clean-RX
  deltas are larger because ccalign itself gains on clean RX (audit A6).
- E6 verdict: STOP_NOISE_LINE.
- Caveats from the audit:
  - Controls were not energy-matched (A5).
  - Fold-1 combined arms used ccalign λ = 1 instead of 3 (A4).
  - The noise idea came from outer-RX diagnostics (A8), so outer RX are development data here.
- LORO (source only) ranks the arms against outer at ρ = 0.43 / −0.10. With the Pluto-held fold, ce
  beats ccalign on LORO (80.7 vs 79.4) while ccalign wins on outer by 2.7 pp.

## 2. Error decomposition, corrected (`generality/`, `p2e_generality.py`)

For each checkpoint × unseen RX, using features h and the frozen head, accuracy on the odd half of
each class, with statistics from the even half:
- `raw`
- `ca`: label-free RX centring.
- `lda_src`: Gaussian head with source class means and pooled within-(class,RX) covariance.
- `lda_cc`: the same with target class means.
- `lda_ccproj`, `lda_ccrand`: the target shift kept only inside the per-class source span, or inside
  a random span of equal dimension.
- `lda_eig`: one receiver coefficient vector shared across classes.
- `oracle`: linear head refit on the target's even half.

The label never enters test-time decoding (regression test `--dataset selftest`). The leaky first-version
`cc` is kept only as `acc_cc_leak` for comparison.

| dataset / group / model | raw | ca | lda_src | lda_cc | proj share | rand share | lda_eig | oracle | (leaky cc) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WiSig internal RX / official ce | 54.9 | 56.4 | 53.1 | **88.4** | 0.74 | 0.35 | 57.7 | 94.2 | 94.5 |
| WiSig internal RX / swad | 51.3 | 52.5 | 51.2 | 86.8 | 0.80 | 0.39 | 55.6 | 93.4 | 90.6 |
| WiSig selection RX / official ce | 39.8 | 40.4 | 35.1 | 84.0 | 0.75 | 0.19 | 37.0 | 89.4 | 86.8 |
| Shen USRP/Pluto / ccalign | 85.8 | 89.7 | 86.6 | **94.7** | 0.98 | 0.47 | 90.9 | 95.6 | 95.3 |
| Shen USRP/Pluto / ce | 83.7 | 87.9 | 84.2 | 93.5 | 0.85 | 0.23 | 87.7 | 94.4 | 96.1 |
| Shen noisy RTL / ccalign | 84.0 | 84.1 | 84.5 | 84.8 | — | — | 84.7 | 85.8 | 89.4 |
| Shen noisy RTL / noise | 85.4 | 85.5 | 86.4 | 86.5 | — | — | 86.6 | 87.6 | 88.9 |
| Shen clean RTL / ccalign | 91.6 | 91.8 | 92.9 | 94.2 | 0.98 | 0.37 | 93.8 | 94.8 | 94.1 |

- WiSig DG baselines (GroupDRO, MLDG, MLDG-date, PCGrad, SWAD) do not beat CE on unseen RX
  (48.6–52.8 vs 54.9) although the information is in the features (oracle 93.4–94.5).
- A bias-only label-free correction (`acc_cclf`) is worse than raw on WiSig (45.1). When class means
  move, the decision directions must move too; that is why the LDA form is the correct measure.

## 3. Input level

- Shen Strict CIS (`input/shen_input.csv`, `p2e_analyses.py input`): between-class variance
  0.0022–0.0024 on every RX; within-class variance 0.0064 (USRP/Pluto) to 0.0213 (rtl_5). Band SNR is
  41–44 dB on all RX and does not order like the within-class variance.
- Outer-RX TX-independent offsets lie in the source offset span: coverage 0.49–0.99, against
  0.07–0.11 for a random subspace inside the source packet PCA span (`input/shen_offset_coverage.csv`).
- WiSig raw IQ (`generality/wisig_input_rawiq.csv`): within-class variance identical over RX (CV 0.00)
  and between-class variance varying 4.3×. In log power spectrum: within 4.8×, between 2.5×. **There
  is no Shen-like noise floor heterogeneity.**

## 4. Why P0–P2E kept failing (cross-phase synthesis)

| cause | evidence |
|---|---|
| **Identifiability.** An unseen receiver applies class-specific shifts; their coefficients are not predictable from the source RX and not estimable without labels | lda_cc ≫ lda_src on both datasets, but lda_eig ≈ lda_src on WiSig; unlabeled EM ≤ raw on WiSig (`eigenrx/`); no source statistic predicts held-Pluto accuracy (`audit/source_predictors.csv`) |
| **Invariance pays with identity.** The only target-free lever is to shrink the shift span, and the receiver and transmitter directions overlap | D phase: dominant RX axis carries 34.6% of TX energy (WiSig); E1/E6: noise invariance costs clean-RX accuracy, more so as a consistency term; P2A: rxnull lowered exposure by inflating the null space |
| **Each loss targeted one proxy of one component.** κ, ρ, margins, worst shift and tail ratios are all functions of the first moment that ccalign already constrains | P2B/P2D: rxlogit, rxnull, rxccp ≈ ccalign; P2D tail audit was outer-side |
| **Selection could not see the target failure.** λ selection on one held RX, which was anti-correlated with outer in fold 1 | ρ(held, outer) = −0.17 in fold 1; LORO arm ranking vs outer 0.43 / −0.10 |
| **Resolution and multiplicity.** Arm gaps ~1 pp with seed SD 0.8 pp; ~40 arms on the same 10 outer RX; 20-epoch horizon | ERRATA_P2D E5; 66/138 fits at epoch ≥ 18 |
| **Specification and implementation errors** (wasted rounds, not the root cause) | rxnull null inflation, rxgeo amin, rxccp double shift, rxcant pooled surrogate, P2E label-leaky cc (ERRATA_P2D, ERRATA_P2E) |

The first row is decisive. It explains why every target-free loss converges to the same level and
why the gap is ~8 pp on Shen USRP/Pluto but ~35 pp on WiSig: same mechanism, different shift size.
It also predicts what does work (§5).

## 5. What does work, on both datasets (`fewshot/`, `p2e_fewshot.py`, `eigenrx/`)

Unlabeled calibration (N target packets, no labels; `p2e_eigenrx.py`), all-packet transductive:

| | raw | ca | EM free | eigen-RX EM | eigen-RX EM + prior |
|---|---:|---:|---:|---:|---:|
| WiSig (formal, internal RX) | 50.3 | 50.9 | 53.0 | 47.4 | 46.9 |
| Shen ccalign | 85.2 | 87.5 | 89.5 | 87.2 | 87.0 |

Shen with N = 200 under a Dirichlet(1) class prior: ca 86.1, eigen-RX 87.0 (the most prior-robust).

Few-shot labelled re-enrolment (m packets per TX on the new receiver, 5 draws; test on the remaining
packets; WiSig: the 6 internal RX that have ≥ 14 packets for every TX):

| model | m | raw | proto | span prior | eig+res | head fine-tune |
|---|---|---:|---:|---:|---:|---:|
| WiSig official ce | 1 / 5 / 10 | 53.5 | 67.8 / 82.0 / 85.1 | 68.4 / 80.3 / 83.9 | 68.9 / 82.2 / 85.1 | 57.0 / 67.4 / – |
| WiSig resnet ce | 1 / 5 / 10 | 35.1 | 63.0 / 83.6 / 86.2 | 65.5 / 81.6 / 85.4 | 66.0 / 83.7 / 86.3 | 45.2 / 67.4 / – |
| Shen ccalign | 1 / 5 / 10 | 87.0 | 88.6 / 89.8 / 90.1 | 87.8 / 89.7 / 90.1 | 89.0 / 90.0 / 90.2 | 87.8 / 88.6 / – |
| Shen ce | 1 / 5 / 10 | 83.9 | 86.4 / 88.0 / 88.5 | 85.6 / 87.9 / 88.5 | 86.9 / 88.1 / 88.6 | 85.5 / 86.5 / – |

Re-estimating class prototypes in a Gaussian head is the effective, dataset-agnostic mechanism.
Source-receiver priors (span, eigen-receiver) add at most ~1 pp at 1 shot. That is consistent with §4
row 1: the span is learnable, the class-specific coefficients are not.

Novelty, stated plainly: prototype / LDA re-estimation and eigenvoice-style subspace adaptation are
known techniques. What P2E adds is (i) a label-free decomposition that separates recoverable shift from
information loss, (ii) two-dataset evidence that cross-receiver RFFI error is mostly class-specific
shift that target-free DG cannot identify, and (iii) the cost curve of re-enrolment. Whether that
supports a methods paper or a diagnosis-plus-protocol paper is a decision for the authors; no target-free
method has been found that beats ccalign.

## 6. WiSig protocol note (unchanged plan, no WiSig training run)

8-class head, nested LORO inside the 14 source RX, a gradient-ratio check before the λ grid, budget
≈ 6.4 job-h with a hard stop at 13 job-h (first version of this report, §6). New in this round:
- ccalign has not been run on WiSig.
- §2 predicts it can move at most the in-span part of a class-specific shift. The DG baselines already
  run there (GroupDRO / MLDG / SWAD / PCGrad) sit at or below CE.

## 7. Files

- Scripts (`scripts/receiver_p2/`): `train_p2e.py`, `run_p2e_chain.py`, `eval_p2e.py`,
  `p2e_audit.py` (its cc columns are leaky, A1), `p2e_generality.py` (label-free decomposition, both
  datasets, selftest), `p2e_eigenrx.py`, `p2e_fewshot.py`, `p2e_fidelity.py` (A2), `p2e_tta.py`,
  `p2e_analyses.py`, `rx_physics.py`, `run_sharded.py`.
- Reports here: `generality/`, `eigenrx/`, `fewshot/`, `input/`, `audit/` (+ `source_predictors.csv`),
  `audit_p2e/`, `fidelity/`, `tta/`, `eval/` (`pairwise.csv`, `pairwise_e6.csv`), `loro_*.csv`,
  `lambda_p2e*.json`, `decision_e4.json`, `decision_e6.json`.
- Checkpoints: `outputs/receiver_p2e/` (not in git).
