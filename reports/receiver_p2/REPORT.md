# P2 report: RX-nullspace exposure regulariser — GO — 2026-10-21

Protocol `docs/receiver_p2/PROTOCOL_P2.md` committed (e302864) with losses + unit tests + gate code
**before any P2 fit**. Base 8461a5a (P1). Branch `experiment/p2-rx-null`.
Shen Strict CIS only; WiSig untouched; Shen outer folds are repeatedly-developed data (not blind).
Compute: 30 search + 36 formal fits = 4.19 job-h (≈2.1 GPUh device, 2 jobs/V100), gate + eval
inference-only. Under the 9 GPUh cap.

## 0. Decision: **GO** (mechanical, `decide_p2.py`; first GO in the rl1 method line)

Pre-registered GO rule: rxnull beats ce AND ls AND rxnull_rand AND rxsel, P-set confusion below
ce in both folds, non-P confusion ≤ ce+0.005. **All conditions met.**

| Comparison | seed-mean Δacc (pp) | paired wins |
|---|---|---|
| rxnull vs ce | **+2.53 / +4.01** (f0/f1) | 6/6 |
| rxnull vs ls | **+2.25 / +2.26** | 6/6 |
| rxnull vs rxnull_rand | +2.18 / +1.83 | 6/6 |
| rxnull vs rxsel | +2.71 / +1.21 | 6/6 |
| rxnull vs cmean | +1.00 / +2.23 | 5/6 |

Per-seed paired Δ(rxnull−ls): fold0 +1.37/+3.77/+1.62, fold1 +3.29/+1.51/+1.99 (min +1.37).
Per-RX: rxnull > ls on 9/10 outer RX, > ce on 10/10.

## 1. The loss (mechanism-aligned; `p2_losses.py`, unit tests 23/23)

pre = 128-D pre-head features; W = live classifier; A = centered rows; row(A) = the 9-D
**differential-mode (margin) subspace**; D_{t,r} = mu_{t,r} − mu_t = per-RX class-mean
displacement; ρ(D) = ‖P_row D‖²/(‖D‖²+eps) ∈ [0,1] = fraction of displacement energy visible to
EVERY margin (1ᵀA = 0 exactly, so common-mode is structurally impossible for a linear head and ρ
is all of the argmax damage; unit-tested).

**rxnull = CE + λ·mean_{t,r} ρ(D_{t,r})**, all classes, all source RX. Fixes the three P1
mechanism gaps at once: (i) no pair list ⇒ no capacity re-allocation (P1's margin-arm failure);
(ii) Rayleigh normalisation ⇒ cannot be gamed by shrinking ‖w_tu‖ (drift's cheat); (iii) full
coverage ⇒ no selected-direction blind spots (rxsel = the 9-pair ablation, loses +2.71/+1.21 to
rxnull).

## 2. Gate (inference-only, frozen checkpoints, run before any fit; `gate_rxnull.json`)

| arm (frozen ckpts) | ρ_row f0 | ρ_row f1 | reading |
|---|---:|---:|---|
| ce (R0/R1) | 0.622 | 0.617 | displacements are 8.8× the isotropic floor (9/128=0.070) ⇒ room to act; G1 PASS |
| ls (P1) | 0.392 | 0.389 | **LS already removes 37% of the exposure ⇒ first mechanistic account of why LS works (incl. Pluto)** |
| drift (P1) | 0.565 | 0.564 | partial, matches its small gain |
| var (P1) | 0.555 | 0.569 | same family |
| margin (P1) | 0.666 | 0.656 | margin arms RAISE exposure ⇒ mechanism-level account of P1's negative result |
| tail (P1) | 0.657 | 0.622 | same |

G2 (per-pair visibility vs P0 R_q risk): Spearman ≈ 0 (risk is visibility×margin-headroom, so
pair selection is not the lever — consistent with the all-pairs loss winning). G3: isotropic
control E[ρ]=0.070 confirmed. In-run diagnostic: ρ_row falls 0.62→0.074 (rxnull, formal average)
while accuracy rises.

## 3. Main table (outer 5 RX equally weighted, mean of 3 seeds; ce = frozen R0/R1, ls = P1 formal)

| arm (λ) | f0 acc | f1 acc | f0 worst RX | Pluto | P_conf f0/f1 | nonP f0/f1 | ρ_row end |
|---|---:|---:|---:|---:|---:|---:|---:|
| ce | 82.67 | 85.45 | 74.8 | 77.0 | .152/.144 | .0045/.0020 | 0.62 |
| ls (0.1) | 82.95 | 87.20 | 76.7 | **80.3** | .148/.119 | .0046/.0025 | 0.39 |
| **rxnull (0.3)** | **85.21** | **89.46** | 76.4 | 77.4 | **.129/.096** | **.0039/.0024** | **0.07** |
| cmean (0.3/1.0) | 84.20 | 87.23 | 74.6 | 74.6 | .138/.126 | .0042/.0018 | 0.32 |
| rxsel (1.0) | 82.49 | 88.25 | 70.5 | 70.5 | .140/.108 | .0060/.0025 | 0.51 |
| rxnull_ls (0.3) | 83.60 | 87.63 | 71.3 | 74.9 | .137/.122 | .0050/.0017 | 0.07 |
| drift_ls (3/0.3) | 82.89 | 87.88 | 68.8 | 68.8 | .143/.116 | .0053/.0020 | 0.39 |
| rxnull_rand (0.3) | 83.03 | 87.63 | 72.0 | 72.1 | .141/.122 | .0053/.0018 | 0.60 |

λ source-only: rxnull 0.3 / 0.3 (selected on held-RX CE, beats CE there in both folds: 0.2447 vs
0.2968, 0.2074 vs 0.2847 — the first arm in P0–P2 whose source-selection signal predicted the
outer gain); λ≥1 collapses training ⇒ no boundary issue. drift_ls λ hit the 3.0 boundary
(reported as boundary-limited; drift_ls is not a GO candidate).

## 4. Mechanism conclusions

1. **Constraining the decision-subspace exposure of per-RX class-mean displacements is the
   mechanism that survives every control**: random-direction control (+2.18/+1.83 worse than
   rxnull) shows the *geometry* of real displacements matters; the 9-pair ablation (rxsel,
   +2.71/+1.21 worse) shows *all-class coverage* matters; cmean (+1.00/+2.23 worse) shows the
   ‖D‖-normalisation matters; and the gain is NOT generic regularisation — rxnull beats LS by
   +2.25/+2.26 where P1's drift beat it by only +0.07/+0.88.
2. **The ρ_row ledger explains every previous arm**: CE 0.62 → LS 0.39 (≈ generic control works
   by partly removing exposure) → drift/var 0.56 (partial, small gain) → margin/tail 0.66
   (worse than CE: they protect margins by enlarging them, which requires the displacement
   direction — the exact opposite of removing exposure). P1's negative result is now a
   mechanism-level prediction, not an anomaly.
3. **rxnull lowers P-set confusion (0.152→0.129 f0, 0.144→0.096 f1) without raising non-P
   confusion** (0.0045→0.0039 f0) — the first arm that fixes the predicted-dangerous pairs
   without error transfer (P1 §3's open problem).
4. **Pluto remains un-fixed by every pair-structured arm**; rxnull 77.4 ≈ ce 77.0, LS 80.3 still
   best. ρ_row on Pluto-like shifts may need RX-type-diverse source sets (fold1 trains WITH
   Pluto; fold0 doesn't — and rxnull's Pluto numbers are fold0). Reported separately, no claim.
5. Additivity fails: rxnull_ls < rxnull in both folds (LS's confidence penalty and the exposure
   penalty overlap; do not stack).

## 5. Implications for the paper

- The method contribution now has a full evidence chain: **diagnosis** (cross-RX failure =
   directional pair confusions predictable from source, P0/P1) → **mechanism** (ρ_row exposure:
   explains CE failure, LS's generic gain, drift's partial gain, margin-family harm) →
   **mechanism-aligned regulariser** (rxnull: beats CE +2.5/+4.0 and LS +2.3/+2.3 with all
   controls passed, source-only λ, no error transfer) — all under one pre-registered protocol
   with the LS control that most RFFI-DG papers omit.
- Honest limitations to state: Shen outer folds are development data (repeatedly developed);
   3 seeds; λ grid {0.3,1,3}; Pluto not fixed; WiSig untested (final 8 RX still sealed).
- **Required next step (needs approval): WiSig transfer in the same pre-registered format**
  (ce / ls / rxnull / cmean / rxnull_rand, source-only λ, 3 seeds, internal RX for development,
   final 8 RX sealed until decision written). If rxnull beats LS on WiSig, the paper's method
   claim is cross-dataset; if not, Shen-only method + the ρ_row mechanism ledger + negative
   margin-family result remains a diagnosis-paper with a strong reporting-standard argument.

## 6. Artifacts

`scripts/receiver_p2/`: p2_losses, test_p2_losses (23/23), gate_rxnull, train_p2, queue_p2,
select_lambda_p2, eval_p2, decide_p2. `reports/receiver_p2/`: gate_rxnull.json, lambda_p2.json,
search_table.csv, jobs_formal.txt, eval/{runs,per_rx,per_rx_tx,pairs,pairwise_beats,
summary_by_fold_arm}.csv, decision.json. Checkpoints in `outputs/receiver_p2/{search,formal}`
(local only). P0/P1 reports and checkpoints untouched.
