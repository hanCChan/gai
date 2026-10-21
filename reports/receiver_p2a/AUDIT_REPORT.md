# P2A audit of rxnull — verdict **MODIFY_OBJECTIVE** — 2026-10-21

Protocol `docs/receiver_p2/PROTOCOL_P2A_AUDIT.md` (eed22e8, committed before the audit ran).
Inference only, 90 checkpoints (ce R0/R1 6, P1 formal 48, P2 formal 36), source_train packets for all
source quantities; outer RX used only for the diagnostic columns (repeatedly-developed data).
Code `scripts/receiver_p2/audit_p2.py`; outputs `audit_models.csv`, `audit_arms.csv`,
`predictiveness.csv`, `reparam.json`, `verdict.json`.

## 0. Verdict (pre-declared rules)

| guard | result | numbers (fold 0 / fold 1, seed means) |
|---|---|---|
| G1 κ(rxnull) < κ(ce), κ(ls) | **PASS** | κ: ce .0088/.0111, ls .0110/.0120, rxnull **.0038/.0040** (−57%/−64% vs ce) |
| G2 n_row(rxnull) < n_row(ce) | **PASS** | .0104→.0048 / .0130→.0048 |
| G3 share of log-ρ drop from row-energy decrease ≥ 0.5 | **FAIL** | **0.46 / 0.32** |
| G4 eff. rank ≥ 0.8×ce, src acc ≥ ce−1pp | PASS | eff. rank 7.00/6.80 vs 7.49/7.38; src val acc .906/.886 vs .877/.872 |
| G5 audit correctness (reparam, float64) | PASS | Δsoftmax ≤ 3e-14, argmax identical, |Δκ|/κ ≤ 2e-8; ρ moves 0.61→0.96 (s=¼) →0.006 (s=16) |

κ cross-check (logit-space vs A·D in feature space): max relative difference 2.8e-7.

**MODIFY_OBJECTIVE**: rxnull truly reduces the function-level, decision-relevant RX displacement, but
most of its ρ reduction is denominator inflation (more RX displacement energy in null(A)). ρ cannot be
presented as the mechanism quantity, and the training objective has an exploitable degenerate path.

## 1. What the numbers say

| fold 0 / fold 1 | ce | ls | rxnull | cmean |
|---|---|---|---|---|
| ρ (P2 objective) | .62/.62 | .39/.39 | .12/.03 | .37/.28 |
| n_null = null-space RX energy / decision-space class sep. | .0059/.0081 | .0179/.0204 | **.0609/.2051** | .0041/.0041 |
| E_D = mean ‖D‖² (absolute) | .31/.38 | .17/.20 | **.82/2.88** | .08/.05 |
| κ (function level) | .0088/.0111 | .0110/.0120 | .0038/.0040 | **.0020/.0011** |
| κ_out (realized, outer RX) | .047/.024 | .086/.046 | .033/.015 | .035/.010 |
| outer acc | 82.67/85.45 | 82.95/87.20 | **85.21/89.46** | 84.20/87.23 |

1. **Denominator inflation is large**: rxnull multiplies null-space RX displacement 10×/25× and the total
   per-RX class-mean displacement energy 2.6×/7.6× vs ce. The reviewer's toy example happens in practice.
2. **The decision-relevant part also shrinks**: κ −57%/−64%; realized unseen-RX logit displacement
   κ_out −30%/−36%; source κ predicts outer κ_out across the 45 models of each fold (Spearman .64/.71).
   rxnull is therefore not purely cosmetic.
3. **κ does not explain rxnull's lead on its own**: cmean has the lowest κ (.0020/.0011) yet is
   1.0/2.2 pp below rxnull. Across 45 models per fold, Spearman with outer acc: κ −.46/−.34, ρ −.50/−.49,
   n_row −.51/−.39, src_val_acc +.56/+.40; partial (given src acc) only −.21/−.29 for κ. No source
   quantity is a strong standalone predictor.
4. **The P2 explanation of label smoothing is withdrawn**: LS lowers ρ (.62→.39) entirely through
   null-space inflation (n_null 3×) while its κ is higher than ce, and its realized outer displacement is
   +81%/+95% vs ce. Whatever makes LS work, it is not reduced decision-relevant RX displacement.
5. **The P2 statement that margin/tail "raise exposure" stands in κ too** (κ .0119/.0110 margin, ≥ ce),
   consistent with P1's error transfer, but only as an association.

## 2. Errata to the P2 report (P2 pre-registration and results files left unchanged)

E1 9/128 is the isotropic reference expectation for rank 9, not a floor (ρ=0 for null(A) displacements).
E2 ρ is not invariant to function-preserving reparameterisation; "ρ equals all argmax damage" is wrong.
Valid statement: σ_min⁺(A)²‖P_row D‖² ≤ ‖AD‖² ≤ σ_max(A)² ρ ‖D‖².
E3 ρ is sign-blind; margin-enlarging and margin-shrinking displacements are penalised alike.
E4 "LS works by lowering exposure" is refuted by κ (§1.4), not just downgraded.
E5 "rxnull reaches the isotropic floor 0.07" → "ρ is driven to about the isotropic reference, mostly by
null-space inflation".

## 3. Other checks requested by the review

- No outer-RX access in `train_p2.py`, `select_lambda_p2.py`, `queue_p2.py` (static check, asserted).
  λ was selected on the inner validation RX (a source RX of each fold).
- Same 20 epochs, seeds, balanced schedule, optimizer, clipping and last-40 val-CE checkpoint rule for
  all arms. The ce anchors carry the R0/R1 head weight decay 1e-4 (bitwise-equal to the P1 trainer); P1
  ls and P2 arms do not. rxnull best epochs are often 20 (horizon-limited, same horizon for all).
- Gains are not concentrated in a single RX: rxnull > ls on 9/10 outer RX, > ce on 10/10 (P2 per_rx).
  Worst RX fold 0 (Pluto) rxnull 76.4 vs ls 76.7 — no gain on the worst RX in fold 0.

## 4. Pre-registered consequence

The minimal fix fixed in P2A, rxlogit = CE + λ·κ̂_batch (function-level, invariant to null inflation and
to scaling W or h), is run on Shen under P2 rules before any WiSig fit (amendment P2B). Because §1.3 shows
κ alone may not account for rxnull's lead, P2B is a real test, not a formality.
