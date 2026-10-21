# P1 report: P0 review fixes + mechanism confirmation (Shen) — 2026-10-21

Protocol `docs/receiver_p0/PROTOCOL_P1.md`, committed (d7f35ab) before any search fit. Base c79a59c.
Shen Strict CIS only; WiSig not touched; Shen outer folds are repeatedly-developed data (not blind).
Compute: 28 search + 48 formal + 3 replay fits. The per-job wall time sums to 5.84 h, but two jobs shared each V100,
so device occupancy is ~3 GPUh. **The job-hour sum exceeds the 4.5 GPUh cap written in the protocol (the protocol did not define the unit), stated here explicitly.**

## 0. Decision: **MODIFY** (mechanical, `decide_p1.py`)

| Question | Answer |
|---|---|
| Is the P0 drift gain real? | It replicates: λ chosen on the source held-out RX = 1.0 (same as P0), +0.35 / +2.63 pp vs CE (5/6 seeds). |
| Where does the gain come from? | **Mostly generic regularisation**: label smoothing 0.1 (zero pair information) already gives +0.28 / +1.75 pp, 6/6 seeds. drift over LS is only **+0.07 / +0.88 pp** (5/6 meets the rule, but fold 0 is well inside seed noise; CE seed SD 0.63 pp). |
| Is pair selection useful? | drift beats drift on 9 random pairs, +0.18 / +1.10 pp (5/6); drift_rand does not beat LS. Weak evidence. |
| Is drift = projection variance? | Not equivalent by the rule (+0.97 / +0.40, 4/6), and var does not beat drift either. var crashes on Pluto (69.9 %), so the higher-order terms in drift (asymmetric softplus) are probably not negligible. Unresolved. |
| Correctly implemented margin protection (A3/A4)? | **Both hurt accuracy**: margin −1.12 / +0.08, tail −0.84 / +0.59 vs CE, and both lose to LS by 1.1–1.7 pp. They do raise the normalized margin on the P set (1.04→1.55–1.63) and lower P confusion a little, but non-P confusion rises 0.0045→0.0064–0.0069 = **shifting errors to other pairs**. |
| Does worst-RX aggregation beat plain margin? | Relatively yes, tail > margin +0.28 / +0.51 (5/6), but both are below CE/LS. tail + drift does not beat drift. **The A4 innovation module has no basis.** |
| Pluto | CE 77.0, LS **80.3**, drift 76.5, tail 77.3, var 69.9. No pair loss fixes it; drift is 3.8 pp below LS on Pluto. |

Pre-registered GO requires tail / tail_drift to beat ce, ls, margin and the random set: none met. MODIFY: drift is the only arm that beats both ce and ls (rule satisfied, but its margin over LS is near zero in fold 0).

## 1. Review findings (code verified, `test_p1_losses.py` 16/16)

1. The P0 pair loss minimised logsumexp(batch-min margin): **wrong sign**, gamma unused (test: larger margin → larger loss). The P0 pair results (λ=1 collapse, λ=0.1 +2.18 pp) say nothing about margin protection. The latter is itself evidence that "any regularisation on CV1 gives ~2 pp".
2. drift = softplus(γ) + κ·Var_r(Δ) + O(Δ³) (measured excess / variance term agree to <5 %). It **penalises the margin-enlarging direction** (Δ>0) and can be lowered by shrinking ‖w_t−w_u‖. In training, its gradient is ~5–10 % of CE. It must not be called "harmful-drift protection".
3. Fair risk comparison (`p1_stats.py`, TX-relabel permutation that preserves shared-TX dependence):

| Aggregation | fold | ρ(R_q) | ρ(R_conf) | p (TX perm) | top-10 recall |
|---|---|---:|---:|---:|---:|
| CE only | 0 | 0.726 | 0.753 | 5e-5 | 0.8 / 0.8 |
| CE only | 1 | 0.673 | 0.661 | 5e-5 | 1.0 / 1.0 |
| 5-method mean (P0-C) | 0 / 1 | 0.736 / 0.718 | 0.791 / 0.805 | 5e-5 | 0.9 / 0.9 |

   Receiver level (10 outer RX): R_q beats R_conf on only 2/10 RX (CE only) and 1/10 (5 methods, sign test p=0.02). **Predictability holds; the margin risk has no advantage over plain source confusion as a selector.** The pair sets chosen by R_q and R_conf are almost identical (fold 1 identical, fold 0 differs by one pair), so no separate R_conf training arm was run.
4. P0's "collapse count 8.3→5.0" counts (outer RX × P pair) cells with conf>0.3, not the number of pairs.

## 2. Main table (outer 5 RX equally weighted, mean of 3 seeds; ce = frozen R0/R1, bitwise equal to the P1 trainer)

| arm (λ) | f0 acc | f1 acc | f0 worst RX | f0 Pluto | P conf f0/f1 | non-P conf f0/f1 |
|---|---:|---:|---:|---:|---:|---:|
| ce | 82.67 | 85.45 | 74.8 | 77.0 | .152/.144 | .0045/.0020 |
| ls (0.1) | 82.95 | 87.20 | 76.7 | **80.3** | .148/.119 | .0046/.0025 |
| drift (1.0/1.0) | **83.02** | **88.08** | 75.1 | 76.5 | .143/.115 | .0051/.0020 |
| drift_rand | 82.84 | 86.97 | **78.2** | 80.2 | .158/.126 | .0036/.0021 |
| var (1.0/0.3) | 82.05 | 87.67 | 69.9 | 69.9 | .138/.113 | .0069/.0027 |
| margin (0.3/0.3) | 81.55 | 85.53 | 75.0 | 75.7 | .142/.125 | .0069/.0039 |
| tail (0.3/0.1) | 81.83 | 86.04 | 75.9 | 77.3 | .144/.125 | .0064/.0034 |
| tail_drift | 82.20 | 87.57 | 75.6 | 77.0 | .137/.114 | .0067/.0026 |
| tail_rand | 81.98 | 87.40 | 72.6 | 75.7 | .155/.118 | .0051/.0024 |

λ was selected entirely on the inner source held-out RX (`lambda_p1.json`). Note that on held-out-RX CE, drift (fold 0) did not beat CE, and margin and tail were both worse than CE. **The source-only selection signal had already predicted that the margin family would not help.**

Per seed (pp, acc): fold 0 drift 81.24 / 83.09 / 84.74 vs LS 83.43 / 83.04 / 82.39. Fold 1 drift 88.29 / 86.87 / 89.07 vs LS 87.04 / 86.67 / 87.89.
RX×seed cells: drift>LS 22/30 (mean +0.47), drift>CE 26/30 (+1.49), LS>CE 17/30 (+1.02).

## 3. Mechanism conclusions

1. **About 2/3 of the P0 drift gain on CV1 is generic regularisation** (LS +1.75 of 2.63). The pair-specific increment is ≤0.9 pp and absent on Pluto.
2. **Directly raising hard-pair margins (with or without worst-RX aggregation) shifts errors to other pairs and lowers accuracy.** This is a substantive negative result: the target-free "protect the hard-pair decision margin" hypothesis does not hold in its direct form with rank-16 Shen. Raising the margin requires growing w_t−w_u or pushing prototypes apart, which squeezes the other classes in a low-effective-rank (≈3.5) feature space.
3. The part of drift that works is closer to "dispersion constraint on the decision projection across receivers" (with an asymmetry beyond the variance term) than to margin protection. Relabelling it as protection of harmful drift is not allowed.
4. Pluto: no method fixes it. LS helps Pluto the most (+3.3 pp), and every pair-structured loss is ≤ LS on Pluto.

## 4. Implications for the paper (my judgement)

- **The basis for a method paper claiming "pair-margin robust learning" as its contribution does not hold right now**: A4/A5 fail, and drift's increment over the simplest generic regulariser is near zero on one fold.
- The defensible contributions so far: (i) cross-RX errors concentrate on a few directional pairs that can be predicted from the source domain (TX-permutation test, both folds, WiSig mirror); (ii) **directly raising the margin of predicted-dangerous pairs does not improve cross-RX generalisation — the errors move to other pairs** (the mechanism-level negative result); (iii) once generic-regularisation controls are included, a large share of the RFFI DG gain disappears (most DG papers in the field do not report an LS-level control).
- Before WiSig, the pre-registered constraints say no further Shen training. Suggested next step (needs approval): put CE / LS / drift / drift+LS on WiSig in the same format (source-only λ, 3 seeds, internal RX for development, final 8 RX still sealed). Its purpose is to test whether "the gain is generic regularisation" also holds on WiSig. If drift does not beat LS on WiSig either, the method line STOPs, and the paper is written as diagnosis + negative mechanism + a reporting standard.

## 5. Artifacts

`scripts/receiver_p0/`: p1_losses, test_p1_losses, p1_stats, freeze_sets_p1, train_p1, queue_p1, select_lambda_p1, eval_p1, decide_p1.
`reports/receiver_p1/`: sets_p1.json, risk_fair*, lambda_p1.json, search_table.csv, jobs_formal.txt, eval/{runs, per_rx, per_rx_tx, pairs (all 90 pairs × RX), pairwise_beats, summary_by_fold_arm}.csv, decision.json.
Checkpoints live only on this machine at `outputs/receiver_p1/{search,formal,repro}` (not uploaded). The P0 report and checkpoints are not overwritten.
