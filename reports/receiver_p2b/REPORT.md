# P2B results: function-level objective + class-conditional alignment control — 2026-10-21

Protocol `docs/receiver_p2/PROTOCOL_P2B.md` (857d861) committed before any P2B fit; losses unit-tested
17/17 before any fit (`test_p2b_losses.py`); P2 tests still 23/23. 12 search + 12 formal fits ≈ 1.3 job-h.
ce / ls / rxnull / cmean checkpoints reused. Decision by `decide_p2b.py`.

## 0. Decision: rule **R1** fires literally → rxlogit is the designated main objective, but the ranking is not resolved

Pre-registered R1 = "rxlogit beats ce and ls, and rxnull does not beat rxlogit". Formal status:
rxlogit beats ce (5/6, +1.71/+3.76 pp) and ls (5/6, +1.43/+2.01 pp); rxnull does not beat rxlogit under
the rule (4/6, +0.82/+0.25). **This is a knife-edge**: rxnull's mean is still higher and its P-set
confusion is lower (see table), so the honest reading is "rxlogit and rxnull are tied within noise at this
n, and both beat ce and ls". R1's literal criterion is met; R1's intent ("function-level objective
suffices") is supported, not proven.

| arm (λ) | f0 acc | f1 acc | worst RX f0 | Pluto f0 | P_conf f0/f1 | nonP f0/f1 |
|---|---:|---:|---:|---:|---:|---:|
| ce | 82.67 | 85.45 | 74.8 | 77.0 | .152/.144 | .0045/.0020 |
| ls | 82.95 | 87.20 | 76.7 | 80.3 | .148/.119 | .0046/.0025 |
| cmean | 84.20 | 87.23 | 74.6 | 74.6 | .138/.126 | .0042/.0018 |
| **ccalign (3/1)** | **84.91** | **89.00** | 75.6 | 76.2 | .125/.108 | .0044/.0015 |
| **rxlogit (3/1)** | 84.39 | 89.21 | 75.3 | 74.4 | **.118/.106** | .0041/.0016 |
| **rxnull (0.3)** | **85.21** | **89.46** | 76.4 | 77.4 | .129/.096 | .0039/.0024 |
| rxnull_rand | 83.03 | 87.63 | 72.0 | 72.1 | .141/.122 | .0053/.0018 |

Pairwise (pre-registered rule): rxlogit−ccalign −0.52/+0.21 (3/6, not a win); rxnull−ccalign +0.30/+0.46
(3/6, not a win). **Neither classifier-geometry arm beats plain normalized class-conditional alignment
by the rule.** rxlogit−rxnull −0.82/−0.25 (2/6); rxnull−rxlogit +0.82/+0.25 (4/6). λ fold 0 at the grid
boundary 3.0 for both new arms (boundary-limited, reported).

## 1. Re-audit (audit_p2.py over all 117 checkpoints; `reports/receiver_p2b/audit/`)

| f0/f1 | ce | ls | rxnull | rxlogit | ccalign | cmean |
|---|---|---|---|---|---|---|
| κ (function level) | .0088/.0111 | .0110/.0120 | .0038/.0040 | **.0018/.0040** | .0024/.0036 | .0020/.0011 |
| n_null | .0059/.0081 | .0179/.0204 | **.0609/.2051** | .0026/.0044 | .0017/.0031 | .0041/.0041 |
| E_D | .310/.381 | .172/.203 | **.822/2.884** | .127/.241 | .106/.179 | .075/.046 |
| κ_out (outer, realized) | .047/.024 | .086/.046 | .033/.015 | .020/.014 | .024/.013 | .035/.010 |
| eff. rank | 7.49/7.38 | 8.38/8.28 | 7.00/6.80 | 6.06/6.49 | 6.82/6.88 | 7.60/7.50 |
| outer acc | 82.67/85.45 | 82.95/87.20 | 85.21/89.46 | 84.39/89.21 | 84.91/89.00 | 84.20/87.23 |

rxlogit does exactly what it was designed to do: κ is the lowest of the classifier arms with **no
null inflation** (n_null at the CE level) and the lowest realized outer displacement; ccalign reaches a
similar κ without any head, confirming that plain displacement alignment already achieves the
function-level effect. P2's MODIFY_OBJECTIVE is resolved: the objective is now function-level.

## 2. What this means

1. **The P2 gain is real and survives the audit**: every arm that lowers κ more than ce improves outer
   accuracy, over 117 checkpoints and both folds. The mechanism-level story is intact.
2. **The distinguishing claim is not established on Shen**: ccalign (no classifier, plain
   class-conditional first-moment alignment) matches rxlogit and rxnull. Inside a single linear head
   trained jointly, alignment of the means and alignment of their logit images are close to the same
   constraint (the head is free to put the aligned means on decision directions), so this is expected,
   and it means the paper cannot claim "classifier geometry beats class-conditional alignment" from
   Shen alone.
3. **What remains defensible**: the normalized ratio (rxnull beats ccalign by +0.30/+0.46, not by the
   rule, but never loses) and the function-level formulation (κ) are the honest novelty elements; the
   literature check (below) shows CIP-mean / ISR-Multiclass / MASF already contain "make environment
   class-mean shifts invisible to the classifier", so novelty must be argued at the level of the
   normalized, end-to-end, receiver-conditioned formulation and the RFFI-specific evidence — not as a
   new principle.
4. **Pluto**: still no classifier arm beats ls (80.3) except rxnull's 76.4 → ls 80.3. Unchanged.

## 3. Literature check (background agent, primary sources; details in reports/receiver_p2b/NOVELTY.md)

Nearest prior art: **CIP-mean** (Chen & Bühlmann 2010.15764) — for a linear model its constraint
E_m[Xᵀβ|Y] = E_1[Xᵀβ|Y] is rxnull's numerator set to zero; **ISR-Multiclass** (2311.00966) — estimate
span{D_{t,e}} from per-class environment means, fit a linear head in its nullspace (hard, post-hoc,
frozen features); **MASF** (NeurIPS 2019) — class-conditional alignment of mean logits across domains
inside a meta-loop. Distinctions that survive: normalized ratio that cannot be satisfied by shrinking
displacement; end-to-end through encoder and head; softmax shift handled by row-centring; RFFI setting.
INLP/LEACE/JSE/orthogonal-classifier are the reverse direction (erase the nuisance); RIEI / IB-RFF /
DRIFT erase or compress without classifier geometry.

## 4. Consequence for the next step

WiSig protocol is written with the five-arm set **ce / ls / ccalign / rxlogit / rxnull** plus the
random-subspace control, and the Shen record above fixes the claims in advance: WiSig is the test of
whether *any* of the exposure-reducing arms beats ls and ccalign on a different dataset. No Shen
re-tuning. See `docs/receiver_p2/WISIG_TRANSFER_PROTOCOL.md`; fits await approval.
