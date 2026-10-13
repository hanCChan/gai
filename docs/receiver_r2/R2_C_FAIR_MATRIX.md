# Current fair R2-C proposal — inactive, separate approval required

R2-A found that200-epoch official-CNN CE51.74% exceeds date-MLDG50.78% on
internal RX/new development date. Historical Shen SWAD is also stronger than
MLDG. These are development results, not a target-selected horizon or a blind
test. Formal comparison must include both matched-pool alternatives rather
than relying on their larger-pool historical numbers.

This updates ONLY the future baseline list in the initial R2-B five-arm45-fit
proposal; the sealed controller rule, its constants, source pools and12-fit
R2-A matrix remain unchanged. No C2-informed controller revision is made.

| Arm | Candidate direction/update | Comparison purpose |
|---|---|---|
| CE | Standard CE Adam using A+B fitting pool | Tests the strongest current WiSig alternative under matched information |
| SWAD | CE trajectory with published parameter averaging | Tests the strongest implemented historical Shen alternative |
| Standard MLDG | Full Adam proposal; WiSig complementary RX/date episodes | Matches the controller's base objective |
| Small-LR MLDG | Fixed LR0.00015; no LR search | Tests ordinary effective-step reduction |
| Mean-source backtracking | Fixed scales1/1/2/1/4/0, mean C1 noise guard | Tests ordinary finite source-risk backtracking |
| Multi-RX controller | Same proposal/scales plus sealed multi-RX harm-budget guard | Isolates incremental multi-RX protection |

All arms use the same source fitting/C1/C2 packet pools, fitting-only
normalization, frozen RX A/B/C rotation, paired initialization and fitting
opportunities. WiSig has3 seeds per arm; Shen has2 historical outer folds×3
seeds per arm. Counts:6×3=18 WiSig fits and6×2×3=36 Shen fits; total54.
No extra searches, retries, replacement seeds or new architecture/loss modules.

The four core controls alone yield36 fits; adding matched SWAD gives45;
adding matched CE gives the CURRENT54-fit fair proposal. Every arm and seed
must be retained if this complete matrix is later approved. Historical CE,
SWAD and MLDG scores are contextual because their fitting packet budgets differ.

Source-only model selection is required throughout. The four MLDG arms and CE
must share the same minimum receiver-equal source-validation CE checkpoint
rule. SWAD's published LossValley averaging/selection is part of its algorithm
and must be disclosed explicitly, with the same source validation packets,
validation frequency and fixed fitting horizon; it must not acquire target
selection or extra fitting opportunities. Compare its selection/averaging cost.

Before any formal fit, freeze horizon using source convergence evidence,
initialization/rotation/sampling hashes, all job definitions, numerical stop
rules, measured/provisional method-specific peak memory and timeouts, and a NEW
complete-matrix reservation estimate/cap. The first approved fitting job of
each mechanism is included in the54, never an additional timing pilot. Its
actual cost must be used to update the remaining forecast with a conservative
margin; budget insufficiency requires an amendment rather than seed truncation.

No target result chooses the horizon, risk constants, checkpoint or model.
C2 is read-only development remeasurement; repeated controller changes would
require disclosure and a newly frozen validation protocol. Report every seed's
Accuracy/F1/RX and TX tails, effective update counts, rejection/scale by stage,
accepted-risk remeasurement, elapsed/reserved cost and failures. Synthetic
tests are not evidence of controlled late-stage stability.

This document does not authorize54 fits, first-job timing fits or use of the
unused R2-A allocation. R2-C requires separate explicit user approval of the
complete final matrix and fresh budget. Current formal R2-C jobs:0.
