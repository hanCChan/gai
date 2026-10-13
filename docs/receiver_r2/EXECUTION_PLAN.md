# R2 staged execution plan — approved R2-A allocation

Base: `84a360bbae2787bb78c00a3590cdb9be2151434f` of `hanCChan/gai`.
Worktree: `/home/keyan/code-y-r2a`; branch: `experiment/r2-a-adam-risk`.
This implements the user's R2 review. Historical experiments and stops remain
unchanged. User reply 批准3GPUh和12项矩阵 grants a fresh3 GPUh allocation. No historical budget is reusable.

## R2-A: six fixed-horizon fits and proposed six short diagnostics

The authorized convergence scope is exactly six WiSig official-CNN fits:
CE and MLDG-RX+Date, seeds 20261013/14/15, 200 epochs, historical source-selected
LR 0.0003, historical fitting packets, balanced sampler, initialization,
second-order MLDG objective, Adam, clipping and source-validation selection.
Only the horizon changes. Preserve both 100- and 200-epoch results. Verify the
first100-epoch schedule and trajectory against the historical paired runs.
No source or target performance may select a winning horizon. No new LR search.

The proposed short diagnostic matrix is six additional fits at seed20261013:
WiSig CE and MLDG-RX+Date, 20 epochs each; Shen CE and MLDG-RX in each of the
two historical folds, 10 epochs each. This is a measurement study of early
training, not a three-seed generalization comparison. Count and horizon are explicitly approved by that reply.

These short fits have separate packet pools. Within each source RX×TX×date,
WiSig positions0:80 fit,80:100 C1,100:120 C2; the historical40 validation packets
stay120:160. Shen class ordinals0:120 fit,120:140 C1,140:160 C2 and160:200 validate.
For Shen compute the scalar normalization only from the120 fitting packets.
The convergence fits retain their original full120-packet WiSig fitting pool;
they must not be silently converted into this diagnostic split.

Freeze stepwise RX permutations ahead of execution. WiSig uses A/B/C=5/5/4;
Shen uses2/1/2. CE uses only A+B fitting packets; standard MLDG uses A and B
(two complementary date episodes on WiSig). C never generates the candidate
gradient. Each control will inherit exactly this same fitting pool, exclusion
schedule, initialization, step count and source-only checkpoint rule.

On the first batch of every short-fit epoch, save local pre/post parameter and
complete Adam states, actual clipped gradient and actual parameter delta. Read
all20 C1 and20 C2 packets per RX×TX×date on C. Record scale1,1/2,1/4,0 finite
loss differences. Scale1 is the actually accepted Adam step; other scales are
read-only counterfactual measurements. Record actual accepted losses separately
and verify equality to the measured scale1 proposal. Do not control training
using these diagnostics in R2-A.

Primary risks use eval mode. Also measure train mode with three paired Dropout
random masks, using identical random state before/after each candidate and
isolating measurement RNG from training. Publish mode disagreement separately.
Record gradient norms before/after clipping, Adam step/moment norms, parameter
delta norms and phase summaries. C1/C2 signs, rankings, paired-delta standard
errors, near-zero and unsupported cells remain visible; do not reduce these
to an unqualified harmful-step percentage. C2 never chooses updates, thresholds,
checkpoints, LR or duration. Local states/weights are private; publish hashes
and derived numeric evidence only.

## Cost and execution gate

The user approved a fresh cap of3 reserved V100 GPUh for these12 fits, numerical checks
and frozen evaluation. The machine has four V100-SXM2-32GB GPUs. Independent
jobs may run one per card; reserved GPUh is the sum of allocated card durations,
not wall-clock duration. The historical method-specific estimate, margins and
unmeasured diagnostic multiplier are in registration.draft.json.

Measure each dataset/method's first registered job; it counts within the12,
never as an extra pilot. Recompute a conservative complete-matrix forecast from
those measurements. Freeze measured peak memory and per-type timeouts before
remaining jobs. Keep a global cap watchdog. If the complete matrix exceeds
the approved cap, report and request a concrete amendment before continuing;
do not select favorable seeds or replace failed jobs.

Numerical errors, leakage and budget overruns halt execution and preserve logs.
A performance decrease on one seed does not justify truncating an approved
matrix. Existing final8 WiSig RX, date3, equalized data, Shen developer/locked
RX and OSU remain excluded. No large downloads, new feature pipeline or backbone.

## R2-B: conditional implementation after credible R2-A measurement

Specify the actual Adam proposal and finite scales {1,1/2,1/4,0}. Any tolerance
must be frozen from source packet estimation error before controlled training;
do not default to a hard all-RX no-increase rule or tune against internal RX.
Rejecting restores parameters and the complete optimizer state. A nonzero
scaled proposal keeps the candidate moments and scales only the parameter
delta; document this policy and verify rollback/reproducibility deterministically.
Report acceptance, complete rejection, mean scale, effective cumulative delta,
source loss, elapsed time and C1/C2 reliability by stage. Freeze the mathematical
definition only after measurement credibility is assessed.

## R2-C: separate approval required

Prepare a complete paired matrix for standard MLDG, fixed-small-LR MLDG,
ordinary mean-source-risk backtracking and the multi-RX controller. Both datasets,
three seeds and the original two Shen folds; identical fitting/risk packet
pools and source-only selection. The minimum four-arm matrix would be36 fits.
Audit comparison with strongest existing methods (WiSig date-MLDG, Shen SWAD);
state whether fair-pool SWAD needs adding before seeking approval.

Only after technical checks and a separate explicit user approval of the final
model count, fresh resource cap, measured peak memory and timeouts may R2-C
train. No R2-C fitting is authorized by this draft or by R2-A approval.
Compare novelty against MLDG, Fishr, PCGrad, CAGrad, SWAD, Group DRO and standard
line-search/trust-region methods using primary sources. Do not claim optimizer
composition as a new mechanism, or risk correlation as causation. Class-tail
protection remains conditional on repeatable mean gains with residual class harm.

## Publication and decisions

Each completed stage publishes code, plan, full results, GO/MODIFY/STOP decision
and derived evidence index to its experiment branch, with complete remote file
tree/SHA verification and report/decision readback. Preserve failures and near-zero
cells; do not publish raw IQ, complete features, weights, states or credentials.
The preparation status is APPROVED_R2_A_PREEXECUTION; this is not research success.
