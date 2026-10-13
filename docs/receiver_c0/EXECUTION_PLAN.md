# R2-C0 execution: cross-RX transfer and controlled-step readiness

Base aaa45ea16f16672a11b695a97607d02b24b238d3; isolated branch
experiment/r2-c0-cross-rx-audit. The user's new review authorizes this audit,
read-only reuse of80 actual Adam snapshots and disposable single-step timing.
No formal fitting,18-fit PoC or54-fit matrix is authorized. Preserve all R2-A/B
code, constants, reports, decisions and previous stops without modification.

Freeze the exact exploratory analysis in registration.json before examining
cross-RX results. Use eval-mode paired packet deltas from the original snapshots.
The original C2 has already been inspected: every new transfer result is
exploratory, not independent confirmation or a blind deployment test.

Three analyses: sameRX/different packet pools; mutually disjoint C1 decision RX
and C2 target RX (all six directed2/2 splits on WiSig, two1/1 splits on Shen);
WiSig date0->1 and1->0 with sameRX and disjointRX variants. C RX were excluded
from the original A/B gradients. Decisions see only original C1 packets on
decision RX/date; original C2 packets on target RX/date remain diagnostic.
No source validation, internal target RX, date2/3 or final8/equalized data enter
these analyses. Constants and finite scales are inherited unchanged.

Compare always-full, fixed-half, mean-risk backtracking and multi-RX guard.
Report each split/snapshot's scales, source/target signs, ranks, false accepted
harm, false rejection of a confidently beneficial full target step, target
mean/worstRX changes and packet/support counts. Do not count reciprocal splits
or80 snapshots as independent experiments. Report raw disagreement/near-zero
cells, RX/date/phase strata, lag1 and first differences with no fitted predictor
or threshold tuning. Target oracle information may label errors, never decide.

Audit observed clipping as a sampled first-batch statistic, rather than all
training steps. Check A/B/C/date/Dropout/Adam semantics and packet disjointness.
Investigate index adjacency and simple cached-packet feature correlations on
source pools only. Original timestamps/session identifiers may be unavailable;
index-order correlation cannot prove temporal/session independence or hardware
causation. Publish summaries and hashes, not packet signals/full features.

Implement a new optimized execution module, leaving the old controller intact.
Cache pre-C1 losses, traverse scales in descending order and stop at the first
accepted candidate; preserve exact original guard arithmetic and optimizer
transaction semantics. C2 is sparse and after decision, evaluating only the
chosen scale against a cached baseline. Test optimized/legacy scale, parameters,
moments and RNG equality, including rejection and next-step behavior. No change
to constants or risk mathematics is allowed.

Time a fixed finite set of reset single steps on four V100s: both datasets,
MLDG snapshots at registered early/middle/final short epochs, standard/legacy
mean/legacy multi/optimized mean/optimized multi. Use original saved gradients
to reproduce exact proposals and separately time full A/B gradient computation;
discard every candidate after measurement. Never concatenate these steps into
an evolving model trajectory. Fresh engineering watchdog360 allocated seconds;
no reuse of the old3-GPUh fitting allocation. Record actual forward calls,
peak allocated/reserved memory, GPU process duration and pessimistic complete
PoC estimates including gradients, validation, sparse C2 and setup/margin.

Only if C0 provides usable cross-RX information propose the18-fit PoC:
three MLDG arms × three paired seeds × WiSig/Shen numeric fold0. The fold0 rule
is fixed before outcomes. Proposed horizons200/20 follow source protocols;
source-only checkpoint selection and accepted opportunities must be aligned.
Label C1's additional identity supervision explicitly; later equal-total-label
CE/MLDG and matched SWAD/CE are required for a publication-level comparison.
No coefficients, fold, horizon or seeds are selected from C2/internal scores.
If evidence is insufficient, deliver an inactive conditional protocol and
MODIFY/STOP on this specific controller, without stopping the research question.

Publish code, this plan, audit/transfer/cost reports, complete derived evidence,
conditional PoC protocol and GO/MODIFY/STOP decision on this branch. Verify all
changed remote bytes/tree/SHA, ZIP members and report/decision readback. Stop
before formal fitting or locked/final evaluation; fresh PoC approval is required.
