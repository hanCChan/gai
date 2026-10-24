# RV1 Outcome Interpretation

Date: 2026-10-24. This note interprets the already frozen measurement; it does
not change the protocol, decision, training recipe, source code or matrix.

## What The Measurement Answers

The three new paired seeds give lowshot method differences:

- w0: -2.47, +6.61, +4.12 pp; mean +2.75 pp; paired SD 4.69 pp.
- w1: -12.37, +2.96, -1.45 pp; mean -3.62 pp; paired SD 7.89 pp.
- Equal-fold mean within each seed: overall mean -0.43 pp; paired SD 6.29 pp.

These are measured conditional seed differences, unlike R0's single-seed
cross-method differences. An apparent approximately four-pp difference is not
well resolved by this finite sample: both fold-level descriptive t intervals
include zero and substantial differences of either sign.

This does not prove equal methods or universal algorithm ineffectiveness.
It does support stopping the claim that this fixed P_RX recipe has demonstrated
stable gains. Four seeds including R0 still give opposing fold means (+3.08 and
-3.70 pp). No extra seed is activated to pursue a favorable mean.

## Risks Are Not Explained Away By Variability

For the new three-seed averages, H0 native zero-shot accuracy drops by 21.96 pp
in w0 and 23.96 pp in w1. In seed 20261025, H0 drops 46.50 and 39.88 pp.
These are exact saved-score observations under the registered H0 check; they
are not evidence of a specific failure mechanism.

Seed 20261026 improves lowshot in both folds (+6.61/+2.96 pp), has seven of
eight positive held RX, and passes the competitiveness checks. It nevertheless
fails the registered risk requirement: w1 H0 drops 7.00 pp, above the 5-pp limit.
Thus none of the three new seeds passes the full two-arm engineering checks.

On the seed averages, w1 worst-RX lowshot drops 9.58 pp. P_RX/H1e is also below
the best fixed C_MATCH secondary-head mean in both folds:
67.19 versus 68.37 percent in w0, 60.04 versus 71.21 percent in w1.
These heads are descriptive fixed controls, not a newly selected deployment head.

## Secondary Checkpoint

The independently verified native_best.pt evaluations are secondary only.
Their new-seed lowshot mean differences are +3.97 pp in w0 and -2.79 pp in w1;
paired SDs are 4.68 and 7.39 pp. Seed-mean gain, per-budget, risk and
competitiveness checks still fail jointly. No checkpoint selection is made
from these target results.

## Verification And Stop

The 109 synthetic/shared regression tests passed before freezing and again
after execution (27.79 and 28.61 seconds respectively). An additional CSV-only
aggregation using the Python statistics module independently reproduced all
six primary paired fold/seed differences, H0 differences and both paired SDs.

The frozen score verifier checked 53,952 rows in 24 directories, with zero
accuracy discrepancy. The complete input, pairing, exposure, manifest,
query-label, access-scope, checkpoint and unchanged-R0 checks passed.
All four A100 devices returned to zero memory usage and zero utilization.

RV1 decision: NO_STABLE_TWO_ARM_GAIN. R0 remains NO_GAIN_IN_FIXED_PILOT.
This is not a full three-arm validation; P_MIX was not repeated.
No D1, new encoder, Shen extension, additional seed or publication was started.
The fixed recipe stops here; any next experiment requires a separately scoped
instruction and protocol.
