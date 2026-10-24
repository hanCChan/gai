# RV1 Paired-Seed Stability Measurement

Frozen on 2026-10-24 after user approval "继续做执行吧". This is a new finite
experiment, not an amendment of R0's NO_GAIN_IN_FIXED_PILOT decision.

## Scope

- WiSig w0/w1, C_MATCH and P_RX, seeds 20261025/20261026/20261027.
- Exactly 12 fits, 4800 steps each; identical R0 recipe, including clip=5.
- Exactly 24 fixed evaluations: last.pt primary and native_best.pt secondary.
- Identical Stage A held-RX manifests, 20 draws, budgets 0/1/2/5/10, fixed heads.
- New reservation: forecast approximately 0.4 A100 GPUh; hard cap 0.6 GPUh,
  including all occupied-device wall time during fits and evaluations.
- No retries, replacement seeds, new heads, temperature tuning, horizon changes,
  D1, P_MIX, Shen, OSU, development/final RX, date 3 or equalized data.
- No publication authorization is inferred for RV1. Keep a local code/protocol
  commit and local derived report; never commit scores, features or weights.

## Design And Interpretation

Three newly fixed seeds are the primary replication cohort. The original seed
20261024 is included only in a separately labeled four-seed descriptive cohort.
For each (fold, seed), C_MATCH and P_RX must have identical initial-state hashes
and identical per-step all_ids chains. The same seed is also paired across folds.
Initialization and episode schedule vary together across seeds; this does not
separately identify their individual variance contributions.

Draws are averaged within physical RX and shot budget; 1-shot/5-shot are averaged
within RX; then RX are equally weighted. Report paired method differences, not
differences between unpaired mean accuracies. Seeds, not support draws or repeated
RX appearances, are the replication unit for conditional seed uncertainty.

Report each fold/seed's lowshot, 1-shot, 5-shot, all RX differences, worst-RX drop,
H0 drop and competitiveness against the best fixed C_MATCH secondary head.
Report each arm's conditional seed SD and the paired-difference SD and SE.
A descriptive 95% t interval uses df=2 (t=4.3026527297) for three seeds,
df=3 (t=3.1824463053) for four. These are conditional on the fixed development RX;
they do not establish population-level hardware generalization or a blind test.
The equally weighted two-fold difference is computed per shared seed before
calculating its SD/interval; folds are not treated as independent observations.

## Frozen Measurement Decision

For each new seed separately, retain the original two-arm engineering thresholds:
both folds lowshot gain >=2 pp; 1-shot and 5-shot deltas each >=0; >=6/8 held RX
positive; each fold worst-RX lowshot drop <=3 pp; H0 mean drop <=5 pp; and P_RX/H1e
lowshot at least the best fixed C_MATCH secondary head. Also report these checks
after averaging the three new seeds within each RX.

TWO_ARM_GAIN_STABLE requires every new seed to pass all the above two-arm checks.
Otherwise report NO_STABLE_TWO_ARM_GAIN. An incomplete matrix, lineage mismatch
or score verification failure yields INCOMPLETE_EXECUTION. These are measurement
outcomes, not statistical proof that a method is effective or ineffective.

P_RX minus P_MIX is NOT_MEASURED. Neither outcome is GO_DESIGN_SHEN or a full
replication of the original three-arm rule. No outcome automatically starts
another experiment or adds fits.

## Execution And Verification

Write-once snapshots live in configs/rffi_post_a/variance_v1. Hash the original
R0 protocol/report/decision/verification/ledger, shared source files and RV1 plan
before any new result. Metadata-only preflight verifies exact source/held roles,
coverage and the original manifest hashes before numeric signal access.

The runner only claims the 12 allowlisted jobs. It monitors the cumulative
occupied-device wall time, terminates on cap/interruption, runs fixed evaluation
in bounded subprocesses, and closes the RV1 reservation in a finally block.
Existing queues or fit outputs cannot be overwritten or resumed.

An independent verifier requires every expected CSV row and score key, finite
200-by-10 scores, balanced query labels, matching checkpoint hashes and matching
Stage A manifest hashes. It recomputes accuracy from every saved score array,
then independently recomputes paired aggregates. Source access logs, training
curves, sampling exposure, execution ledger and provenance hashes are indexed.
The original R0 files and reservation are rehashed after execution.
