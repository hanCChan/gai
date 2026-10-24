# RV1: Paired-Seed Measurement - NO_STABLE_TWO_ARM_GAIN

Date: 2026-10-24. WiSig development data; not a blind test.

## Result

Exactly 12/12 fits completed, zero failed fits/retries/replacement seeds; 24/24 fixed evaluations.
Occupied-device wall time: **0.390996 A100 GPUh**, hard cap 0.6.
Reservation is closed. R0 remains **NO_GAIN_IN_FIXED_PILOT**; no D1/P_MIX/Shen/new encoder was run.

| Fold | Paired gains by new seed (pp) | Mean (pp) | Paired SD (pp) | Conditional 95% t interval (pp) |
|---|---|---:|---:|---|
| w0 | 20261025: -2.47, 20261026: +6.61, 20261027: +4.12 | +2.75 | 4.69 | [-8.90, +14.41] |
| w1 | 20261025: -12.37, 20261026: +2.96, 20261027: -1.45 | -3.62 | 7.89 | [-23.23, +15.98] |

Combined folds are averaged within each shared seed: mean -0.43 pp, SD 6.29 pp, interval [-16.06, +15.20] pp.

## Risk And Thresholds

| Fold / seed | 1-shot (pp) | 5-shot (pp) | Positive RX | Worst-RX drop (pp) | H0 drop (pp) |
|---|---:|---:|---:|---:|---:|
| w0 / 20261025 | -2.19 | -2.74 | 1/4 | 6.26 | +46.50 |
| w1 / 20261025 | -9.61 | -15.14 | 0/4 | 20.25 | +39.88 |
| w0 / 20261026 | +8.29 | +4.93 | 4/4 | 0.00 | +2.38 |
| w1 / 20261026 | +2.79 | +3.13 | 3/4 | 1.94 | +7.00 |
| w0 / 20261027 | +4.99 | +3.24 | 2/4 | 5.59 | +17.00 |
| w1 / 20261027 | +0.50 | -3.41 | 2/4 | 10.29 | +25.00 |

The two-arm engineering rule passes per new seed: 20261025: False, 20261026: False, 20261027: False.
Seed-mean two-arm checks: `{"competitive": false, "gain": false, "per_budget": false, "risk": false, "rx_consistency": false}`.
TWO_ARM_GAIN_STABLE requires all three new seeds to pass; it is not a significance test.
P_RX minus P_MIX is **NOT_MEASURED**. There is no full three-arm GO decision.

## Conditional Seed Variability

| Fold | C_MATCH lowshot mean / SD (%) | P_RX lowshot mean / SD (%) |
|---|---|---|
| w0 | 64.44 / 5.98 | 67.19 / 2.15 |
| w1 | 63.66 / 7.92 | 60.04 / 1.83 |

The measured variability jointly changes initialization and the episode schedule.
It does not isolate these components, prove a mechanism, or define the noise level of every training method.
Intervals are descriptive and conditional on the same eight development RX.
Support draws and overlapping fit RX are not treated as independent replication units.

## Four-Seed Supplement

The original R0 seed is included only as a separately labeled descriptive supplement.

| Fold | Four-seed mean gain (pp) | SD (pp) | Conditional 95% t interval (pp) |
|---|---:|---:|---|
| w0 | +3.08 | 3.89 | [-3.10, +9.27] |
| w1 | -3.70 | 6.45 | [-13.96, +6.56] |

## Verification

Independent score replay checked **53952 rows** in 24 directories; maximum accuracy difference zero.
All six pairings have matching initialization, sample-chain and exposure hashes.
Eight manifest hashes and query-label arrays match across every evaluation.
28 input assets were rehashed; frozen code/config and R0 evidence hashes remain unchanged.
All accesses are WiSig source RX, dates 0/1; no final/development/equalized access.

## Artifacts

`decision.json`, `summary.json`, `paired_fold_seed.csv`, `paired_rx_seed.csv`,
`verification.json`, `execution_ledger.json`, `preflight.json`, `evidence_index.json`.
Scores and weights remain local and excluded from Git.

No next stage or additional fitting is activated.
