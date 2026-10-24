# Stage A: STOP_DECOMPOSITION

Date: 2026-10-24. Fixed T0 encoder-held source-selection screening; not a final-data test.

Six T0 fits completed without failed fits or retries. All twelve corrected A0/A3 evaluations
passed independent score-to-metric and aggregate verification. Twelve initial evaluation
failures (CPU/CUDA native-head mismatch, before rows were written) are retained separately.
Total conservative A100 reservation: **0.253980 GPUh**, below 1.5 GPUh.

## Decision

H4 failed the preregistered +2 pp WiSig 1-shot gate in both folds. On Shen, H4 was below
fixed H2L on all four RX at both 1 and 5 shots. Stop claiming the current decomposition
is necessary; preserve simple calibration and negative evidence. No Stage B/T1/final test.

| WiSig split | H4 1-shot | strongest simple head | simple 1-shot | delta pp | gate |
|---|---:|---|---:|---:|---|
| w0 | 50.638 | H1 | 55.831 | -5.194 | FAIL |
| w1 | 46.138 | H2* | 57.062 | -10.925 | FAIL |

## Wisig

| Head | 0-shot | 1-shot | 2-shot | 5-shot | 10-shot |
|---|---:|---:|---:|---:|---:|
| H0 | 33.69 | 33.69 | 33.69 | 33.69 | 33.69 |
| H0S | 32.38 | 32.38 | 32.38 | 32.38 | 32.38 |
| H1 | N/A | 55.86 | 59.85 | 63.19 | 64.08 |
| H1e | N/A | 55.98 | 59.98 | 62.73 | 63.67 |
| H2fix | N/A | 43.60 | 53.41 | 61.52 | 63.59 |
| H2L | N/A | 49.76 | 58.62 | 62.98 | 64.20 |
| H2* | N/A | 56.17 | 58.97 | 62.40 | 63.28 |
| H3 | N/A | 49.72 | 53.48 | 57.49 | 60.04 |
| H4 | 32.50 | 48.39 | 50.80 | 52.75 | 53.54 |
| H4-AK | 32.50 | 48.20 | 50.69 | 52.69 | 53.52 |
| H4-pooled | 32.75 | 52.85 | 59.13 | 62.42 | 63.79 |
| H5a | N/A | 57.20 | 60.44 | 63.26 | 64.22 |
| H5b | N/A | 56.18 | 61.50 | 64.90 | 66.57 |
| H5c* | 31.87 | 58.16 | 62.03 | 64.94 | 66.66 |

## Shen

| Head | 0-shot | 1-shot | 2-shot | 5-shot | 10-shot |
|---|---:|---:|---:|---:|---:|
| H0 | 55.19 | 55.19 | 55.19 | 55.19 | 55.19 |
| H0S | 60.06 | 60.06 | 60.06 | 60.06 | 60.06 |
| H1 | N/A | 62.24 | 64.95 | 68.74 | 70.51 |
| H1e | N/A | 62.56 | 65.12 | 69.13 | 70.66 |
| H2fix | N/A | 63.22 | 64.24 | 68.10 | 70.08 |
| H2L | N/A | 66.20 | 67.31 | 70.57 | 72.42 |
| H3 | N/A | 59.74 | 62.79 | 66.72 | 68.98 |
| H4 | 58.56 | 60.33 | 61.16 | 63.02 | 64.74 |
| H4-AK | 58.50 | 59.96 | 60.89 | 62.72 | 64.36 |
| H4-pooled | 60.75 | 63.07 | 63.83 | 66.39 | 68.28 |
| H5a | N/A | 62.39 | 64.93 | 69.01 | 70.51 |
| H5b | N/A | 63.38 | 66.02 | 70.19 | 71.59 |

## Interpretation Limits

- WiSig folds have disjoint selection RX but overlapping fitting RX; do not claim independent datasets/seeds.
- Shen cross-fitted H2*/H5c* are unavailable. The other LORO encoder fitted this RX, so pooling cannot fix exclusion.
- Shen fixed-head results average four RX, not support draws as independent samples; raw per-RX rows are retained.
- Both WiSig and both Shen-B jobs selected the final step. New episode-schedule encoders may be underconverged.
- These are new T0 encoders and different query/capture protocols, not reproductions of historical P2E accuracies.
- H4 underperformance does not prove all Bayesian heads or all learned representations fail.
- H4-pooled beats H4 in the aggregate; finite-center U brings only small aggregate changes. No mechanism is inferred.
- Probability metrics use temperature 1; source-only temperature fitting was not executed.
- Shen row order is not verified temporal separation. OSU numeric evaluation was not run.
- Third independent multi-RX validation remains unresolved; the program is not a completed three-dataset study.

## Evidence

- `decision.json`, `paired.json`, `wisig_aggregate.csv`, `shen_aggregate.csv`, and `evidence_index.json`.
- Full local prediction arrays, manifests, confusion counts and per-task verification are in the indexed directories.
- Initial failure logs remain in each task parent; successful results are only under `gpu_head_fix_v2`.
- Source integrity: 68 source assets matched historical SHA256 registration; no sealed/development numeric access.
- Test counts and command timings are recorded in the final execution verification record.
- Weights, signal arrays, feature vectors and class-score arrays are local only; no push performed.
