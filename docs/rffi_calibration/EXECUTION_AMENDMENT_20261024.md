# Stage A Execution Amendment (2026-10-24)

Recorded before the first real-data score or fit.

- User delegation: "全交给你去运行". Execute the minimal six T0 fits under
  1.5 A100 GPUh, A0 historical diagnostics, and A3 frozen heads. Optional T1,
  Stage B, sealed numeric data, and push stay disabled.
- CPU checks: 56 tests passed (14.49 s); synthetic dry run passed (1,284 rows,
  2.4 s); metadata preflight has no problems. Four A100-40GB GPUs were idle.
- Wire the existing SQLite queue into a bounded launcher. Direct real-data
  training requires a running claim. Count process wall time per reserved GPU,
  retain failed jobs, verify source hashes against historical registration,
  and independently verify checkpoint hashes, curves, selection, and finiteness.
- Require BOTH active reservation and feature-extraction permission for numeric
  access. The previous guard checked only the latter.
- Save local class-score arrays and full support/query manifests for independent
  metric verification. Do not publish weights, features, signals, or class scores.
- Shen has only ONE encoder-held target per checkpoint. H2*/H5c* cannot be
  selected by leaving that RX out of a one-RX score table. Pooling scores across
  the other LORO checkpoint is NOT a strict fix: that encoder fitted this target
  RX. Report fixed-head comparisons and explicitly unavailable tuned-family
  gates; no extra fits or outcome-driven protocol replacement.
- Probability metrics remain unscaled (temperature 1); the temperature utility
  exists, but the evaluation CLI does not implement its source-only fitting.
  Report this gap, not a completed calibrated-probability comparison.
- Methods, sample schedule, hyperparameters, fit/eval roles, budgets, and draws
  are unchanged. Cross-fitting on WiSig remains within each four-selection-RX
  fold, not pooled across encoders.

## Evaluation Attempt Amendment

Six fits completed with zero failures/retries (0.0623871209 A100 GPUh).
All twelve initial evaluations failed BEFORE rows/metrics were produced:
GPU feature extraction moved the wrapper to CUDA, and `linear_head()` returned
CUDA parameters to a CPU frozen head. Original logs are retained.
Total conservative reservation after those failures: 0.0813093787 A100 GPUh.
Fix only the exported native-head parameters to detached CPU float64, add a
GPU integration regression, and run a NEW `gpu_head_fix_v2` evaluation attempt
under the same cumulative cap. No fit retry, hyperparameter change, score-based
selection, dataset change, or overwritten failure evidence.

## Closure Errors

After all twelve corrected evaluations exited 0 and passed verification, the
launcher failed while writing the final ledger because a task argument list
shadowed the parsed CLI Namespace. Complete progress already persisted all
12 entries and cumulative resource seconds; a metadata-only recovery command
restored the ledger without rerunning a task. The final report CLI then exposed
a NumPy bool JSON serialization issue; use native scalar types and regenerate
only the derived report. Both errors are recorded in final verification.
Development numeric access now requires its own explicit flag, in addition to
the active-reservation and final-seal checks; no actual development access occurred.
