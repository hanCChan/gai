# Local Research Memory

Updated 2026-10-24. Branch `experiment/rffi-budget-calibration-v1`, base `b6ce3c2`.

- CPU synthetic dry run and metadata preflight passed. Source integrity:
  68 assets match historical hashes; access audit finds only source RX,
  WiSig dates 0/1. No OSU/dev/final signal access.
- Minimal Stage A: six T0 fits, no failed fits/retries. Both WiSig and both
  Shen-B jobs selected the final step; underconvergence remains possible.
- Twelve GPU evaluations initially failed at H0 CPU/CUDA mismatch before
  rows/metrics. New device-fix attempt: all twelve A0/A3 verified. Failed
  evidence retained; no fit retraining.
- Decision STOP_DECOMPOSITION, current fixed T0 screening only. WiSig H4
  loses strongest simple head by 5.19375/10.925 pp at m=1. Shen H4 loses
  fixed H2L at m=1/5 on all four RX. Simple calibration retained.
- Shen strict H2*/H5c* unavailable (one held RX per encoder); pooling other
  LORO encoder scores would violate target exclusion. Temperature scaling
  not executed; probability metrics are unscaled.
- Total conservative A100 reservation 0.2539801585 GPUh, cap 1.5. Reservation
  closed. Optional T1/Stage B/final tests not authorized by unused budget.
- Third multi-RX data unresolved. WiFi_RFFI DataPort proxy 403. MobRFFI
  source documents 3/4 RX but published S3 bucket NoSuchBucket and license
  unverified. Matching-title IoT-J code readable; missing models/full text.
- Full local results: `reports/rffi_calibration/stage_A/REPORT.md`,
  `decision.json`, `evidence_index.json`, and execution verification.
  Raw scores/manifests/weights remain local and outside Git. No push.

## Publication Update (2026-10-24)

The later explicit user instruction authorized GitHub publication. Completed
code, protocol, reports, decisions, source checks and derived evidence were
pushed to hanCChan/gai on experiment/rffi-budget-calibration-v1. Publication
commit c6cd4de2439dfd8b9d9aaaeb7d15ff9576a4efb3 remotely verified: complete tree
3,154 blobs, all 134 changed paths, report/decision byte-identical readback.
The preceding "No push" line describes execution closure before authorization.
Credentials, signals, features, weights and per-query class scores remain local.
See PUBLICATION.md. Stage A stays stopped; no new training or evaluation.
