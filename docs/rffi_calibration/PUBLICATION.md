# Stage A Publication

Date: 2026-10-24.

The user explicitly authorized publication of the completed Stage A round.
Repository: `hanCChan/gai`.
Branch: `experiment/rffi-budget-calibration-v1`.
Research result commit: `a63f74bb9a74e34a0eba2309e5f9088762452a2c`.

## Scope

Publish code, tests, frozen protocol, execution amendments, reports, decisions,
external source checks, local research memory, input provenance hashes,
draw-level summary metrics, failure logs, and independent verification records.
Do not publish credentials, signals, numeric features, model weights,
full support/query manifests, or per-query class scores.

The earlier "no push" statements describe the state at execution closure,
before this separate publication authorization. Scientific results and
execution records remain unchanged. Stage A stays stopped and its numerical
access reservation remains closed. No further training or evaluation.

## Verification

Publication is complete only after the remote branch commit matches the local
commit, all changed file paths and Git blob hashes match the remote tree, and
the remote report and decision are read back and compared with local bytes.
Record the verified publication commit and checks below after they succeed.

## Verified Upload

Published and remotely verified commit:
`c6cd4de2439dfd8b9d9aaaeb7d15ff9576a4efb3`.

- Remote branch SHA equals the local publication commit.
- Complete local/remote tree: **3,154 blobs**, identical paths, modes and Git
  blob hashes. All **134 changed files** were included.
- Credential scan passed. No token, signals, features, weights or per-query
  score arrays were added to the publication.
- Remote `REPORT.md` read back byte-identically: 4,134 bytes, SHA256
  `53ebd448343bba53e62b029fff1ecf6268449b3bc76e0b77b3f8fcc5b1a5d4e3`.
- Remote `decision.json` read back byte-identically: 2,052 bytes, SHA256
  `1a572bd61705151ff824bb0c4c1509a445b7fec74a9911929ec3ac217ac1efe3`.
- Decision remains `STOP_DECOMPOSITION`; numerical reservation remains closed.

This verification record is a subsequent documentation-only commit on the
same branch. No new scientific execution or historical evidence changes.
