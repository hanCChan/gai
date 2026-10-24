# Post-Stage-A Publication

Date: 2026-10-24.

The user explicitly requested "推送到github先", authorizing publication of the
completed C0/D0, R0 and RV1 work.

Repository: `hanCChan/gai`.
Branch: `experiment/rffi-post-stage-a-diagnostics-v1`.
Observed remote commit before publication:
`7b83a74646da15924f03770d931fd4a7a49cc0e5` (C0/D0).

## Scope

Publish source code, synthetic tests, frozen protocols, source provenance
hashes, closed reservations, execution ledgers, reports, decisions, independent
verification and derived draw-level summary metrics. Retain the existing
commit history; add the previously uncommitted R0 closure and results.

Do not publish credentials, raw IQ, numeric features, model weights, full
support/query manifests, per-query scores or SQLite queue state.
No force-push, main merge or other experiment-branch changes.

Earlier no-publication statements in the frozen execution plans and result
records describe the state before this separate publication instruction.
Those historical files are preserved byte-for-byte. Publication does not
change scientific decisions, reopen reservations or authorize numerical work.

R0 remains `NO_GAIN_IN_FIXED_PILOT` (six fits, twelve fixed evaluation jobs).
RV1 remains `NO_STABLE_TWO_ARM_GAIN` (twelve fits, twenty-four evaluations).
The R0 and RV1 reservations, and the earlier Stage A reservation, remain closed.

Result CSV files use narrowly scoped `-text` Git attributes to preserve their
original CRLF bytes and match the recorded local evidence hashes. Earlier RV1
commits normalized CSV line endings to LF; the publication commit restores
only the original bytes, without changing any numeric values or frozen inputs.

## Verification

Scan all outgoing changed paths and blob contents for excluded artifacts and
credentials before any remote write. Confirm the remote branch is an ancestor
of local HEAD, then push only this branch with a normal fast-forward update.

Verify the remote branch SHA matches the local publication commit. Compare
the full recursive remote Git tree with local paths, modes and blob hashes.
Read back the D0, R0 and RV1 reports and decisions and compare with committed
bytes. Check both published numerical registrations remain inactive.

The verified publication commit, complete-tree comparison, outgoing-file
checks and report/decision SHA256 hashes are recorded subsequently in
`reports/rffi_post_a/publication_verification.json`.
