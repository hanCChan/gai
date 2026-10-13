# Research execution and publication

The user's 2026-10-13 instructions apply to future work in this repository.

- Every completed research execution must publish its code, plan, results,
  decision, and evidence index to `hanCChan/gai` on the appropriate experiment
  branch. Verify the remote file tree and read back the report/decision before
  calling an upload complete. Give exact GitHub links in the final response.
- Upload is already authorized. Do not repeatedly request permission to upload.
  Do not force-push, merge into main, or change another experiment branch.
  GitHub connector account `hanCChan` is the intended account for this repository.
- Preserve existing P0/G1/G2-0/VIRP/EISR code and evidence. Use isolated worktrees
  and source input paths; do not copy large signal datasets into Git.
- Keep unpublished manuscripts, rejection letters, raw IQ, model weights,
  credentials, and personally identifying correspondence out of publication.
  Publish provenance hashes and factual summaries instead.
- P0/C0/G2/VIRP/EISR remain stopped. The latest explicit user plan approves
  D0.5 archived signed-damage analysis and B0 exactly eight new CE runs with
  seeds 20261013/14, Full/Strict CIS, two source-only folds, fixed20 epochs.
  This later authorization overrides the earlier D0 no-training restriction.
- Use only ten source RX cached CIS/labels/rows/manifests and archived scores.
  Exclude OSU, four development RX, six locked RX, Day3/4 and raw IQ.
- Cap B0 at eight actual fitting jobs and one reserved V100 GPU-hour; preserve
  the eight zero-step provenance-preflight failures and all fitted-run failures,
  no seed replacement or validation-selected checkpoint/LR/augmentation.
- Prepare a concrete B1 implementation/budget plan, but do not train B1/B2 or
  develop new loss/network/gate/pairwise algorithm. Stop after authorized work.
- If publication fails, keep local artifacts, state what remains unpublished,
  and report the actual error. Local commits alone do not mean GitHub upload.
