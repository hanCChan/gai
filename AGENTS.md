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
- Prior D0/D0.5/B0 and stopped G2/VIRP/EISR/P0/C0 evidence stays immutable.
- Latest explicit user approval authorizes B1 standard CE, RXxTX Group DRO,
  GRL and SupCon implementation and 32 fitting jobs (16 inner searches +16
  formal fits), at most2 reserved V100 GPU-hours, only ten source RX.
- Follow the approved matrix from commit2b89877: Strict CIS, rank16 CNN,
  fixed10 inner/fixed20 formal, two LR candidates per method/fold, paired seeds,
  common group-balanced batch schedule. Never select on outer evaluation RX.
- Exclude OSU, development/locked RX, Day3/4 and raw IQ. No large downloads,
  extra loss-weight searches, fitting retries, seed replacements, ArcFace/B2 or
  novel loss/network/gate algorithm work. Stop after B1 publication/verification.
- If publication fails, keep local artifacts, state what remains unpublished,
  and report the actual error. Local commits alone do not mean GitHub upload.

B1 execution is complete: 32 fitting jobs, no failures/retries, reserved GPU
hours 0.47049565710694374. Independent verification passed. Complete publication
and remote readback, then stop; do not rerun or broaden B1 using remaining budget.
