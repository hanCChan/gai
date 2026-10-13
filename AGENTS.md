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

## Latest user instruction: M0–M3 signal mechanism screening (2026-10-13)

The new user-supplied plan and reply 不设预算 authorize source-only raw IQ
mechanism screening, traditional frozen-model input controls and conditional
minimal PoC. This overrides the old B1-only raw-IQ/download prohibition for
this new experiment. B1 itself remains completed and stopped. Only ten source
train RX are allowed; no development/locked RX, Shen Day3/4 or OSU.
Official WiSig ManyRx download is authorized now; keep signals sealed until
Shen method freeze. M3 has no resource cap, but all mechanism/identity/novelty
gates and one finite preregistered matrix still apply. Do not mistake budget
approval for scientific evidence. Follow docs/m0_signal/EXECUTION_PLAN.md.
Preserve historical evidence and publish completed execution with remote readback.

## Latest user instruction: receiver transformation sharedness and conditional PoC

On 2026-10-13 the user authorized WiSig acquisition using the proposed alternate
routes and a new receiver-transformation PoC only if sharedness passes. They
explicitly required held-out TX to be excluded from BOTH encoder optimization
and mapping estimation. This new experiment supersedes the prior M3 correction
hypothesis; it does not reopen B1 or any stopped experiment.

The preregistered 20 strict eight-TX/five-RX IQ encoders completed 24000 optimizer
steps. Fixed rank2 mappings failed sharedness in all four fold/seed strata.
Conditional classification PoC: zero jobs. Stop this concrete mechanism;
no rank/coefficient search or failure-driven algorithm replacement is authorized.
Eight zero-step launch aborts are preserved and disclosed. Total reserved GPUh
including them: 0.11499796053388309. See reports/receiver_transform/2026-10-13.

WiSig acquisition is a separate data task. Keep the signal values sealed;
metadata-only inert inspection may verify structure and payload lengths.
Kaggle is a third-party source with Unknown listed license; record original
CC BY-NC-SA 4.0 requirements and do not claim byte-equivalence without an
official checksum. Never publish the dataset, numeric features or weights.

WiSig ManyRx acquisition completed and sealed on 2026-10-13 via Kaggle v1:
archive1310655283bytes, SHA256
b0eca705816b03a47290286fd23ca1a10e02c6f9bd29b5461cfd92c13a86fbdd.
ZIP CRC and inert metadata inspection passed:10TX,32RX,4dates,256 samples,
equalized[0,1],2560 structures. Non-equalized249666 and equalized247684 packets;
never assume packet-level pairing. Official byte equivalence remains unproven.
See reports/wisig_acquisition/2026-10-13; no additional research jobs authorized
by successful acquisition after the rank2 sharedness stop.

## Latest user instruction: WiSig W0/W1 structure window (2026-10-13)

The user supplied the next-stage review and said 理解方案，然后执行.
This authorizes audited numerical reading of unequ­alized development WiSig,
metadata/coverage-based split freezing, at most two official-architecture pilots,
and CPU W1 analysis using the frozen encoders. Prior strict encoder+mapping TX
exclusion is preserved. Shen rank2 and all historical stops remain immutable.

Metadata-only initial final8 RX and last date stay reserved. The preliminary
16-source split failed coverage before any model jobs; among the24 development
RX the14 with complete>=160-cell coverage become source, the10 others internal.
No classification outcome was used to change this split. The original failed
coverage registration/audit is retained. Final8 RX and date3 opaque buffers may
be parsed as file structure but must not become numeric arrays or feature data.

W0: two fixed seeds, official CNN port, max100 epochs/source-only patience5,
no LR search, retry, replacement seed or extra fitting jobs. W1 does not revive
rank2 and does not infer causal receiver hardware. Finish W0/W1, publish remote
verified report/decision and an inactive W2 plan. Do NOT execute W2/W3 or broaden
the matrix without a later user instruction. No extra datasets/downloads.

W0/W1 finished:2 pilots,0 fitting failures/retries,0.0331146915541659 reserved
GPUh. Both official-CNN source encoders pass the frozen quality gate.
Observed TXxRX interactions repeat across date, but the fixed conditional
transfer diagnostic fails all4 strict TX/date conditions and harms known-TX
identity classification. Decision STOP_TRANSFORMATION_GENERATION_W1_UNSUPPORTED;
W2/W3 jobs0. Do not treat repeatability as cross-identity predictability or revive
rank2. Final8 RX,last date,equalized numerical values remain reserved. Publish
and remotely verify this completed window,then stop. Later instructions needed
for any additional fitting or new mechanism. See reports/wisig_w01/2026-10-13.
