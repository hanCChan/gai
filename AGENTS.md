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

## Latest user instruction: Receiver-held-out R0–R1 (2026-10-13)

The user supplied the continuous algorithm research plan and instructed understanding, updating the plan and strict execution unless blocking questions arise. This authorizes an independent R0–R1 worktree, closed-set ten-TX unequalized WiSig development caches and the ten Shen source RX Strict CIS caches, up to 6 reserved V100 GPU hours. Preserve all historical stops and reports. Freeze finite source-only hyperparameter selection and three paired seeds. Final WiSig eight RX, last date and equalized data stay sealed. No extra downloads, seed replacement, failed-fit retry, architecture search, R2 training or final test. Publish full code, plan, reports and derived evidence on its own GitHub branch, remotely read back and stop at R1. R2 is design only. The new instruction supersedes old B1/W0 fitting prohibitions only for this experiment.

R0–R1 budget amendment: user explicitly approved 提高到7GPUh，完成原矩阵. Maximum now7 reserved V100 GPUh; keep the original34 searches+51 formal fits, no extra jobs, no R2 or final tests. Original scientific preregistration/seals stay immutable.

R0–R1 completed and stopped:85 fitting jobs (34 source searches+51 formal),0 fitting failures/retries/seed replacements;4.290339055144722 fitting reserved GPUh,4.328058021221111 total conservative reserved GPUh including numerical checks/evaluation/setup allowance. Independent verification passed for112 input assets,1662 receiver/date confusion matrices and515 derived-only ZIP members. Final eight WiSig RX,last date,equalized data remain sealed; R2/final-test jobs0. Publish completed evidence and remotely verify,then stop. Do not use remaining7GPUh approval for extra jobs.

## Latest user instruction: R2-A/B staged revision

The R2 review authorizes an isolated worktree based on84a360b. The subsequent
reply 批准3GPUh和12项矩阵 explicitly grants a NEW3 reserved V100 GPUh cap for
exactly six200-epoch WiSig CE/date-MLDG fits and six preregistered short Adam
diagnostics. Preserve all old stops; do not reuse R0-R1 allocation. Follow
docs/receiver_r2/EXECUTION_PLAN.md and registration.json. R2-B implementation
is conditional on credible measurements; R2-C formal training requires separate
explicit approval. No new architecture, data downloads, class-tail losses,
fit retries, replacement seeds or final tests. Publish each completed stage
and verify full remote bytes/report/decision/SHA.

R2-A/B execution completed and stopped:12 fits (six200-epoch WiSig and six
short source-only diagnostics),0 fitting failures/retries/seed replacements;
1.0346291015355564 total conservative reserved V100 GPUh.
Independent verification passed:80 full Adam snapshots,135680 risk aggregates,
432 confusion matrices,112 input hashes, six exact source-validation/selected
checkpoint replays. CE logging reduction difference<=7.16e-8 is disclosed.
Decision MODIFY for research, GO for minimal-controller engineering only.
No controlled R2-C jobs; formal45-fit proposal is inactive and needs separate
approval. Final8/date3/equalized and developer/locked Shen remain excluded.
Do not use remaining allocation to start more jobs. Publish and remotely verify
this completed stage, then stop.

Future fairness-plan amendment only: matched CE and matched SWAD supplement
the four MLDG controls, yielding an inactive six-arm54-fit R2-C proposal.
This supersedes only the initial45-fit planning count. No controller-rule
change, new training or remaining-allocation reuse is authorized.

## Latest user instruction: R2-C0 cross-RX audit

The new review and 按照方案继续执行 authorize the isolated aaa45ea-based
C0 CPU snapshot transfer audit, unchanged-rule execution optimization and
disposable controlled-step timing. Do not alter R2-A/B source, constants,
reports or decisions. No18/54-fit training or final data is authorized.
Use docs/receiver_c0/EXECUTION_PLAN.md and its frozen registration. Shen fold0
is specified before outcomes for any conditional future18-fit proposal.
Publish this audit with complete remote SHA/report readback, then stop before
any formal fit; a new budget is required for R2-C1.

R2-C0 completed and stopped:80 existing snapshots,0 formal fits or continuous
training trajectories; four-card disposable-step engineering reservation
0.07311376312111154 GPUh (168 reset Adam steps, not fitted models).
Independent verification passed for4800 C1-only decisions,112 input assets,
30 historical evidence files and24 four-card equivalence cases.
Decision MODIFY_CONTINUE_RESEARCH: WiSig frozen crossRX harm detection is
chance-level; Shen has partial source signal, but1/1 multi guard equals mean.
No18/54-fit matrix has been activated, approved or budget-requested.
Conditional18-fit forecast12.2066 GPUh, suggested14 cap remains inactive.
Publish/verify complete C0 audit and protocol, then stop before training.

## Latest user instruction: low-budget receiver calibration v1 (2026-10-24)

New task on branch experiment/rffi-budget-calibration-v1 (base b6ce3c2), defined by
/workspace/research/CLAUDE_RFFI_EXECUTION_20261024.md and the user's 2026-10-24 message.
Cross-domain method transfer (ProtoNet, SimpleShot, MetaQDA, Simple CNAPS, Bayesian shrinkage)
is an accepted form of contribution when attributed and compared with same-task RFFI work.
WiSig, Shen and a third dataset (OSU first; TRANSFER_STRESS_ONLY, see
docs/rffi_calibration/DECISION_LOG.md) are all part of the final validation plan.
This round authorises code, synthetic tests and metadata-only dry runs ONLY: no Stage A real-data
evaluation, no fitting, no sealed data, no GitHub push. The reservation in
configs/rffi_calibration/registration.json stays inactive until the user approves
docs/rffi_calibration/STAGE_A_EXECUTION_PLAN.md. All earlier STOP decisions remain immutable.

## Latest user instruction: execute delegated Stage A (2026-10-24)

The user's "全交给你去运行" authorizes the minimal Stage A: exactly six T0
fits, maximum 1.5 A100 GPU hours including inference, A0 historical eight-class
diagnostics, and source-selection frozen-head evaluation. No optional T1,
Stage B, failed-fit retry, sealed data or push. External candidate metadata,
legal open-access paper retrieval, and a local execution record are allowed;
no paid access or large signal downloads. Preserve all historical evidence.

Stage A completed and stopped on 2026-10-24: six T0 fits, zero failed fits or
fit retries; 12 initial evaluation device failures retained, 12 corrected
A0/A3 evaluations independently verified. Conservative total 0.2539801585
A100 GPUh. Decision STOP_DECOMPOSITION: both WiSig gates failed; H4 below
H2L on all four Shen RX at m=1/5. Shen tuned-family gates unavailable.
Third multi-RX dataset and RF paper full text remain unresolved. Reservation
is closed; no further fitting, numeric evaluation, optional T1, Stage B,
final/dev access, or push from unused allocation. Local record/commit only.

## Latest user instruction: publish completed Stage A (2026-10-24)

The user explicitly authorized GitHub publication of this completed round and
provided a local token file for authentication. Publish code, protocol, reports,
decisions, source checks, local research memory, and derived summary evidence
on experiment/rffi-budget-calibration-v1 in hanCChan/gai. Verify the remote commit
and file tree, then read back the report and decision. No force-push, main merge,
other-branch changes, credentials, signals, numeric features, model weights,
or per-query class scores. This overrides the earlier no-push instruction only;
the closed reservation and STOP_DECOMPOSITION remain in effect.

## Latest user instruction: post-Stage-A C0 (2026-10-24)

Branch experiment/rffi-post-stage-a-diagnostics-v1 from dc805ff, following
/workspace/research/CLAUDE_RFFI_POST_STAGE_A_PLAN_20261024.md section 12. This round authorises
C0 coding, synthetic tests, metadata audit and D0 reading of existing summaries/CSV/curve/status
ONLY. No feature extraction, signal reads, checkpoint deserialisation, training, held-RX
re-evaluation, development/sealed access, large downloads or push. Per-query score replay, D1 and
R0 each need a separate explicit approval (docs/rffi_post_a/STAGE_REQUESTS.md). The publication
authorisation above belonged to the completed Stage A round and does not extend to this branch.
Stage A STOP_DECOMPOSITION and its closed reservation are unchanged; no Stage B/T3.

Publication amendment (2026-10-24): the user then said 先推到github, authorising publication of
this C0/D0 round on experiment/rffi-post-stage-a-diagnostics-v1 in hanCChan/gai (code, configs,
tests, docs, D0 derived summaries). No force-push, main merge or other-branch change; no
credentials, signals, features, weights or per-query scores. This does not approve score replay,
D1 or R0; all reservations stay inactive.

## Latest User Instruction: RV1 Paired-Seed Measurement (2026-10-24)

The user's "继续做执行吧" approves the preceding selection: exactly twelve new
WiSig C_MATCH/P_RX fits (two folds, three paired seeds), unchanged R0 recipe
and fixed evaluations. RV1 has a new independent 0.6 A100 GPUh hard cap, based
on the approximately 0.4 GPUh measured forecast. Follow
docs/rffi_post_a/RV1_EXECUTION_PLAN.md and the write-once variance_v1 snapshots.
Keep R0 NO_GAIN_IN_FIXED_PILOT, its closed reservation and all historical stops
unchanged. No D1, P_MIX, Shen, encoder replacement, retries, seed replacement,
extra fits, sealed access or automatic continuation. Local commits/reports are
allowed; this instruction does not extend C0/D0 publication approval to RV1.

RV1 completed and stopped on 2026-10-24: 12/12 fits, 24/24 fixed evaluations,
zero fit/evaluation failures, retries or replacement seeds; 0.3909960282407701
A100 GPUh under the new 0.6 cap. The RV1 reservation is closed. Independent
verification checked all 53952 accuracy rows, six paired initialization/sample
chains/exposure profiles, eight fixed manifests and 28 input hashes; original
R0 evidence and registration hashes remain unchanged. 109 regression tests
passed before freezing and after execution. Decision NO_STABLE_TWO_ARM_GAIN:
new-seed mean/paired SD w0 +2.753125/4.691437 pp, w1 -3.621875/7.892364 pp.
No new stage, remaining-cap reuse, extra seed, D1, P_MIX, Shen or encoder
replacement is authorized. Code/protocol and derived results are recorded
locally; no RV1 GitHub publication has been performed.

## Latest User Instruction: Publish Completed Post-Stage-A Work (2026-10-24)

The user's "推送到github先" authorizes GitHub publication of this completed
C0/D0, R0 and RV1 branch to hanCChan/gai on
experiment/rffi-post-stage-a-diagnostics-v1. Include code, tests, frozen plans,
closed registrations, reports, decisions and derived summary evidence.
Exclude credentials, signals, numeric features, weights, full manifests,
per-query scores and queue databases. Do not force-push, merge into main or
change another branch. Verify the remote commit/tree and read back the reports
and decisions before claiming publication complete. This is publication only:
all scientific decisions, closed reservations and execution stops remain.

Completed post-Stage-A results were pushed and remotely verified on 2026-10-24
at d1d4e527feccf12f0855280104a9df5f72e873e6. Full remote/local tree equality
passed for 3364 file entries; the 212-path/245-historical-blob publication
audit passed. D0/R0/RV1 reports and decisions read back byte-identically.
Both numerical reservations remain closed; no excluded artifacts were
published. The subsequent documentation-only verification commit must also
be pushed and remotely verified. No new scientific execution is authorized.
