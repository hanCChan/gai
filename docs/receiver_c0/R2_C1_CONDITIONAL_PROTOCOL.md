# R2-C1 complete conditional protocol — inactive

The current C0 decision is MODIFY_CONTINUE_RESEARCH. Partial source ranking
information exists, but the frozen WiSig harm decision is chance-level and
incremental multi-RX choices are sparse. This protocol is frozen for review;
it is NOT activated, scientifically cleared or a request to spend14 GPUh now.
Any activation requires explicit scientific reassessment and fresh budget.

## Finite matrix

Exactly18 formal fits:3 methods×3 paired seeds×2 dataset settings.
Methods: standard MLDG, MLDG+mean-source backtracking, MLDG+multi-RX guard.
WiSig official CNN,200 fixed epochs; Shen rank16 author CNN,20 fixed epochs,
numeric historical fold0. Fold0 was specified in the C0 registration before
outcomes and is not selected for favorable classification results. Seeds are
20261013/14/15; LR0.0003, innerα0.01,β1, clip5, original Adam epsilon1e-7/1e-8.
No search, fitting retry, replacement seed, new architecture, feature generator
or loss component. Complete all approved seeds even if accuracy decreases.

WiSig uses the original14 source RX/date0/1,80 fitting packets perRX×TX×date,
20 C1,20 C2, original40 source-validation packets. Shen fold0 uses its original
five fitting RX,120/20/20/40 perRX×TX, fitting-only scalar normalization.
Validation and any target RX never enter proposals, thresholds or selection.

Per-step A/B/C is5/5/4 (WiSig),2/1/2 (Shen), from a seed+997 permutation frozen
before execution. All three arms use identical gradient fitting packets and
exclusion schedules. WiSig MLDG averages the two complementary date episodes
with its seed+1997 date permutation. Balanced scheduling samples two packets
perRX×TX×date cell; every fitting-pool packet occurs once per epoch in the
opportunity schedule, but C's packets do not produce gradients in that step.
WiSig40 opportunities/epoch ->8000; Shen60/epoch ->1200 perfit.

The formal original controller uses C1 on ALL current C RX, with the original
mean-plus-multi-RX harm-budget formula and2×paired-SE margin. C0's1/1 and2/2
disjoint-RX analyses are prediction audits, not an unannounced replacement of
the formal controller membership. Sparse sameRX C2 diagnostics in a controlled
trajectory must not be presented as independent crossRX confirmation. If a
later proposal instead requires RX-disjoint C1/C2 control groups, register that
different protocol explicitly; Shen1/1 multi guard would degenerate to mean.

Standard MLDG always accepts scale1. Backtracking uses only the fixed mean guard;
multi guard adds exactly the original second condition, with scales1/.5/.25/0.
Before proposal, preserve parameters and entire Adam state. Nonzero scale keeps
candidate moments; rejection restores parameters/steps/moments/parameter groups.
Do not rewind fitting RNG. Optimized per-step execution caches only current
pre-C1 loss, exits at first accepted scale and retains the accepted loss vector.
C2 is read only on each epoch's first opportunity, after choice, at chosen scale
against pre-update parameters; it never tunes the rule or chooses a checkpoint.

## Selection, verification and metrics

All methods select minimum source-only RX-equal validation CE, with first/lower
epoch for ties; fit full horizon without target-dependent early stopping.
Freeze initialization, source pool, schedule and rotation hashes before launch.
Save actual parameters/optimizer/RNG before and after sparse audit steps locally;
publicly report state/delta hashes, scalar norms and losses only. Record every
opportunity's objective, clipping, chosen scale, accepted update norm and effective
update/packet counts. Check exact legacy/optimized semantics, rejection and next
Adam update before the first formal fit; replay recorded candidates independently.

Only after all18 checkpoint selections freeze may development classification
evaluation occur. WiSig uses the historical source seen/new-date and internal
RX seen/new-date development roles, never final8/date3/equalized values. Shen
evaluates fold0's five other historical source RX, never four developer/six
locked RX. Report perseed Accuracy, supported-class Macro-F1, RX q10/worstTX
tails, accepted updates/scales by phase, C1/C2 errors and all costs. No target
score adjusts LR, horizon, coefficient, fold, checkpoint or a replacement seed.

C1 identity feedback is additional supervised information. This18-fit mechanism
PoC cannot establish equal-total-label-budget benefit. If useful, next approval
must cover Shen fold1, CE/SWAD, fixed-small-LR and fit+C1-label ordinary baselines
with disclosed computation; no automatic expansion to54 or paper novelty claim.

## Cost, engineering stops and approval

Observed reset-step forecast with50% margin plus explicit source-validation,
logging/checkpoint, setup/cold-kernel and sparse-C2 allowances: about12.207
reserved V100 GPUh for18 fits. Conditional proposed cap14 GPUh. These estimates
come from three existing short states per dataset on all four cards; they are
not a deterministic full-training bound. Observed allocator-reserved envelope
up to2214MiB percard; propose8GiB allocated-memory engineering ceiling to leave
room for actual full-step/state logging. Four jobs can run one percard, not DDP.

The first already-approved formal job of each mechanism must count in the18,
record real per-opportunity/epoch cost and peak allocated/reserved memory, and
reforecast the COMPLETE remainder before dispatch. Provisional timeout perfit
is twice its conservative one-fit forecast plus120 seconds; both timeout and
reservation cap must be explicitly approved. Do not replace incomplete seeds.
Budget insufficiency requires a concrete amendment rather than truncation.

NaNs/nonfinite gradients, data boundary violation, nondeterministic transaction
failure,500 consecutive fully rejected opportunities, memory/timeout breach or
budget exhaustion halt and preserve all logs as engineering failures. Low
classification accuracy or a single negative seed is not this stop criterion.

Current authorized formal fits:0. No first-job timing may start an evolving
trajectory before the new matrix/resource approval. Historical unused allocation
is unavailable. Publish C0 and read back remote SHA/report/decision, then stop.
