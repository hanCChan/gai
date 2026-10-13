# Minimal actual-Adam update control: source-only definition

This is a concrete R2-B engineering implementation, frozen before aggregate
C2 analysis. No controlled model fitting or R2-C is authorized or performed.
Read-only replay on R2-A snapshots is a local behavior diagnostic, not a
counterfactual controlled trajectory or an estimate of generalization gains.

## Proposal and measurements

Let θ,S be parameters and complete Adam state before a step. The same source
fitting pool, balanced batch schedule and A/B/C rotation is used in all arms.
A/B generate the historical second-order MLDG objective (WiSig includes its
two complementary date episodes), then clipping at5 and the actual Adam step
produce θ₀,S₀. Define δ=θ₀−θ and candidates θ(s)=θ+sδ for
s∈{1,1/2,1/4,0}. C1 and C2 are packet-disjoint source training pools, never
the source40 validation packets or target RX. Risk estimation is eval-mode CE;
paired train-mode Dropout results remain a separate diagnostic.

For RX r and C1 packet i define dᵣᵢ(s)=CEᵣᵢ(θ(s))−CEᵣᵢ(θ). Let
Dᵣ(s)=meanᵢdᵣᵢ(s), σᵣ(s)=sampleSDᵢdᵣᵢ(s)/√nᵣ, and
εᵣ(s)=2σᵣ(s)+10⁻⁷+10⁻⁶meanᵢ|CEᵣᵢ(θ)|. All class/date cells have
equal packet count, so this RX mean gives equal cell weighting. This paired
standard error is a descriptive finite-packet noise estimate, not a guarantee
under correlated packets or repeated adaptive testing. The multiplier2 and
numeric floor are fixed; neither C2 nor internal RX selects them. No τ/λ search.

Define D̄(s)=meanᵣDᵣ(s), ε̄(s)=meanᵣεᵣ(s). Traverse nonzero scales in
descending order and accept the first that satisfies:

1. D̄(s)≤ε̄(s).
2. maxᵣ[Dᵣ(s)−εᵣ(s)]≤max(0,−D̄(s))+ε̄(s).

If none pass, choose0. Rule1 bounds average deterioration using source
estimation error. Rule2 limits each RX's excess beyond its noise margin by
the observed average improvement budget. It permits positive individual RX
changes and does not demand Dᵣ≤0 everywhere. This is one fixed engineering
hypothesis, not an asserted optimal or novel decision rule.

Ordinary source-risk backtracking uses only rule1, with the same scales, C1,
noise estimator and Adam-state semantics. That matched control isolates the
incremental multi-RX guard. It is a finite empirical backtracking control,
not an Armijo convergence guarantee: the Adam proposal need not be a descent
direction on C1. C2 is read only after scale selection; its results neither
choose scales nor alter the definition. Unsupported/nonfinite C1 cells reject
and produce an engineering flag, without target-data substitution.

## Exact optimizer-state semantics

Before making the proposal, clone all parameters and the entire Adam state,
including per-parameter steps, first/second moments and parameter groups.
For s>0 commit θ+sδ and keep S₀. Moments therefore describe the full proposed
gradient; scaling does not pretend to be a different Adam learning rate history.
For s=0 restore θ and S exactly. Restore relevant model buffers as well if
future architectures add them; current backbones have none and measurement
asserts this. RNG consumed by the fitting objective advances normally; proposal
rejection does not rewind sampling. Risk measurement uses isolated RNG and
restores every module's train/eval flag. Deterministic tests cover rollback,
the next Adam update, scaling moments and Dropout reproducibility.

Implementation: src/receiver_r2/controller.py and measurement.py.
The executable controlled_adam_step transaction receives already-computed
A/B clipped gradients, creates the full Adam proposal, measures C1, fixes the
scale, then reads C2 and commits with the documented state policy. Composite
CPU tests confirm changing C2 labels cannot change parameters, moments or scale.
R2-A fitting accepts scale1 exclusively. Thus its100% acceptance cannot be
reported as the behavior of this controller. Offline choices on its frozen
snapshots are separately labelled and provide no cumulative controlled model.

## Fair future matrix and remaining technical limitations

The first four arms are standard MLDG (LR0.0003), fixed-small-LR MLDG
(LR0.00015), ordinary mean-risk backtracking and this multi-RX guard. All must
use the exact same fitting/C1/C2 packets, normalization, RX exclusion rotation,
paired initialization, steps and source-only model selection. Historical SWAD
trained on a larger fitting pool and is contextual evidence only. To test the
strongest implemented alternative fairly, add matched-pool SWAD as a fifth
arm: WiSig three fits plus Shen two folds×three seeds per arm yields45 fits,
instead of the four-arm minimum36. This is a proposed future matrix requiring
separate approval, not a request to start it now. Horizon must be frozen after
R2-A assessment and before estimating/approving R2-C.

C1 is a training-control signal; C2 is a development diagnostic. Reusing C2
to revise this policy would invalidate any subsequent claim of independent
remeasurement on that pool. Record any revision and require a new frozen
validation protocol rather than silently claiming independence. C1/C2 do not
establish risk on an unseen deployment RX, hardware causation or paper novelty.
The diagnostic runs cover one seed and early training, not a controlled trajectory
or late-stage all-rejection stability. Those questions remain for an explicitly
approved PoC, with effective updates, stage acceptance, tail/seed performance,
elapsed cost and engineering stop conditions all disclosed.
