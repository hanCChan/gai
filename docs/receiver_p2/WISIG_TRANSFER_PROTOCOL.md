# WiSig transfer protocol for the exposure family (draft for approval, 2026-10-21)

Status: **not started**. Writing and a single-job cost probe only; fits need explicit approval.
Base commit: the P2B commit on `experiment/p2-rx-null`. Shen record fixed in `reports/receiver_p2b/`.

## 0. Why WiSig, and what it decides

Shen is repeatedly-developed data (fold outer RXs have informed every decision since P0). WiSig gives a
different instrument family (N210/B210/X310), a different backbone (a 180-parameter-per-filter CNN in
quadrature, non-equalised IQ), a larger packet store, and 10 internal RXs for development while 8 RXs,
date 3 and the equalised variant stay sealed (registration `roles`, unchanged since 2026-10-13).
The question is narrow and pre-declared: **does any exposure-reducing arm beat both ls and ccalign on
WiSig, as rxnull/rxlogit did on Shen over ce/ls (but not over ccalign)?**

## 1. Data and splits (registration-derived; nothing new is unsealed)

- Source RX (14, labelled, fitting): `roles.source`; internal RX (10, development confirmation only):
  `roles.internal_rx`; final RX (8), date 3, equalised: still sealed, not read.
- Train: source RX × date {0,1} × encoder TX {0..7}, positions [0,120). Validation (source only):
  same cells, positions [120,160). Strict-held TX {8,9} excluded throughout (`prior_strict_tx_exclusion_preserved`).
- Development folds (source-only, defined from the 14 source RX, frozen in
  `configs/receiver_p2/wisig_folds.json` when this protocol is approved):
  - **W0** = the existing registration split, reused unchanged: train `wisig_inner_train`
    (10 RX), held `wisig_inner_held` = {1, 7, 16, 30}.
  - **W1** = a disjoint second split, stratified: held = the next 4 source RX by registration order
    that are not in `wisig_inner_held`, keeping at least one B210 and one N210 in both train and held;
    train = the other 10. Frozen the same way.
  - Honest limitation, stated now: the 14 source RX are 6 B210, 7 N210 and **1 X310** (index 22), so
    a strict leave-one-hardware-family-out fold is impossible; no hardware family is fully unseen in
    either fold, and the paper must not claim family-level generalisation from WiSig development folds.
    Family-level behaviour is reported descriptively (per-family held accuracy), and the sealed final
    8 RX (which do include X310 units) remains the only family-held-out evidence, untouched here.
- The 10 internal RX: **development confirmation after the method and λ are frozen** — they may not
  select λ, method, horizon, or seeds. If anything is changed after inspecting them, that inspection is
  reported as development data, not confirmation.

## 2. Horizon rule (fixed before any fit, source-only)

Run one CE search job (fold W0) for up to 150 epochs, record source-validation receiver-equal CE over
that fold's 10 training source RX at positions [120,160). Horizon h = the smallest value in
{50, 100, 150} at which the validation CE of the last 10 epochs is within 1% relative of its minimum.
All arms and all folds use h. If validation CE is still decreasing at 150, h = 150 and every result is
labelled horizon-limited (as in Shen, where rxnull best epochs sit at 20). No arm gets a different
horizon; no early stopping. Checkpoint = minimum source-validation receiver-equal CE (existing rule).

## 3. Arms (6 × 2 folds × 3 seeds = 36 formal fits; 5 × 3 λ × 2 folds = 30 search fits)

| arm | loss | note |
|---|---|---|
| ce | CE + weight decay as the registered CE | existing search artifacts reused for lr (0.0003 / 0.0005), no refit |
| ls | CE, label smoothing 0.1 | P1 control |
| ccalign | CE + λ·mean‖D_enc‖²/mean‖C‖² on the 80-d encoder output of `OfficialCNN.encode` | CIDG first moment; the arm that tied the method arms on Shen |
| rxlogit | CE + λ·mean‖A_enc·D_enc‖²/(mean‖A_enc·C‖²+eps), A_enc = row-centred centred head | function level, main candidate |
| rxnull | CE + λ·mean‖P_row D_enc‖²/(‖D_enc‖²+eps) | P2 feature-level form |
| rxnull_rand | rxlogit numerator with a random 9-dim subspace of the 80-d space, fixed per fold (rng 20261021+fold) | subspace-necessity control; reported for rxlogit, not stacked with it |

Losses operate on `encode(x)` (pre-dropout, post-ReLU 80-d) with the live head; λ ∈ {0.3, 1, 3}, chosen
per fold by minimum source-validation receiver-equal CE on the **training** source RX (positions
[120,160)) — never on the held fold or internal RX. Same seeds {20261013..15} as all other phases.

## 4. Metrics and decision

Primary: receiver-equal accuracy on the held development fold's 4 RX (W0) / 4 RX (W1); report over
seeds. Secondary: macro-F1, worst held RX, per RX×TX confusion, per-hardware-family accuracy, κ and
ρ computed on the held fold's source-side packets, and **κ_out realized on the held RX** (the
function-level analogue of the Shen audit), plus P-ISR post-hoc on frozen `ce` and `rxlogit` features.

"X beats Y" = Shen P2 rule (seed-mean Δ>0 in both folds and ≥5/6 paired). Decision, pre-declared:
- **Method line confirmed** iff some arm beats ce, ls AND ccalign on both folds by the rule.
- **Partly confirmed** if an arm beats ce and ls but not ccalign → the paper reports exposure reduction
  as a mechanism finding, and the method claim is "equal to class-conditional alignment with a
  decision-subspace formulation", not "better".
- **Not confirmed** if no arm beats ls → the Shen gain does not transfer; paper becomes diagnosis +
  negative result + reporting standard, with WiSig as the counter-example.
- Pluto-like worst-RX behaviour is reported separately, always.

## 5. Budget

Estimates. The registration records a 12-fit planning figure of **0.2700 reserved GPUh for 12
same-scale standard models** ("62.3167 s for the first job, 1.3 margin"), i.e. ≈ 62 s of GPU time per
short fit is the only measured datum on this machine; the W0 100-epoch formal at the registered lr is
larger than that planning job and must be measured, not extrapolated. A 3-epoch timing probe
(`scripts/receiver_p2/wisig_cost_probe.py`, CE and rxlogit, no checkpoint kept, no held/internal RX
evaluated) produces the s/epoch number; **it still has to be run** (blocked in the authoring session).

Planning bound until measured, stated explicitly as an extrapolation: at the P1-style device accounting
the 30 search (50 ep) + 36 formal (100 ep) fits are estimated at **6–12 GPUh**; the running P2-family
total is 4.19 (P2) + 1.3 (P2B) job-h. **Nothing starts until the user approves this document and the
measured budget**, and the W0 CE convergence run for the horizon rule (§2) is included in the request.
