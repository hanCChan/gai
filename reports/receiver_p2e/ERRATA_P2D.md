# P2D errata (P2E-0, 2026-10-22)

Recorded after the external review of 23dbc03 and a re-read of `scripts/receiver_p2/p2d_losses.py`.
Original P2D results and decision files are left unchanged; this file says how to read them.

## E1. `rxgeo`: worst-direction selection is inverted (implementation bug) → r1 `rxgeo` = INVALID ablation

```python
proj = dl @ rows.T              # (R, K-1): margin change of shift r against competitor u
hurt = (-proj).amin(1)          # BUG: smallest contraction per RX
worst[t] = dl[hurt.argmax()]
```

The docstring says "worst margin contraction across all competitor pairs", which is
`(-proj).amax(1)` followed by `argmax` over RX. `amin` picks, per RX, the competitor the shift hurts
least, so the arm optimised a different (and weaker) quantity. The r1 `rxgeo` row (beats ce 2/6) is
therefore **not** evidence that a geometry-only worst-shift penalty fails. It is marked INVALID; it was
never a main arm and no decision depended on it.

## E2. `rxccp` / `rxccp_soft`: the "source-RX → source-RX transfer" is not a transfer

The implementation adds `delta_{t,s} = A(mu_{t,s} - mu_t)` to the logits of every class-t sample,
whatever receiver r it came from. Under the additive model `z_{t,r,i} = zbar_t + delta_{t,r} + e_i`,
the actual counterfactual move r → s is `z - delta_{t,r} + delta_{t,s}`. The code applies
`z + delta_{t,s}`, i.e. the sample keeps its own receiver offset and gets a second one added
(s = r doubles the offset). The P2D report described the term as robustness to the worst *source*
receiver; it is more accurately robustness to an **extra** perturbation drawn from the source-shift
set (closer to extrapolation). The numbers stand; the semantic claim is withdrawn. Whether the
"double shift" helped (r1 beats ce 6/6) by acting as extrapolation was not tested.

## E3. `rxcant`: neither Cantelli nor per-receiver

- The loss is `s2 / (s2 + softplus(mu)^2 + eps)`. Cantelli's one-sided bound is
  `P(M <= 0) <= s2 / (s2 + mu^2)` and holds only for `mu > 0`; with softplus it is a smooth surrogate,
  not an upper bound, and it is not valid as a bound when `mu <= 0`.
- The docstring says per-RX cell statistics averaged over cells; the code concatenates all source-RX
  packets of class t (`torch.cat(cells)`) and takes one pooled mean/variance, so it penalises pooled
  (within + between RX) spread, not per-receiver tail risk.
r3's failure therefore shows that **this** pooled softplus surrogate does not transfer. It does not
show that a correctly specified per-receiver tail-risk objective fails. The same naming applies to
`tail_audit.py` ("Cantelli" columns there are the softplus surrogate).

## E4. r1 STOP_TAIL vs the chain continuing

`decision_p2d_r1.json` says STOP_TAIL (no arm beat ls). r2 was then launched by hand; the chain's
ITERATE logic did not exist yet. Already disclosed in the P2D REPORT §2; repeated here for completeness.

## E5. Two protocol limits that apply to all of P2–P2D (new)

- **Selection RX.** λ was selected on one held source RX per fold (rtl_6 in fold 0, Pluto in fold 1),
  and the search-stage epoch was selected on that same RX. Both held RX are low-noise receivers, so the
  selection could not see the noisy-RTL failure type (see P2E REPORT §2). P2E replaces this with nested
  leave-one-source-RX-out selection, with epochs selected on the fitting RX only.
- **Horizon.** 66/138 P1–P2D formal fits selected an epoch ≥ 18 of 20 (33 at exactly 20); `ls` averages
  19.3. Arm differences of ~1 pp are confounded with convergence speed. P2E adds a 40-epoch check.
- **Resolution.** Median seed std of outer accuracy is 0.79 pp; with 3 paired seeds the SE of a
  difference is ≈ 0.65 pp. The P2B/P2D "ties" with ccalign (|Δ| ≤ 1.3 pp) are within noise in both
  directions. P2E uses 5 seeds and the 8/10 paired rule.
