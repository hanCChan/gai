# P2C results: post-hoc ISR-Multiclass control — 2026-10-21

Protocol `docs/receiver_p2/PROTOCOL_P2C_ISR.md` (committed before running, e72c7f3).
Inference + linear-head refit on frozen features; no encoder training. 3 base checkpoints × 2 folds ×
3 seeds × 3 arms. `reports/receiver_p2c/{isr_posthoc,summary}.csv`.

## Results (outer RX, mean of 3 seeds, %)

| base features | frozen head f0/f1 | + ISR projection | + ISR norm-filtered | removed dirs (f0/f1) |
|---|---|---|---|---|
| ce | 82.67 / 85.45 | 81.74 / 87.37 (−0.93/+1.92) | 81.80 / 87.84 (−0.87/+2.39) | 40 / 40 |
| rxlogit | 84.39 / 89.21 | 83.88 / 88.80 (−0.51/−0.41) | 83.32 / 89.23 (−1.07/+0.02) | 40 / 40 (25–40 filtered) |
| rxnull | 85.21 / 89.46 | 84.35 / 89.24 (−0.86/−0.23) | 84.21 / 89.24 (−1.00/−0.22) | 40 / 40 |

## Reading (per the pre-registered interpretation)

1. **The post-hoc route does not reproduce the trained arms.** On CE features the ISR projection buys
   +1.9/+2.4 pp on fold 1 but loses −0.9 pp on fold 0 and loses on the worst RX (74.8→69.5 fold 0), i.e.
   it does not reach ls (82.95/87.20) or ccalign (84.91/89.00) on both folds. Projecting the frozen
   features is not equivalent to training with the constraint.
2. **On rxlogit / rxnull features there is nothing left to project out**: ISR and norm-filtered ISR are
   flat to −1 pp — consistent with the audit (κ ≈ .002–.004 after training; the frozen features carry
   almost no decision-visible RX displacement). This is direct evidence that the trained arms did the
   work, rather than the post-hoc linear fix being a sufficient competitor.
3. Worst-RX behaviour degrades slightly under all post-hoc projections (fold 0: −0.9 to −5.3 pp),
   another sign that hard projection on frozen features over-removes.

Standing claim fixed by this control: the paper reports ISR-Multiclass as a competitor, and states that
on both datasets-to-date the end-to-end exposure family beats it on accuracy and worst-RX. It does **not**
claim the principle is new (NOVELTY.md §1).
