# Prior-work check and claim boundaries

Primary sources revisited for this stage; these checks do not add fitting jobs.

| Work | Existing mechanism | Required distinction for this experiment |
|---|---|---|
| [MLDG, AAAI2018](https://ojs.aaai.org/index.php/AAAI/article/view/11596) | Simulates domain shifts within source training through a meta-learning objective. | Receiver/date partitioning and an Adam outer optimizer do not establish novelty. |
| [Fishr, ICML2022](https://proceedings.mlr.press/v162/rame22a.html) | Matches domain gradient variances to support out-of-distribution generalization. | This controller does not implement Fishr; no claim to outperform it is supported. |
| [PCGrad, NeurIPS2020](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf) | Projects conflicting task gradients before combining them. | Direct finite Adam loss measurements differ operationally, but need an empirical comparison to explain value beyond gradient surgery. |
| [CAGrad, NeurIPS2021](https://proceedings.neurips.cc/paper/2021/file/9d27fdf2477ffbff837d73ef7ae23db9-Paper.pdf) | Eq2 already defines worst per-task finite loss change, approximated to first order; Eq3 searches near the mean gradient for worst local improvement. | Finite losses, multiple RX and bounded updates are not new by themselves. Restricted scalar scaling is less expressive than direction optimization. |
| [SWAD, NeurIPS2021](https://proceedings.neurips.cc/paper/2021/file/bcb41ccdc4363c6848a1d760f26c28a0-Paper.pdf) | Dense parameter averaging with loss-valley selection seeks flat solutions. | Strongest historical Shen alternative must be compared using matched pools and disclosed computation. |
| [Group DRO, ICLR2020](https://arxiv.org/abs/1911.08731) | Optimizes worst-group training performance; regularization matters for worst-group generalization. | Protecting source groups is an existing idea; risk estimation must establish more than another group weighting. |
| [Boyd & Vandenberghe, Chapter9 slides](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf) | Backtracking shrinks a step until a prescribed sufficient-decrease condition holds. | Candidate scaling plus acceptance is line-search engineering. Our empirical noise guard supplies no Armijo convergence theorem for Adam. |
| [Burke optimization notes, trust regions](https://sites.math.washington.edu/~burke/crs/516/) | Existing trust-region and line-search frameworks control optimization proposals. | Bounded proposals and reliability checks alone cannot establish an independent algorithm contribution. |

The currently implemented finite-scaling rule is an engineering hypothesis.
C1/C2 packet separation improves measurement validity; it is not a novel
domain-generalization theorem. Its noise margins are descriptive and may be
miscalibrated under correlated packets or adaptive reuse. No new direction is
learned, and existing optimizer behavior may explain any later benefit.

R2-A tests convergence and measurement validity only. R2-B supplies exact code
and optimizer-state semantics, with offline behavior diagnosed separately.
No controlled fit, Fishr/CAGrad replication or formal R2-C comparison has occurred.
Reliable improvement over small LR, mean-risk backtracking and matched SWAD,
across seeds/datasets/tails with disclosed costs, remains necessary before
developing a method-paper claim. Source pool CV is not an independent blind test.
