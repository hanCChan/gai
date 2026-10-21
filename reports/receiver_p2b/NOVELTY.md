# Novelty assessment for the exposure-reducing family (2026-10-21)

Sources verified against primary pages (arXiv API/HTML, Crossref, PMLR, AAAI OJS, author code) by a
background verification pass; anything it could not open is marked UNVERIFIED. Purpose: fix what the
paper may and may not claim, before WiSig.

## 1. Prior art that already contains the core principle

| work | citation | mechanism | relation to our loss |
|---|---|---|---|
| **CIP-mean** | Chen & Bühlmann, arXiv 2010.15764 (JMLR-accepted per arXiv comment; volume UNVERIFIED) | Constraint E_m[Xᵀβ\|Y] = E_1[Xᵀβ\|Y] for all source environments m; finite-sample penalty Σ_{m,k}(mean z_cond^(m)·β − mean z_cond^(k)·β)². Frozen last-layer features, head only | **For a linear head this is the numerator of our loss set to zero.** Unnormalised, so shrinking the displacement also satisfies it; no encoder training |
| **ISR-Multiclass** | Wang, Balasubramaniam, Si, Li, Zhao, arXiv 2311.00966 (extends ISR-Mean, ICML 2022) | Per-class per-environment means → per-class PCA giving span{D_{t,e}} → SVD of stacked bases → spurious subspace P′ → fit the linear classifier in Null(P′) | Hard post-hoc version of the same idea, frozen features, no end-to-end training. **Our §P2C control P-ISR implements it** |
| **MASF** | Dou, Castro, Kamnitsas, Glocker, NeurIPS 2019, arXiv 1910.13580 | s_c^(k)=softmax(T(z̄_c^(k))/τ); symmetric KL between domains averaged over classes, in a meta-loop | Aligns mean *logits* across environments — the centred part is exactly our head row space. Distinction: alignment (pulls together) vs our normalised exposure (can rotate into the nullspace, not erase) |
| **INLP / LEACE** | Ravfogel et al., ACL 2020, arXiv 2004.07667; Belrose et al., NeurIPS 2023, arXiv 2306.03819 | Project features onto the nullspace of a *nuisance classifier* (iterated), or a closed-form affine eraser so no linear probe recovers the concept | Reverse direction: they erase the nuisance from features. We keep it and make the *task* head blind to it. Must be cited and distinguished |
| **JSE** | Holstege et al., ICML 2024, PMLR 235 | Jointly estimate orthogonal spurious/main-task subspaces, remove the spurious one | Spuriousness estimated from *labels of the nuisance*; we never define the nuisance subspace, only the head's |
| **Orthogonal classifier** | Xu, He, Shen, Jaakkola, ICLR 2022, arXiv 2201.11259 | A classifier using only directions another classifier ignores | Variation control, not DG |
| **CoRE** | Heinze-Deml & Meinshausen, arXiv 1710.11469 (venue UNVERIFIED) | Penalise conditional variance of logits given (Y, ID) | Variance of logits per class across environments, no projection geometry |
| **NAP** | Solomonoff et al., ICASSP 2005 | Speaker-verification channel compensation by projecting out a nuisance subspace | Classic precedent for channel nuisance subspace projection in speaker ID |
| **CIDG** | Li, Gong, Tian, Liu, Tao, AAAI 2018 | P(h(X)\|Y) invariant across source domains | Our `ccalign` control implements its first-moment version |
| **LAM** | Gao et al., UAI 2024, arXiv 2305.07888 | Attribution-consistency regulariser on semantic-sharing augmentations, weight w²_{u,y} | Per-unit diagonal weighting on true-class column; no subspace projection, needs augmentation pairs |
| **SWAD** | Cha et al., NeurIPS 2021 | Dense weight averaging, flat minima | Optimisation stability only |
| **RIEI** | Zhang et al., IEEE IoT-J 2025, DOI 10.1109/JIOT.2024.3486068, arXiv 2411.03636 | Feature split [Z_X, Z_ν]; CE + λ1·L_MI − λ2·L_IE (per-sample orthogonality + entropy) | Per-sample orthogonality between feature halves; no head geometry, no class-conditional means |
| **IB-RFF** | Yu Wang, IEEE TIFS vol. 21, 2026, DOI 10.1109/TIFS.2026.3723081; code github.com/BeechburgPieStar/IB-RFF | FAN backbone; loss = CE + α·HSIC_nCCA(hidden, input) per the repo (paper text UNVERIFIED) | Compression against the input; no receiver/label structure, no class means |
| **DRIFT v2** | Pan et al., IEEE TCCN 2026 early access, DOI 10.1109/TCCN.2026.3737891, arXiv 2510.09405v2 | CE(TX)+CE(RX), GRL receiver discriminator, per-receiver centre loss, MSE separation | Erasure + separation; no classifier geometry |

## 2. What we may claim (fixed now, binding for the paper)

**May claim**
1. Cross-receiver failures concentrate on directional pairs predictable from source RX (P0/P1, both folds,
   WiSig mirror) — diagnosis, independent of any regulariser.
2. A measured mechanism quantity: the per-RX, per-class mean displacement visible to the classifier,
   normalised by the decision-space class separation (κ; source-side ρ as its feature-space surrogate),
   is much smaller under every arm that improves unseen-RX accuracy, and unchanged or larger under
   margin-protection arms (P1 negative result) and under label smoothing (whose ρ drop is null-space
   inflation, P2A §1.4). Over 117 checkpoints this is the most consistent mechanism account we have.
3. Label smoothing must be reported as a control; over the 117 checkpoints its apparent "exposure
   reduction" is not a reduction of decision-relevant displacement.
4. Normalised, end-to-end exposure minimisation is a distinct *formulation* from CIP-mean / ISR /
   MASF: the normalisation forbids the shrink-to-zero solution, the encoder is trained, and the RFFI
   application is new. Its empirical advantage over plain class-conditional alignment (ccalign) is
   **not shown on Shen** (P2B: +0.30/+0.46, 3/6) and must not be claimed without WiSig evidence.

**May not claim**
- "First to make environment-induced class-conditional shifts invisible to the classifier" — CIP-mean,
  ISR-Multiclass and MASF precede it.
- "ρ is the intrinsic risk / a lower bound / all argmax damage" — P2A errata E1–E3.
- "Classifier geometry beats class-conditional alignment" on Shen evidence.

## 3. Controls the paper must carry

Required: ce, ls, ccalign (CIDG first moment), cmean (unnormalised projected, = head-level CIP),
rxlogit, rxnull, random-subspace direction, and **P-ISR** (post-hoc ISR-Multiclass on frozen CE and on
frozen rxlogit features) — see `scripts/receiver_p2/isr_posthoc.py`. WiSig additionally: SWAD and MLDG
as optimisation-stability controls if budget allows, IB-RFF/RIEI only if reproduction is faithful.
