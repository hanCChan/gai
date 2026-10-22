"""P2D losses: worst-source-shift margin robustness (PROTOCOL_P2D §1). Pure functions, CPU
unit-tested in test_p2d_losses.py. Same notation as p2_losses: pre = pre-head features,
W = live classifier weight (K, d), A = W - mean_rows(W).

Motivation (audit-driven, see reports/receiver_p2b + tail_audit): kappa/rxlogit penalises
||A D_{t,r}||^2 - the squared displacement of class means - which cannot distinguish a
beneficial shift (pushes the true class ahead of its competitors) from a harmful one (flips
an argmax), and ignores within-class margin spread entirely. Cantelli on the true margin
M_{tu}(x) = z_t - z_u:

    P(M_{tu} <= 0 | t, r) <= sigma^2_{tu,r} / (sigma^2_{tu,r} + mu^2_{tu,r})

says the remaining cross-RX error after mean alignment is governed by margin TAIL risk, not
mean displacement energy. rxccp targets exactly that quantity, without any hard-coded pair
list (P1's lesson) and without null-inflation (P2A's lesson):

    rxccp:  lambda * mean_t max_{delta in shifts(t)} CE(z + delta, t)   (per-batch, all classes)

where shifts(t) are the K-1 per-source-RX *logit-space* mean shifts of class t relative to
its batch class mean: delta_{t,r} = A (mu_{t,r} - mu_t). Applying delta to the WHOLE batch
share of class t approximates "every packet of class t lands on RX r's shifted decision
geometry". Common-mode is removed by A automatically (rows of A sum to 0), and the inner
max is exactly a minimax over the observed source-shift simplex - the empirical analogue of
robustness to the worst source receiver, which is what failed on Pluto. No normalisation
denominator exists to game: the only way to lower the term is to make every source shift
leave the true class winning on the actual CE, including its tail samples.

  tailrisk (diagnostic, not trained): mean_{t,u} softplus-tilded Cantelli ratio
        sigma2 / (sigma2 + softplus(mu))  per (t,u,r) on batch margins. Used by
        tail_audit.py on frozen checkpoints and by train_p2.py logging; never a loss.

  rxccp_ls: CE with label smoothing 0.1 (P1 convention) instead of plain CE.
"""
import torch
import torch.nn.functional as F


def centered_rows(W):
    return W - W.mean(0, keepdim=True)


def class_logit_shifts(pre, W, y, rb, min_rx=2):
    """delta_{t,r} = A (mu_{t,r} - mu_t) for every class t seen on >= min_rx RX of the batch.
    Returns a dict t -> (R_t, K) tensor of logit-space mean shifts (common-mode-free by A).
    E_r delta = 0 exactly on balanced batches."""
    A = centered_rows(W)
    out = {}
    for t in torch.unique(y):
        mus = []
        for r in torch.unique(rb):
            m = (y == t) & (rb == r)
            if bool(m.any()):
                mus.append(pre[m].mean(0))
        if len(mus) < min_rx:
            continue
        mus = torch.stack(mus)
        D = mus - mus.mean(0, keepdim=True)
        out[int(t)] = D @ A.T
    return out


def _shifted_ce(logits, yb, delta_for, base_ce=None):
    """mean over batch of CE(z + delta_{y_i, worst-r}, y_i); worst-shift per class chosen by
    per-class mean CE over that class's candidate shifts (minimax over the observed shift
    simplex). Falls back to `base_ce` for classes with no samples in the batch."""
    worst = {}
    for t, dl in delta_for.items():
        m = yb == t
        if not bool(m.any()):
            if base_ce is not None:
                worst[t] = logits.new_zeros(logits.shape[1])
            continue
        if dl.shape[0] == 1:
            continue
        ces = torch.stack([F.cross_entropy(logits[m] + d, yb[m], reduction='mean') for d in dl])
        worst[t] = dl[ces.argmax()]
    if not worst:
        return logits.sum() * 0.
    z = logits.clone()
    for t in worst:
        if bool((yb == t).any()):
            z[yb == t] = z[yb == t] + worst[t]
    return F.cross_entropy(z, yb)


def rxccp(logits, pre, W, y, rb, min_rx=2):
    """P2D main arm: max over source shifts of shifted class CE. Returns 0 if no class has
    >= min_rx receivers in the batch (pre.sum()*0 keeps the graph alive)."""
    shifts = class_logit_shifts(pre, W, y, rb, min_rx)
    if not shifts:
        return logits.sum() * 0.
    return _shifted_ce(logits, y, shifts)


def rxccp_soft(logits, pre, W, y, rb, min_rx=2, beta=1.0):
    """P2D r2 candidate: SOFTMAX relaxation of the inner max. Instead of committing to one
    worst shift per class (r1 post-mortem: the argmax collapses onto one hard receiver per
    class - b210 for 7/10 classes on fold 0 - reproducing P1's fixed-hard-set failure), the
    class damage is the beta-temperature log-sum-exp over that class's candidate shifts:

        per class t:  (1/beta) * log mean_r exp(beta * CE(z_t + delta_{t,r}, t))
        total:        mean_t of the above  (a smooth max = soft distributionally-robust CE)

    beta=0 reduces to the plain mean over shifts (fully soft); beta=inf -> r1's hard argmax.
    beta=1 with mean (not max) inside the exp keeps all source RX in play while still
    weighting the harder ones exponentially."""
    shifts = class_logit_shifts(pre, W, y, rb, min_rx)
    if not shifts:
        return logits.sum() * 0.
    total = logits.sum() * 0.
    n_c = 0
    for t, dl in shifts.items():
        m = y == t
        if not bool(m.any()):
            continue
        if dl.shape[0] == 1:
            continue
        ces = torch.stack([F.cross_entropy(logits[m] + d, y[m], reduction='mean') for d in dl])
        if beta <= 0:
            soft = ces.mean()
        else:
            # (logsumexp(beta*ces) - log n) / beta; the earlier form subtracted log n outside the 1/beta
            # (audit B3) and was correct only at beta = 1, the only value used in r2
            soft = (torch.logsumexp(beta * ces, 0) - torch.log(torch.as_tensor(float(len(ces)), dtype=ces.dtype, device=ces.device))) / beta
        total = total + soft
        n_c += 1
    if n_c == 0:
        return logits.sum() * 0.
    return total / n_c


def rxcant(logits, pre, W, y, rb, min_rx=2, eps=1e-6):
    """P2D r3 candidate: train the AUDIT STATISTIC directly. tail_audit showed the Cantelli
    ratio  s2/(s2 + softplus(mu)^2)  predicts outer accuracy (Spearman -0.60/-0.65, beats
    kappa given q10 partial) and tracks realised flip rates (0.73-0.75); r1/r2 showed that
    mean-shift-targeting arms (hard or soft max) leave worst-RX tail risk untouched. This
    loss minimises the batch-level per-(class, competitor) Cantelli ratio of the TRUE margins
    M_{t,u} pooled over source RX cells: for each class t (>= min_rx RX present), per-RX mean
    mu_{t,u,r} and variance s2_{t,u,r} of z_t - z_u are computed per RX cell, averaged over
    cells (pooled risk across source RX), and the tilded ratio is averaged over (t, u).

    This constrains the statistic that (a) explains remaining error after ccalign and (b)
    cannot be lowered by class-separation inflation alone (mu enters via softplus(mu)^2
    against s2; growing separation DOES lower it, but only through the actual margin mean,
    which is exactly the safety quantity we want, including its tail samples)."""
    K = logits.shape[1]
    vals = []
    for t in torch.unique(y):
        cells = []
        for r in torch.unique(rb):
            m = (y == t) & (rb == r)
            if not bool(m.any()):
                continue
            z = logits[m]
            comp = torch.stack([z[:, int(t)] - z[:, u] for u in range(K) if u != int(t)], 1)
            cells.append(comp)
        if len(cells) < min_rx:
            continue
        comp = torch.cat(cells, 0)                    # all packets of class t, all RX pooled
        mu = comp.mean(0)                            # (K-1,) pooled mean margin per competitor
        s2 = comp.var(0, unbiased=False)             # pooled variance (within + across RX)
        vals.append(s2 / (s2 + F.softplus(mu).pow(2) + eps))
    if not vals:
        return logits.sum() * 0.
    return torch.stack(vals).mean()


def rxgeo(logits, pre, W, y, rb, min_rx=2):
    """P2D geometry-only variant: same per-RX worst-shift structure, but the inner "damage"
    is measured by the shift's own margin-geometry energy max_{r} max_{u!=t} -w_hat_tu . delta_{t,r}
    (worst margin contraction across all competitor pairs) instead of a shifted CE. Isolates
    "where the shifts point" from "what the softmax tail does with them" - the ablation that
    separates rxccp's CE-level mechanism from a pure geometric penalty."""
    shifts = class_logit_shifts(pre, W, y, rb, min_rx)
    if not shifts:
        return logits.sum() * 0.
    A = centered_rows(W)
    dim = W.shape[1]
    K = W.shape[0]
    worst = {}
    for t, dl in shifts.items():
        if dl.shape[0] == 1:
            continue
        # margin contraction of shift delta (a K-vector) against pair direction m_{tu}:
        # m_{tu} . delta = (A_t - A_u) . (D @ A^T) = (A_t - A_u) . A . D  (d in logit space
        # equals A D in feature space). Equivalent: project delta onto centred-row
        # differences expressed in LOGIT space. Since delta = A D, the margin change is
        # (e_t - e_u)^T delta (a pure logit-space inner product: margin t-u = z_t - z_u).
        # So: hurt_{r,u} = -(delta_r[u->index]) with e-metric: contraction = -delta_r[t-vs-u]
        # = -(delta_r[t] - delta_r[u]) is wrong scale-wise; the correct margin change is
        # exactly (e_t - e_u)^T delta_r because M_{tu} = z_t - z_u.
        eye = torch.eye(K, dtype=dl.dtype, device=dl.device)
        rows = torch.cat([ (eye[t] - eye[u]).unsqueeze(0) for u in range(K) if u != t ], 0)
        rows = rows / (rows.norm(dim=1, keepdim=True) + 1e-12)   # (K-1, K)
        proj = dl @ rows.T                                        # (R, K-1)
        hurt = (-proj).amin(1)                                    # worst contraction per RX
        worst[t] = dl[hurt.argmax()]
    z = logits.clone()
    for t, dd in worst.items():
        if bool((y == t).any()):
            z[y == t] = z[y == t] + dd
    return F.cross_entropy(z, y)

def margin_stats(logits, y, rb):
    """Batch margin tail statistics (diagnostic only, no gradient): for each class t with
    >= 2 RX in batch, per-RX mean mu and variance sigma2 of M_{tu} over the 90 (or K(K-1))
    ordered pairs, plus softplus-Cantelli risk. Returns dict with float tensors."""
    K = logits.shape[1]
    out = {}
    for t in torch.unique(y):
        mus, s2s, rs = [], [], []
        for r in torch.unique(rb):
            m = (y == t) & (rb == r)
            if not bool(m.any()):
                continue
            z = logits[m]
            # margins vs every competitor u != t: M_{t,u} = z_t - z_u (device-safe loop)
            comp = torch.stack([z[:, int(t)] - z[:, u] for u in range(K) if u != int(t)], 1)
            mus.append(comp.mean(0))
            s2s.append(comp.var(0, unbiased=False))
            rs.append(int(r))
        if len(mus) < 2:
            continue
        mu = torch.stack(mus)          # (R, K-1)
        s2 = torch.stack(s2s)
        out[int(t)] = dict(mu=mu, s2=s2, rx=torch.tensor(rs))
    return out


def tailrisk(logits, y, rb, eps=1e-6):
    """Mean over (t, pair u, r) of the tilded-Cantelli ratio
        s2 / (s2 + softplus(mu)^2)
    computed on batch margins (diagnostic; lower = safer tails). Means over R per pair first,
    then over pairs and classes. Returns None if no class has >= 2 RX."""
    stats = margin_stats(logits, y, rb)
    if not stats:
        return None
    vals = []
    for t, d in stats.items():
        mu = d['mu'].mean(0)           # (K-1,) mean margin per competitor over RX
        s2 = d['s2'].mean(0)
        vals.append(s2 / (s2 + F.softplus(mu).pow(2) + eps))   # squared, as in tail_audit (audit B4)
    return torch.stack(vals).mean()
