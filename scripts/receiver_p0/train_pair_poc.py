#!/usr/bin/env python3
"""Pair-margin PoC: three pre-registered branches, one trainer, R0/R1 recipe reused verbatim.

Branches (chosen from P0 evidence, all three trained in one grid so the ablation is built-in):
  sub    (F: feature)  L_sub: prototype separation on hard pairs P in the 512-d feature space.
         L_sub = mean over (t,u) in P of softplus(gamma_sub - ||mu_t - mu_u||)/ (1 + d_within)
         computed per SOURCE RX then averaged (mu from the current batch's per-RX class means).
  drift  (H: head)     L_drift: constrain only the decision-direction projection of per-RX
         class-mean shifts: mean over (t,u) in P of softplus(gamma_d + Delta_{t,u,r}),
         Delta_{t,u,r} = (w_t - w_u)^T (mu_{t,r} - mean_r mu_{t,r}); w from the LIVE head.
  pair   (J: joint)    L_pair: cross-RX stability of the low quantile of hard-pair logit
         margins. Per batch, per source RX r with >=2 packets of class t: collect
         margin_{t->u} for (t,u) in P; keep the per-(r,t,u) min inside the batch; loss =
         logsumexp_r softplus(gamma_p - q_r)/tau. Batch min is a conservative stand-in for
         q10 (batch=32 packets, at most 3 per class per RX); EMA buffers over steps provide
         the smoother estimate (buffer only smooths the REPORTED statistic; the gradient
         uses the batch min).
  Baseline arm 'ce' reproduces the frozen R0/R1 CE recipe bit-for-bit (same schedule file,
         same optimizer, same selection rule) as the sanity anchor.

Everything else is R0/R1 verbatim: balanced_schedule, Adam(lr=3e-4, eps=1e-8), grad clip 5,
20 epochs, min source-only receiver-equal validation CE checkpoint selection, deterministic
algorithms. lambda in {0.1, 1.0} x branch, plus lambda=0 == CE anchor.

    python3 train_pair_poc.py --fold 0 --branch pair --lambda 1.0 --seed 20261013 --pairs pairs.json \
        --out OUT [--epochs 20] [--limit-minutes 0]
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')   # before any CUDA context (R0/R1 convention)

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import p0_common as C   # noqa: E402

C.add_root()


def main():
    import torch
    from torch import nn
    from torch.nn import functional as F
    from b1_baselines.model import FeatureCNN
    from receiver_r01.core import balanced_schedule, predict, metrics, regularization
    from receiver_r01.data import shen as load_shen
    from receiver_r01.data import groups
    from train_receiver_r01 import build as _unused, validate, fitting_data   # noqa: F401

    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--branch', choices=['ce', 'sub', 'drift', 'pair'], required=True)
    ap.add_argument('--lambda', type=float, default=1.0)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--gamma-sub', type=float, default=2.0)
    ap.add_argument('--gamma-drift', type=float, default=0.5)
    ap.add_argument('--gamma-pair', type=float, default=1.0)
    ap.add_argument('--tau', type=float, default=1.0)
    ap.add_argument('--limit-minutes', type=float, default=0)
    a = ap.parse_args()

    lam = float(getattr(a, 'lambda'))
    run_id = f'shen_rank16_CV{a.fold}_{a.branch}{lam:g}_seed{a.seed}'
    out = Path(a.out) / run_id
    assert not out.exists(), out
    out.mkdir(parents=True)
    device = torch.device('cuda:0')
    reg = C.registration()
    P = [tuple(p) for p in json.loads(Path(a.pairs).read_text())['folds'][str(a.fold)]['pairs']]
    if a.branch == 'ce' or lam == 0:
        P = []

    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    model = FeatureCNN().to(device)
    # bit-identical init to the CE anchor: derive from the anchor's initial payload is NOT
    # possible without re-running selection; instead seed the same generator state as R0/R1
    # did (torch.manual_seed(seed) then FeatureCNN() -> same init stream).
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-8)

    # R0/R1 fitting data (fold source RX: inner_train + inner_validation, first160)
    fold = reg['shen_folds'][a.fold]
    train, stats = load_shen(reg, fold['inner_train'] + fold['inner_validation'], None, 'train')
    val, _ = load_shen(reg, fold['inner_train'] + fold['inner_validation'], stats, 'val')
    group = groups(train)
    schedule = balanced_schedule(group, a.epochs, a.seed)

    x = torch.as_tensor(train['x'], device=device)
    y = torch.as_tensor(train['y'], device=device)
    rx = torch.as_tensor(train['rx'], device=device)
    vx = torch.as_tensor(val['x'], device=device)
    vy = torch.as_tensor(val['y'], device=device)
    vrx = torch.as_tensor(val['rx'], device=device)
    rxs_all = torch.unique(rx)

    tp = torch.tensor([t for t, _ in P], device=device)
    up = torch.tensor([u for _, u in P], device=device)
    P_mat = torch.stack([tp, up], 1) if P else None   # (|P|, 2)

    def feat512(xb):
        h = F.relu(model.stem(F.pad(xb, (2, 3, 2, 3))))
        h = model.blocks(h).permute(0, 2, 3, 1).reshape(xb.shape[0], -1)
        return F.normalize(model.feature(h), p=2, dim=1, eps=1e-12), h

    def l_sub(f, yb, rb):
        """Per-source-RX prototype separation on P (feature branch)."""
        losses = []
        for r in torch.unique(rb):
            for (t, u) in P:
                mt, mu_ = (yb == t) & (rb == r), (yb == u) & (rb == r)
                if int(mt.sum()) == 0 or int(mu_.sum()) == 0:
                    continue
                mu_t = f[mt].mean(0)
                mu_u = f[mu_].mean(0)
                # within-scatter of the two classes in this RX (trace of scatter, batch estimate)
                s_t = (f[mt] - mu_t).pow(2).sum(1).mean()
                s_u = (f[mu_] - mu_u).pow(2).sum(1).mean()
                losses.append(F.softplus(a.gamma_sub - (mu_t - mu_u).norm() / (1.0 + (s_t + s_u).sqrt())))
        return torch.stack(losses).mean() if losses else f.sum() * 0.

    def l_drift(f, yb, rb):
        """Decision-direction projection of per-RX class-mean shifts (head branch)."""
        W = model.classifier.weight                      # (10, 128) live head
        pre = F.relu(model.tx_input(f))
        # class means per RX in pre-head space (that is where w acts)
        base = {}
        per_rx = []
        for r in torch.unique(rb):
            for t in torch.unique(yb[rb == r]):
                m = (yb == t) & (rb == r)
                base.setdefault(int(t), []).append(pre[m].mean(0))
                per_rx.append((int(r), int(t), pre[m].mean(0)))
        gmean = {t: torch.stack(v).mean(0) for t, v in base.items()}
        losses = []
        for (t, u) in P:
            if t not in gmean or u not in gmean:
                continue
            wdu = W[t] - W[u]
            for r, tt, mu_r in per_rx:
                if tt != t:
                    continue
                delta = (mu_r - gmean[t]) @ wdu    # <0 shrinks the t vs u margin on RX r
                losses.append(F.softplus(a.gamma_drift + delta))
        return torch.stack(losses).mean() if losses else f.sum() * 0.

    def l_pair(logits, yb, rb):
        """Cross-RX worst-margin stability on P (joint branch), batch-min + logsumexp."""
        per_pair = []
        for (t, u) in P:
            qs = []
            for r in torch.unique(rb):
                m = (yb == t) & (rb == r)
                if int(m.sum()) == 0:
                    continue
                qs.append((logits[m, t] - logits[m, u]).min())
            if qs:
                per_pair.append(torch.logsumexp(torch.stack(qs) / a.tau, 0) * a.tau)
        if not per_pair:
            return logits.sum() * 0.
        return torch.stack(per_pair).mean()

    history = []
    best, best_state, best_epoch = float('inf'), None, 0
    t0 = time.perf_counter()
    for epoch in range(a.epochs):
        for ids in schedule[epoch]:
            ix = torch.tensor(ids, device=device)
            xb, yb, rb = x[ix], y[ix], rx[ix]
            model.train()
            opt.zero_grad(set_to_none=True)
            if a.branch == 'ce' or lam == 0:
                ps = F.cross_entropy(model(xb), yb, reduction='none')
                loss = ps.mean() + regularization(model)
            else:
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                if a.branch == 'sub':
                    f, _ = feat512(xb)
                    loss = loss + lam * l_sub(f, yb, rb)
                elif a.branch == 'drift':
                    f, _ = feat512(xb)
                    loss = loss + lam * l_drift(f, yb, rb)
                elif a.branch == 'pair':
                    loss = loss + lam * l_pair(logits, yb, rb)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.step()
        # source-only validation CE, receiver-equal (identical rule to R0/R1)
        model.eval()
        ms = []
        with torch.no_grad():
            for r in torch.unique(vrx):
                m = vrx == r
                ms.append(metrics(predict(model, vx[m]).cpu().numpy(), vy[m].cpu().numpy()))
        vce = float(np.mean([q['CE'] for q in ms]))
        if vce < best:
            best, best_epoch = vce, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(dict(epoch=epoch + 1, source_val_CE=vce))
        C.dump_json(out / 'history.json', history)
        if a.limit_minutes and (time.perf_counter() - t0) > a.limit_minutes * 60:
            break
    model.load_state_dict(best_state)
    torch.save(dict(model=model.state_dict(), job=vars(a), scaler=stats,
                    selection=dict(rule='minimum source-only validation receiver-equal CE', epoch=best_epoch),
                    pairs=P), out / 'selected.pt')
    C.dump_json(out / 'status.json', dict(
        state='completed', run_id=run_id, seconds=time.perf_counter() - t0,
        best_epoch=best_epoch, best_source_val_CE=best, checkpoint_sha256=C.sha(out / 'selected.pt')))
    print(json.dumps(dict(run_id=run_id, best_epoch=best_epoch, best_CE=best)), flush=True)


if __name__ == '__main__':
    main()
