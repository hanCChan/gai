#!/usr/bin/env python3
"""P2E trainer: R0/R1 Shen recipe verbatim (FeatureCNN rank16, Adam 3e-4/eps 1e-8, balanced_schedule(seed),
clip 5, deterministic) + receiver-calibrated counterfactual augmentation (PROTOCOL in REPORT_P2E §4).

arms (lam = the searched scalar):
  ce, ls                 baselines (no lam)
  ccalign                CE + lam * ccalign (P2B loss, unchanged)
  noise                  CE on inputs with receiver-calibrated noise, scale lam
  noiseiso               control: same expected energy, flat (pixel-isotropic) profile
  ccanoise               ccalign (lam from --cca-lam fold map) + receiver-calibrated noise, scale lam
  sig                    CE on inputs whose receiver signature is replaced by a virtual one drawn in the
                         source signature span with extrapolation factor lam
  sigrand                control: same-norm offset in a random direction
  ncons                  (E6) CE on CLEAN inputs + lam * KL(p_clean.detach() || p_noisy), noisy view =
                         calibrated noise at fixed scale NCONS_SCALE on every packet
  nconsiso               (E6) control: same with the flat (isotropic) profile
  ccancons               (E6) ccanoise's ccalign term (on clean features) + ncons
Receiver statistics come ONLY from the fitting packets of the fitting RX (standardized units):
  v_r = per-pixel within-class variance on RX r; profile P = max_r (v_r - min_r' v_r')_+  (noise floor
  excess of the noisiest fitting RX over the cleanest, pixel-wise); c_r = class-balanced mean (signature).
  noise: x + sqrt(u * lam * P) * eps,  u ~ U(0,1), applied to each packet with prob 0.5.
  sig:   x - c_r + m + (lam * g * sigma_a) U,  g ~ N(0, I)   (U, sigma_a: span / spread of source c_r)

stages:
  loro    fit = the fold's 5 source RX minus --held (index 0..4); epoch selected on fitting-RX val packets
          (positions 160..199); the held RX is evaluated only at the selected epoch (never selects anything)
  formal  fit = all 5 source RX; epoch selected on their val packets (as P2)

    python3 train_p2e.py --stage loro --held 2 --fold 0 --arm noise --lam 2 --seed 20261016 --out OUT
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'receiver_p0')]
import p0_common as C   # noqa: E402
import p2_losses as L   # noqa: E402

C.add_root()
ARMS = ('ce', 'ls', 'ccalign', 'noise', 'noiseiso', 'ccanoise', 'sig', 'sigrand', 'ncons', 'nconsiso', 'ccancons')
# ccanoise / ccancons ccalign weight. E1/E6 used the P2B values {0: 3, 1: 1} (selected on one held RX),
# while the P2E ccalign anchor was selected by LORO at 3.0 in both folds -> fold-1 comparisons of the
# combined arms carried a 3x weaker alignment term (audit A4). New runs pass --cca-lam explicitly.
CCA_LAM = {0: 3.0, 1: 1.0}
NCONS_SCALE = 2.0            # E6: fixed before any E6 fit (middle of the E1 noise grid)
CONS = ('ncons', 'nconsiso', 'ccancons')


def run_id(a):
    st = f'loro{a.held}' if a.stage == 'loro' else 'formal'
    ep = '' if a.epochs == 20 else f'_ep{a.epochs}'
    return f'shen_rank16_CV{a.fold}_{a.arm}{a.lam:g}_Rq{ep}_{st}_seed{a.seed}'


def receiver_stats(x, y, rx):
    """Per-RX noise-floor profile and signature span from fitting packets (numpy, standardized units)."""
    rs = np.unique(rx)
    v = np.stack([np.mean([x[(y == t) & (rx == r)].var(0) for t in np.unique(y)], 0) for r in rs])
    prof = np.clip(v - v.min(0, keepdims=True), 0, None).max(0)          # (1, 52, 126)
    c = np.stack([np.mean([x[(y == t) & (rx == r)].mean(0) for t in np.unique(y)], 0) for r in rs])
    flat = c.reshape(len(rs), -1).astype(np.float64)   # float64: float32 SVD kept the null direction (audit B2)
    m = flat.mean(0)
    _, S, Vt = np.linalg.svd(flat - m, full_matrices=False)
    k = S > 1e-6 * S.max()
    U = Vt[k]
    sig_a = ((flat - m) @ U.T).std(0)
    return dict(prof=prof, c=c, rs=rs, m=m, U=U, sig_a=sig_a)


def main():
    import torch
    from torch.nn import functional as F
    from b1_baselines.model import FeatureCNN
    from receiver_r01.core import balanced_schedule, predict, metrics
    from receiver_r01.data import shen as load_shen, groups

    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['loro', 'formal'], required=True)
    ap.add_argument('--held', type=int, default=-1)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--lam', type=float, default=0.)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--p-aug', type=float, default=0.5)
    ap.add_argument('--cca-lam', type=float, default=-1., help='ccalign weight for ccanoise/ccancons; -1 = legacy CCA_LAM')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    assert (a.stage == 'loro') == (a.held >= 0)
    rid = run_id(a)
    out = Path(a.out) / rid
    assert not out.exists(), out
    out.mkdir(parents=True)

    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    model = FeatureCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-8)

    reg = C.registration()
    fold = reg['shen_folds'][a.fold]
    src = fold['inner_train'] + fold['inner_validation']
    fit = [k for i, k in enumerate(src) if i != a.held]
    train, stats = load_shen(reg, fit, None, 'train')
    val, _ = load_shen(reg, fit, stats, 'val')
    held = load_shen(reg, [src[a.held]], stats, 'all')[0] if a.stage == 'loro' else None
    schedule = balanced_schedule(groups(train), a.epochs, a.seed)
    x = torch.as_tensor(train['x'], device=device)
    y = torch.as_tensor(train['y'], device=device)
    rx = torch.as_tensor(train['rx'], device=device)
    vx = torch.as_tensor(val['x'], device=device)
    vy = torch.as_tensor(val['y'], device=device)
    vrx = torch.as_tensor(val['rx'], device=device)

    RS = receiver_stats(train['x'], train['y'], train['rx'])
    prof = torch.as_tensor(RS['prof'], dtype=torch.float32, device=device)
    prof_iso = torch.full_like(prof, float(prof.mean()))
    cmap = {int(r): torch.as_tensor(RS['c'][i], dtype=torch.float32, device=device) for i, r in enumerate(RS['rs'])}
    mvec = torch.as_tensor(RS['m'], dtype=torch.float32, device=device)
    U = torch.as_tensor(RS['U'], dtype=torch.float32, device=device)
    sig_a = torch.as_tensor(RS['sig_a'], dtype=torch.float32, device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(a.seed * 7919 + 17)

    def augment(xb, rb):
        if a.arm in ('ce', 'ls', 'ccalign') + CONS:
            return xb
        B = xb.shape[0]
        on = (torch.rand(B, generator=gen, device=device) < a.p_aug).float().view(B, 1, 1, 1)
        if a.arm in ('noise', 'noiseiso', 'ccanoise'):
            p = prof_iso if a.arm == 'noiseiso' else prof
            u = torch.rand(B, generator=gen, device=device).view(B, 1, 1, 1)
            eps = torch.randn(xb.shape, generator=gen, device=device)
            return xb + on * torch.sqrt(u * a.lam * p) * eps
        cr = torch.stack([cmap[int(r)] for r in rb.tolist()])
        g = torch.randn(B, U.shape[0], generator=gen, device=device)
        cv = (mvec + (a.lam * g * sig_a) @ U).view_as(cr)
        delta = cv - cr
        if a.arm == 'sigrand':
            r = torch.randn(delta.shape, generator=gen, device=device)
            n = delta.flatten(1).norm(dim=1) / r.flatten(1).norm(dim=1)
            delta = r * n.view(B, 1, 1, 1)
        return xb + on * delta

    def pre_head(xb):
        h = F.relu(model.stem(F.pad(xb, (2, 3, 2, 3))))
        h = model.blocks(h).permute(0, 2, 3, 1).reshape(xb.shape[0], -1)
        return F.relu(model.tx_input(F.normalize(model.feature(h), p=2, dim=1, eps=1e-12)))

    combo = a.cca_lam if a.cca_lam >= 0 else CCA_LAM[a.fold]
    cca_lam = a.lam if a.arm == 'ccalign' else combo if a.arm in ('ccanoise', 'ccancons') else 0.

    def noisy_view(xb):
        p = prof_iso if a.arm == 'nconsiso' else prof
        u = torch.rand(xb.shape[0], generator=gen, device=device).view(-1, 1, 1, 1)
        return xb + torch.sqrt(u * NCONS_SCALE * p) * torch.randn(xb.shape, generator=gen, device=device)

    def evaluate(X, Y, R):
        ms = []
        with torch.no_grad():
            for r in torch.unique(R):
                m = R == r
                ms.append(metrics(predict(model, X[m]).cpu().numpy(), Y[m].cpu().numpy()))
        return float(np.mean([q['CE'] for q in ms])), float(np.mean([q['accuracy'] for q in ms]))

    history = []
    best, best_state, best_epoch, best_acc = float('inf'), None, 0, 0.
    t0 = time.perf_counter()
    for epoch in range(a.epochs):
        for bi, ids in enumerate(schedule[epoch]):
            ix = torch.tensor(ids, device=device)
            xb, yb, rb = x[ix], y[ix], rx[ix]
            model.train()
            opt.zero_grad(set_to_none=True)
            pre = pre_head(augment(xb, rb))
            logits = model.classifier(pre)
            ce = F.cross_entropy(logits, yb, label_smoothing=0.1 if a.arm == 'ls' else 0.)
            aux = cca_lam * L.ccalign(pre, yb, rb) if cca_lam else ce * 0.
            if a.arm in CONS:
                lq = F.log_softmax(model.classifier(pre_head(noisy_view(xb))), 1)
                lp = F.log_softmax(logits, 1).detach()
                aux = aux + a.lam * (lp.exp() * (lp - lq)).sum(1).mean()
            loss = ce + aux
            assert torch.isfinite(loss)
            if bi == 0:
                diag = dict(epoch=epoch + 1, ce=float(ce), aux=float(aux))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.step()
        model.eval()
        vce, vacc = evaluate(vx, vy, vrx)
        if vce < best:
            best, best_epoch, best_acc = vce, epoch + 1, vacc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(dict(diag, val_CE=vce, val_acc=vacc))
        C.dump_json(out / 'history.json', history)
    model.load_state_dict(best_state)
    st = dict(state='completed', run_id=rid, seconds=time.perf_counter() - t0, best_epoch=best_epoch,
              best_val_CE=best, best_val_acc=best_acc, noise_profile_mean=float(prof.mean()),
              sig_dims=int(U.shape[0]))
    if held is not None:
        hce, hacc = evaluate(torch.as_tensor(held['x'], device=device), torch.as_tensor(held['y'], device=device),
                             torch.as_tensor(held['rx'], device=device))
        st.update(held_rx=src[a.held], held_CE=hce, held_acc=hacc)
    job = dict(vars(a), method=a.arm, set='Rq', run_id=rid, fit_keys=fit)
    torch.save(dict(model=model.state_dict(), job=job, scaler=stats, selection=dict(epoch=best_epoch)),
               out / 'selected.pt')
    st['checkpoint_sha256'] = C.sha(out / 'selected.pt')
    C.dump_json(out / 'status.json', st)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in st.items()
                      if k not in ('checkpoint_sha256',)}), flush=True)


if __name__ == '__main__':
    main()
