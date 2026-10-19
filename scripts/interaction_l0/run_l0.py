#!/usr/bin/env python3
"""L0-B runner (server): how do frozen R0/R1 (+R2-A h200) models use TX x RX structure, and which
physical cue do their unseen-RX errors follow?

What is optimised: NO network parameter is changed. The only optimisation is the source-only refit
of small fresh heads (same capacity as each model's head; Adam, 500 full-batch steps) on frozen
features, used as the head-mismatch control: 9 refits per model (no erasure, link/rx/cond at k=8,
random k=8 x 5 draws). This is therefore "frozen-backbone inference + small head refits", not a
zero-optimisation experiment.

Data partition (all source-only except evaluation):
  source_fit  source RX, dates 0/1, positions 0-120 (WiSig) / first 160 per TX (Shen): estimates
              erasure directions and refits heads
  source      source RX, dates 0/1, positions 160-200 (WiSig) / 160-200 (Shen): disjoint from
              source_fit; representation statistics, identity retention, input interventions
  internal_* / outer_RX  development pools, evaluation only (never used for directions or heads)
Data are read only through receiver_r01.data (asserts block WiSig final RX, date 3, equalized data;
Shen reads only the ten source-pool caches).

Two modes; the full run refuses to start unless a preflight with identical code said GO:
  python run_l0.py --root R01 [--r2-root R2A] --out PRE --preflight
  python run_l0.py --root R01 [--r2-root R2A] --out FULL/shard0 --full --preflight-dir PRE --shard 0/4
Full shards (one per GPU) are combined by merge_l0.py, which also computes ensembles.
"""
import argparse, hashlib, json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

# torch.use_deterministic_algorithms(True) needs this for cuBLAS on CUDA (as in every R0/R1 launcher)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from link_core import (complete_grid, cell_means, crossfit_energies, effect_directions, erase_oblique,
                       effects, random_basis, erase, loro_prototype_accuracy, rx_equal_accuracy)
from physics import (to_complex, to_cache, stf_descriptors, shift_cfo, reshape_tones, add_noise_to_snr, measured_change)

ERASE_K = (4, 8, 16)
KINDS = ('link', 'rx', 'cond')
PCA_Q = 24
RANDOM_DRAWS = 5
REFIT_K = 8
REFIT = dict(steps=500, lr=3e-3, l2=1e-4, seed=20261019)
SEED = 20261019
ATTRIBUTION_PERMUTATIONS = 200
MIN_PERIODICITY = 0.5
# engineering gates for GO_FULL_L0 (not scientific thresholds)
ENGINEERING = dict(split_gap_max=1e-4, cond_max=1e10, cfo_error_hz_max=500.0, tone_error_rms_max=0.2,
                   snr_error_db_max=1.5, low_periodicity_max=0.05, budget_cap_gpuh=2.0, budget_margin=1.5)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_shas():
    return {f.name: sha(f) for f in sorted(HERE.glob('*.py'))}


def split(model):
    """(encoder, head, fresh same-capacity head factory); model(x) == head(encoder(x)) is checked."""
    import torch.nn.functional as F
    from torch import nn
    name = type(model).__name__
    if name == 'OfficialCNN':
        return model.encode, (lambda f: model.head(model.drop(f))), (lambda: nn.Linear(80, 10))
    if name == 'WiSigResNet':
        return (lambda x: model.head[:3](model.body(x))), (lambda f: model.head[3:](f)), (lambda: nn.Linear(80, 10))
    if name == 'FeatureCNN':
        return ((lambda x: model.forward_with_features(x)[1]), (lambda f: model.classifier(F.relu(model.tx_input(f)))),
                (lambda: nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 10))))
    raise ValueError(name)


def run_batches(fn, x, device, batch=512):
    import torch
    with torch.no_grad():
        # numpy projections are float64; networks and heads are float32
        out = [fn(torch.as_tensor(np.ascontiguousarray(x[i:i + batch], dtype=np.float32), device=device)).float().cpu()
               for i in range(0, len(x), batch)]
    return torch.cat(out).numpy()


def refit_head(make_head, f, y, device):
    """Source-only refit of a fresh head on frozen features (full batch, fixed recipe)."""
    import torch
    from torch.nn import functional as F
    torch.manual_seed(REFIT['seed'])
    h = make_head().to(device); opt = torch.optim.Adam(h.parameters(), lr=REFIT['lr'])
    X = torch.as_tensor(np.asarray(f, dtype=np.float32), device=device); Y = torch.as_tensor(y, device=device)
    losses = []
    for step in range(REFIT['steps']):
        opt.zero_grad()
        loss = F.cross_entropy(h(X), Y) + REFIT['l2'] * sum((p ** 2).sum() for n, p in h.named_parameters() if 'weight' in n)
        loss.backward(); opt.step()
        if step in (0, REFIT['steps'] - 1): losses.append(float(loss.detach()))
    h.eval()
    return h, dict(refit_loss_first=losses[0], refit_loss_last=losses[-1])


def logit_stats(lg):
    e = np.exp(lg - lg.max(1, keepdims=True)); pr = e / e.sum(1, keepdims=True)
    return dict(mean_logit_norm=float(np.linalg.norm(lg, axis=1).mean()), mean_max_prob=float(pr.max(1).mean()))


def energies_by_date(f, data):
    """Cross-fitted TX/RX/link energies on the complete grid, averaged over dates."""
    tx, rx, day, pos = data['y'], data['rx'], data['day'], data['position']
    per = []
    for d in np.unique(day):
        m = day == d; txs, rxs = complete_grid(tx, rx, m)
        per.append(crossfit_energies(*(cell_means(f, tx, rx, m, txs, rxs, h, pos) for h in (0, 1))))
    return {k: float(np.mean([e[k] for e in per])) for k in ('tx', 'rx', 'link')}


def source_metrics(f, data):
    """Source-only representation statistics, per date then averaged; cells TX x RX."""
    tx, rx, day, pos = data['y'], data['rx'], data['day'], data['position']
    per = []
    for d in np.unique(day):
        m = day == d; txs, rxs = complete_grid(tx, rx, m)
        A, B = (cell_means(f, tx, rx, m, txs, rxs, h, pos) for h in (0, 1))
        e = crossfit_energies(A, B)
        # Fisher-type identity separability: TX energy over mean within-cell variance
        within = np.mean([f[m & (tx == t) & (rx == r)].var(0).sum() for t in txs for r in rxs])
        e.update(loro_proto=loro_prototype_accuracy(f, tx, rx, m, txs, rxs), separability=e['tx'] / within,
                 grid=f'{len(txs)}x{len(rxs)}')
        per.append(e)
    out = {k: float(np.mean([e[k] for e in per])) for k in ('tx', 'rx', 'link', 'loro_proto', 'separability')}
    total = out['tx'] + out['rx'] + out['link']
    out.update(link_share=out['link'] / total, rx_share=out['rx'] / total, link_ratio=out['link'] / out['tx'],
               grids=';'.join(e['grid'] for e in per))
    return out


def domain_key(data):
    return data['rx'] * 10 + (data['day'] + 1)


def wisig_physics(pools):
    """Model-free descriptors, calibrated label-preserving interventions on the source pool, and a
    requested-vs-measured validation of every intervention on the real packets."""
    rng = np.random.default_rng(SEED)
    src = pools['source']; z = to_complex(src['x']); d = stf_descriptors(z)
    tx, rx, day = src['y'], src['rx'], src['day']
    eff = {}
    for dd in np.unique(day):
        m = day == dd; txs, rxs = complete_grid(tx, rx, m, min_count=1)
        for name, v in (('cfo', d['cfo_hz'][:, None]), ('tone', d['tone_shape']), ('snr', d['snr_db'][:, None])):
            g, a, b, i = effects(cell_means(v, tx, rx, m, txs, rxs))
            eff[(name, dd)] = dict(g=g, a=a, b=b, i=i, txs=list(txs), rxs=list(rxs))
    sigma_b = float(np.mean([eff[('cfo', dd)]['b'][:, 0].std(ddof=1) for dd in np.unique(day)]))
    n = len(tx)
    other_rx = np.empty(n, int); other_tx = np.empty(n, int); look = []
    for k in range(n):
        e = eff[('tone', day[k])]
        rr = [r for r in e['rxs'] if r != rx[k]]; tt = [t for t in e['txs'] if t != tx[k]]
        other_rx[k] = rr[rng.integers(len(rr))]; other_tx[k] = tt[rng.integers(len(tt))]
        look.append((e['txs'].index(tx[k]), e['rxs'].index(rx[k]), e['rxs'].index(other_rx[k]), e['txs'].index(other_tx[k])))
    look = np.array(look)
    def pick(name, part, rows, cols=None):
        out = np.zeros((n, eff[(name, 0)][part].shape[-1]))
        for dd in np.unique(day):
            m = day == dd; e = eff[(name, dd)][part]
            out[m] = e[rows[m]] if cols is None else e[rows[m], cols[m]]
        return out
    t_i, r_i, r2_i, t2_i = look.T
    cfo_tx = pick('cfo', 'a', t2_i)[:, 0] - pick('cfo', 'a', t_i)[:, 0]
    tone_req = {'tone_link_swap': pick('tone', 'i', t_i, r2_i) - pick('tone', 'i', t_i, r_i),
                'tone_rx_swap': pick('tone', 'b', r2_i) - pick('tone', 'b', r_i),
                'tone_tx_swap': pick('tone', 'a', t2_i) - pick('tone', 'a', t_i)}
    cfo_req = {'cfo_rx_+1sd': np.full(n, sigma_b), 'cfo_rx_-1sd': np.full(n, -sigma_b),
               'cfo_rx_+2sd': np.full(n, 2 * sigma_b), 'cfo_rx_-2sd': np.full(n, -2 * sigma_b), 'cfo_tx_swap': cfo_tx}
    # SNR of the same TX at another source RX (cell mean = grand + TX + RX + link), per packet
    snr_target = np.array([eff[('snr', day[k])]['g'][0] + eff[('snr', day[k])]['a'][t_i[k], 0] + eff[('snr', day[k])]['b'][r2_i[k], 0]
                           + eff[('snr', day[k])]['i'][t_i[k], r2_i[k], 0] for k in range(n)])
    complex_iv = {k: shift_cfo(z, v) for k, v in cfo_req.items()}
    complex_iv.update({k: reshape_tones(z, v) for k, v in tone_req.items()})
    complex_iv['snr_link_swap'] = add_noise_to_snr(z, d['snr_db'], snr_target, rng)
    targets = {'cfo_tx_swap': other_tx, 'tone_tx_swap': other_tx}
    validation = []
    for name, z1 in complex_iv.items():
        mc = measured_change(z, z1); ok = mc['finite']
        row = dict(intervention=name, packets=n, nonfinite=int((~ok).sum()),
                   low_periodicity_after=float((mc['periodicity_after'] < MIN_PERIODICITY).mean()),
                   input_rel_change_mean=float(mc['input_rel_change'][ok].mean()),
                   measured_cfo_change_hz_median=float(np.median(mc['d_cfo_hz'][ok])),
                   measured_tone_change_rms=float(np.sqrt(np.mean(mc['d_tone'][ok] ** 2))),
                   measured_snr_change_db_median=float(np.median(mc['d_snr_db'][ok])))
        if name in cfo_req:
            err = mc['d_cfo_hz'][ok] - cfo_req[name][ok]
            row.update(requested='cfo_hz', requested_median=float(np.median(cfo_req[name])), error_median_abs=float(np.median(np.abs(err))),
                       tone_leakage_rms=row['measured_tone_change_rms'])  # tone change caused by a CFO-only intervention
        elif name in tone_req:
            req = tone_req[name] - tone_req[name].mean(1, keepdims=True)
            row.update(requested='tone_log_power', requested_rms=float(np.sqrt(np.mean(req ** 2))),
                       error_rms=float(np.sqrt(np.mean((mc['d_tone'][ok] - req[ok]) ** 2))),
                       cfo_side_effect_hz_median_abs=float(np.median(np.abs(mc['d_cfo_hz'][ok]))))
        else:
            req = np.minimum(snr_target, d['snr_db']) - d['snr_db']; changed = ok & (req < -0.5)
            row.update(requested='snr_db', changed_fraction=float(changed.mean()),
                       error_median_abs=float(np.median(np.abs(mc['d_snr_db'][changed] - req[changed]))) if changed.any() else np.nan,
                       unreachable_fraction=float((snr_target >= d['snr_db']).mean()))
        validation.append(row)
    x = {k: (to_cache(v), targets.get(k)) for k, v in complex_iv.items()}
    ref = {}
    for dd in np.unique(day):  # source TX means over source RX (date 2 uses date-1 statistics)
        c, t_ = eff[('cfo', dd)], eff[('tone', dd)]
        ref[dd] = dict(txs=np.array(c['txs']), cfo=c['g'][0] + c['a'][:, 0] + c['b'][:, 0].mean(),
                       tone=t_['g'] + t_['a'] + t_['b'].mean(0))
    return dict(sigma_b_hz=sigma_b, interventions=x, ref=ref, validation=validation,
                source_low_periodicity=float((d['periodicity'] < MIN_PERIODICITY).mean()))


def attribution(pred, data, ref, rng):
    """Do CNN errors follow a pure-CFO or pure-tone-shape nearest-TX rule built from source stats?"""
    d = stf_descriptors(to_complex(data['x'])); ok = d['periodicity'] >= MIN_PERIODICITY
    rows = []
    for cue in ('cfo', 'tone'):
        guess = np.empty(len(pred), int)
        for dd in np.unique(data['day']):
            m = data['day'] == dd; R = ref[min(dd, 1)]
            dist = (np.abs(d['cfo_hz'][m, None] - R['cfo'][None]) if cue == 'cfo'
                    else ((d['tone_shape'][m, None, :] - R['tone'][None]) ** 2).sum(-1))
            guess[m] = R['txs'][dist.argmin(1)]
        wrong = (pred != data['y']) & ok; right = (pred == data['y']) & ok
        null = []
        for _ in range(ATTRIBUTION_PERMUTATIONS):
            g2 = guess.copy()
            for r in np.unique(data['rx']):
                idx = np.flatnonzero(wrong & (data['rx'] == r)); g2[idx] = guess[rng.permutation(idx)]
            null.append((pred[wrong] == g2[wrong]).mean() if wrong.any() else np.nan)
        rows.append(dict(cue=cue, cue_rule_accuracy=float((guess[ok] == data['y'][ok]).mean()),
                         agree_given_cnn_wrong=float((pred[wrong] == guess[wrong]).mean()) if wrong.any() else np.nan,
                         agree_given_cnn_wrong_null=float(np.nanmean(null)),
                         agree_given_cnn_right=float((pred[right] == guess[right]).mean()) if right.any() else np.nan,
                         cnn_errors=int(wrong.sum()), excluded_low_periodicity=int((~ok).sum())))
    return rows


def select_preflight(jobs):
    """One job per architecture group (WiSig official, WiSig ResNet, Shen CNN, R2-A if given)."""
    seen, out = set(), []
    for j in jobs:
        key = (j['dataset'], j['backbone'], j['group'].split('/')[-1])
        if key not in seen:
            seen.add(key); out.append(j)
    return out


def _git_head(path):
    import subprocess
    r = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return r.stdout.strip() or None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='R0/R1 worktree with outputs/receiver_r01')
    p.add_argument('--r2-root', help='optional R2-A worktree with outputs/receiver_r2 (h200 runs)')
    p.add_argument('--out', required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true'); mode.add_argument('--full', action='store_true')
    p.add_argument('--preflight-dir', help='required with --full: a preflight output that decided GO_FULL_L0')
    p.add_argument('--shard', default='0/1', help='full mode: i/n runs jobs i, i+n, ... (one shard per GPU)')
    a = p.parse_args()
    shard_i, shard_n = map(int, a.shard.split('/')); assert 0 <= shard_i < shard_n
    if a.full:
        assert a.preflight_dir, '--full needs --preflight-dir'
        pre = json.loads((Path(a.preflight_dir) / 'preflight.json').read_text())
        assert pre['decision'] == 'GO_FULL_L0', f"preflight decided {pre['decision']}: {pre.get('reasons')}"
        assert pre['code_sha256'] == code_shas(), 'code changed since preflight; rerun preflight'
    root = Path(a.root); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(root / 'src'), str(root / 'scripts')]
    import torch
    import pandas as pd
    from receiver_r01.data import wisig, shen
    from train_receiver_r01 import build

    reg = json.loads((root / 'configs/receiver_r01/registration.json').read_text())
    all_jobs = [dict(j, group=f"{j['dataset']}/{j['backbone']}/r01", ckpt=root / 'outputs/receiver_r01/formal' / j['run_id'])
                for j in reg['formal_jobs']]
    if a.r2_root:
        r2 = json.loads((Path(a.r2_root) / 'configs/receiver_r2/registration.json').read_text())
        all_jobs += [dict(j, group=f"{j['dataset']}/{j['backbone']}/r2a_h200", ckpt=Path(a.r2_root) / 'outputs/receiver_r2' / j['run_id'])
                     for j in r2['jobs'] if j['stage'] == 'horizon']
    jobs = select_preflight(all_jobs) if a.preflight else all_jobs[shard_i::shard_n]
    refits_per_model = 1 + len(KINDS) + RANDOM_DRAWS
    (out / 'manifest.json').write_text(json.dumps(dict(
        mode='preflight' if a.preflight else 'full', shard=a.shard, all_jobs=len(all_jobs), code_sha256=code_shas(), erase_k=ERASE_K, kinds=KINDS, pca_q=PCA_Q,
        random_draws=RANDOM_DRAWS, refit=REFIT, refit_k=REFIT_K, head_refits_per_model=refits_per_model, seed=SEED,
        jobs=[j['run_id'] for j in jobs], root=str(root), r2_root=a.r2_root, root_git_head=_git_head(root),
        network_parameters_changed=False, head_refits=True, final_RX_read=False, date3_read=False,
        equalized_read=False, started_unix=time.time()), indent=2, default=str))

    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats()
    cache, load_seconds = {}, defaultdict(float)
    sync = (lambda: torch.cuda.synchronize()) if device.type == 'cuda' else (lambda: None)

    def pools(job, scaler):
        t0 = time.perf_counter()
        if job['dataset'] == 'wisig':
            if 'wisig' not in cache:
                src, internal = reg['wisig']['roles']['source'], reg['wisig']['roles']['internal_rx']
                cache['wisig'] = dict(source=wisig(reg, src, [0, 1], [160, 200]),
                                      internal_seen_date=wisig(reg, internal, [0, 1], [0, 200]),
                                      internal_unseen_date=wisig(reg, internal, [2], [0, 200]),
                                      source_fit=wisig(reg, src, [0, 1], [0, 120]))
                cache['wisig_physics'] = wisig_physics(cache['wisig'])
                load_seconds['wisig'] += time.perf_counter() - t0
            return cache['wisig']
        fold = reg['shen_folds'][job['fold']]; keys = fold['inner_train'] + fold['inner_validation']
        key = ('shen', job['fold'], scaler['mean'], scaler['std'])  # scaler is a fold property
        if key not in cache:
            cache[key] = dict(source=shen(reg, keys, scaler, 'val')[0], outer_RX=shen(reg, fold['outer_evaluation'], scaler, 'all')[0],
                              source_fit=shen(reg, keys, scaler, 'train')[0])
            load_seconds[f"shen{job['fold']}"] += time.perf_counter() - t0
        return cache[key]

    runs, erasures, sens, attrib, checks = [], [], [], [], []
    eval_roles = lambda data: [r for r in data if r != 'source_fit']
    tick = time.perf_counter()
    for job in jobs:
        timer = defaultdict(float); t0 = time.perf_counter()
        status = json.loads((job['ckpt'] / 'status.json').read_text())
        assert status['state'] == 'completed' and sha(job['ckpt'] / 'selected.pt') == status['checkpoint_sha256'], job['run_id']
        cp = torch.load(job['ckpt'] / 'selected.pt', map_location='cpu', weights_only=False)
        model = build(job).to(device); model.load_state_dict(cp['model']); model.eval()
        enc, head, make_head = split(model)
        timer['checkpoint_io'] += time.perf_counter() - t0
        data = pools(job, cp.get('scaler'))
        t0 = time.perf_counter()
        x0 = torch.as_tensor(data['source']['x'][:256], device=device)
        with torch.no_grad():
            gap = float((model(x0) - head(enc(x0))).abs().max())
        assert gap < ENGINEERING['split_gap_max'], (job['run_id'], gap)
        feats = {r: run_batches(enc, d['x'], device) for r, d in data.items()}
        logits = lambda f, h=head: run_batches(h, f, device, 4096)
        sync(); timer['encode'] += time.perf_counter() - t0
        finite = dict(features=bool(all(np.isfinite(v).all() for v in feats.values())))
        src, fit = data['source'], data['source_fit']
        ident = dict(run_id=job['run_id'], group=job['group'], dataset=job['dataset'], backbone=job['backbone'],
                     fold=job['fold'], method=job['method'], seed=job['seed'])
        t0 = time.perf_counter()
        row = dict(**ident, feature_dim=feats['source'].shape[1], split_gap=gap,
                   **{f'feat_{k}': v for k, v in source_metrics(feats['source'], src).items()})
        row.update({f'logit_{k}': v for k, v in source_metrics(logits(feats['source']), src).items() if k in ('link_share', 'link_ratio')})
        preds = {}
        for role in eval_roles(data):
            lg = logits(feats[role]); d = data[role]; preds[role] = lg.argmax(1)
            row[f'acc_{role}'] = rx_equal_accuracy(preds[role], d['y'], d['rx'], d['day'])
            row[f'classes_{role}'] = int(len(np.unique(d['y'])))
            if role != 'source' and a.full:  # saved for ensembles / error overlap in merge_l0.py
                e = np.exp(lg - lg.max(1, keepdims=True)); (out / 'probs').mkdir(exist_ok=True)
                np.savez(out / 'probs' / f"{job['run_id']}__{role}.npz", prob=(e / e.sum(1, keepdims=True)).astype(np.float32),
                         y=d['y'], rx=d['rx'], day=d['day'], position=d['position'])
        timer['representation_stats'] += time.perf_counter() - t0

        # erasure: directions and refit heads from source_fit only; identity retention on the disjoint source pool
        dk = domain_key(fit); m = np.ones(len(dk), bool); txs, dks = complete_grid(fit['y'], dk, m)
        c = feats['source_fit'].mean(0); dim = feats['source'].shape[1]
        base_energy = energies_by_date(feats['source'], src)
        spread = float(np.mean(np.sum((feats['source'] - feats['source'].mean(0)) ** 2, 1)))
        conds = []
        def score(transform, kind, k, draw=None, extra=None):
            rec = dict(**ident, kind=kind, k=k, draw=draw, **(extra or {}))
            fs = transform(feats['source'])
            rec['feature_change_DT'] = float(np.mean(np.sum((fs - feats['source']) ** 2, 1)) / spread)
            en = energies_by_date(fs, src)
            rec.update(tx_energy_retained=en['tx'] / base_energy['tx'], link_energy_retained=en['link'] / base_energy['link'],
                       rx_energy_retained=en['rx'] / base_energy['rx'] if base_energy['rx'] > 0 else np.nan)
            for head_name in ('frozen', 'refit'):
                if head_name == 'refit' and k != REFIT_K and kind != 'none':
                    continue
                t1 = time.perf_counter()
                if head_name == 'frozen':
                    h = head
                else:
                    h, info = refit_head(make_head, transform(feats['source_fit']), fit['y'], device); rec.update(info)
                    sync(); timer['refit'] += time.perf_counter() - t1
                for role in eval_roles(data):
                    lg = logits(transform(feats[role]), h); dd = data[role]
                    finite['logits'] = finite.get('logits', True) and bool(np.isfinite(lg).all())
                    rec[f'{head_name}_acc_{role}'] = rx_equal_accuracy(lg.argmax(1), dd['y'], dd['rx'], dd['day'])
                    if role in ('internal_unseen_date', 'outer_RX'):
                        rec.update({f'{head_name}_{s}': v for s, v in logit_stats(lg).items()})
            return rec
        t0 = time.perf_counter()
        erasures.append(score(lambda f: f, 'none', 0))
        for kind in KINDS:
            for k in ERASE_K:
                if k > min(PCA_Q, dim): continue
                info = {}
                V, G, share = effect_directions(feats['source_fit'], fit['y'], dk, m, txs, dks, k, kind, PCA_Q, info=info)
                conds.append(info['cond_Bq'])
                finite['projection'] = finite.get('projection', True) and bool(np.isfinite(V).all() and np.isfinite(G).all())
                erasures.append(score(lambda f, V=V, G=G: erase_oblique(f, V, G, c), kind, k,
                                      extra=dict(mean_share=float(share.mean()), cond_Bq=info['cond_Bq'])))
        for k in ERASE_K:
            if k > min(PCA_Q, dim): continue
            for draw in range(RANDOM_DRAWS):
                Q = random_basis(dim, k, 1000 * k + draw)
                erasures.append(score(lambda f, Q=Q: erase(f, Q, c), 'random', k, draw))
        sync(); timer['erasure_total_incl_refit'] += time.perf_counter() - t0

        if job['dataset'] == 'wisig':
            t0 = time.perf_counter()
            phys = cache['wisig_physics']; base = preds['source']
            for name, (xp, target) in phys['interventions'].items():
                pr = logits(run_batches(enc, xp, device)).argmax(1)
                acc = rx_equal_accuracy(pr, src['y'], src['rx'], src['day'])
                sens.append(dict(**ident, intervention=name, sigma_b_hz=phys['sigma_b_hz'], acc=acc,
                                 acc_drop_pp=100 * (row['acc_source'] - acc), flip_rate=float((pr != base).mean()),
                                 moved_to_target=float((pr == target).mean()) if target is not None else np.nan))
            sync(); timer['interventions'] += time.perf_counter() - t0
            t0 = time.perf_counter(); rng = np.random.default_rng(SEED)
            for role in ('internal_seen_date', 'internal_unseen_date'):
                for r_ in attribution(preds[role], data[role], phys['ref'], rng):
                    attrib.append(dict(**ident, role=role, **r_))
            timer['attribution'] += time.perf_counter() - t0

        refit_rows = [e for e in erasures if e['run_id'] == job['run_id'] and 'refit_loss_last' in e]
        checks.append(dict(**ident, split_gap=gap, finite_all=bool(all(finite.values())), **{f'finite_{k}': v for k, v in finite.items()},
                           cond_Bq_max=float(max(conds)) if conds else np.nan, refits=len(refit_rows),
                           refit_loss_decreased=bool(all(np.isfinite(e['refit_loss_last']) and e['refit_loss_last'] < e['refit_loss_first'] for e in refit_rows)),
                           **{f'seconds_{k}': v for k, v in timer.items()}, seconds_job=float(sum(timer.values()))))
        runs.append(row); del model
        for name, rows in (('run_metrics', runs), ('erasure', erasures), ('sensitivity', sens), ('attribution', attrib), ('checks', checks)):
            pd.DataFrame(rows).to_csv(out / f'{name}.csv', index=False)
        print(json.dumps({k: row[k] for k in ('run_id', 'feat_link_share', 'feat_link_ratio', 'acc_source')}),
              f"job {checks[-1]['seconds_job']:.0f}s total {time.perf_counter() - tick:.0f}s", flush=True)
    if 'wisig_physics' in cache:
        pd.DataFrame(cache['wisig_physics']['validation']).to_csv(out / 'intervention_validation.csv', index=False)
    peak = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == 'cuda' else 0.0
    timing = dict(seconds=time.perf_counter() - tick, jobs=len(jobs), device=str(device), peak_allocated_MiB=peak,
                  data_load_seconds=dict(load_seconds))
    (out / 'timing.json').write_text(json.dumps(timing, indent=2))

    if a.preflight:
        C = pd.DataFrame(checks); V = pd.DataFrame(cache['wisig_physics']['validation']) if 'wisig_physics' in cache else pd.DataFrame()
        reasons = []
        if not (C.split_gap < ENGINEERING['split_gap_max']).all(): reasons.append('split gap')
        if not C.finite_all.all(): reasons.append('non-finite features/logits/projections')
        if not (C.cond_Bq_max < ENGINEERING['cond_max']).all(): reasons.append('ill-conditioned between-cell scatter')
        if not (C.refit_loss_decreased & (C.refits == refits_per_model)).all(): reasons.append('head refit did not complete/decrease')
        if not V.empty:
            cf = V[V.requested == 'cfo_hz']; tn = V[V.requested == 'tone_log_power']; sn = V[V.requested == 'snr_db']
            if (cf.error_median_abs > ENGINEERING['cfo_error_hz_max']).any(): reasons.append('CFO intervention not realised')
            if (tn.error_rms > ENGINEERING['tone_error_rms_max']).any(): reasons.append('tone intervention not realised')
            if (sn.error_median_abs > ENGINEERING['snr_error_db_max']).any(): reasons.append('SNR intervention not realised')
            if (V.nonfinite > 0).any(): reasons.append('non-finite intervened packets')
            if cache['wisig_physics']['source_low_periodicity'] > ENGINEERING['low_periodicity_max']: reasons.append('too many packets outside L-STF')
        count = defaultdict(int)
        for j in all_jobs: count[(j['dataset'], j['backbone'], j['group'].split('/')[-1])] += 1
        per_arch = {}
        for _, r in C.iterrows():
            key = (r.dataset, r.backbone, r.group.split('/')[-1])
            per_arch['/'.join(key)] = dict(seconds_per_job=float(r.seconds_job), jobs_in_full=count[key])
        est = (sum(v['seconds_per_job'] * v['jobs_in_full'] for v in per_arch.values()) + sum(load_seconds.values())) * ENGINEERING['budget_margin'] / 3600
        if est > ENGINEERING['budget_cap_gpuh']: reasons.append(f'budget estimate {est:.3f} GPUh above cap')
        decision = 'GO_FULL_L0' if not reasons else 'HOLD_ENGINEERING'
        (out / 'preflight.json').write_text(json.dumps(dict(decision=decision, reasons=reasons, engineering_gates=ENGINEERING,
            code_sha256=code_shas(), per_architecture=per_arch, full_jobs=len(all_jobs), head_refits_full=len(all_jobs) * refits_per_model,
            full_estimate_reserved_gpuh_with_margin=est, timing=timing, note='engineering gate only; human review before --full'),
            indent=2, default=float))
        print(json.dumps(dict(decision=decision, reasons=reasons, full_estimate_gpuh=round(est, 4), peak_MiB=round(peak)), indent=2))
        return

    print('done', f"{timing['seconds']:.0f}s", f'peak {peak:.0f} MiB')

if __name__ == '__main__':
    main()
