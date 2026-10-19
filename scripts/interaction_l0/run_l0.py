#!/usr/bin/env python3
"""L0-B runner (server, inference only): how do frozen R0/R1 (+R2-A h200) models use TX x RX
structure, and which physical cue do their unseen-RX errors follow?

Measurements only; decisions are made afterwards by summarize_l0.py with frozen rules.
No fitting of the network, no optimizer on it, no new checkpoints. The only trained objects are
small source-only refit heads (same capacity as the original head) on frozen features, used as
the head-mismatch control for erasure. Data are read only through receiver_r01.data (asserts
block WiSig final RX, date 3, equalized data; Shen reads only the ten source-pool caches).
Internal WiSig RX / Shen outer RX are already-viewed development pools: exploratory evidence.

    python run_l0.py --root /home/keyan/code-y-r0r1 [--r2-root /home/keyan/code-y-r2a] --out OUT [--limit 2]

Outputs: manifest.json run_metrics.csv erasure.csv sensitivity.csv attribution.csv ensemble.csv
error_overlap.csv
"""
import argparse, hashlib, itertools, json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from link_core import (complete_grid, cell_means, crossfit_energies, effect_directions, erase_oblique, effects,
                       random_basis, erase, loro_prototype_accuracy, rx_equal_accuracy, error_overlap)
from physics import (to_complex, to_cache, stf_descriptors, shift_cfo, reshape_tones, add_noise_to_snr)

ERASE_K = (4, 8, 16)
KINDS = ('link', 'rx', 'cond')
PCA_Q = 24
RANDOM_DRAWS = 5
REFIT_K = 8
REFIT = dict(steps=500, lr=3e-3, l2=1e-4, seed=20261019)
SEED = 20261019
ATTRIBUTION_PERMUTATIONS = 200
MIN_PERIODICITY = 0.5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
        out = [fn(torch.as_tensor(np.ascontiguousarray(x[i:i + batch]), device=device)).float().cpu()
               for i in range(0, len(x), batch)]
    return torch.cat(out).numpy()


def refit_head(make_head, f, y, device):
    """Source-only refit of a fresh head on frozen features (full batch, fixed recipe)."""
    import torch
    from torch.nn import functional as F
    torch.manual_seed(REFIT['seed'])
    h = make_head().to(device); opt = torch.optim.Adam(h.parameters(), lr=REFIT['lr'])
    X = torch.as_tensor(f, device=device); Y = torch.as_tensor(y, device=device)
    for _ in range(REFIT['steps']):
        opt.zero_grad()
        loss = F.cross_entropy(h(X), Y) + REFIT['l2'] * sum((p ** 2).sum() for n, p in h.named_parameters() if 'weight' in n)
        loss.backward(); opt.step()
    h.eval()
    return h


def logit_stats(lg):
    e = np.exp(lg - lg.max(1, keepdims=True)); pr = e / e.sum(1, keepdims=True)
    return dict(mean_logit_norm=float(np.linalg.norm(lg, axis=1).mean()), mean_max_prob=float(pr.max(1).mean()))


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
    """Model-free descriptors and calibrated, label-preserving interventions on the source pool."""
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
    interventions = {
        'cfo_rx_+1sd': (shift_cfo(z, sigma_b), None), 'cfo_rx_-1sd': (shift_cfo(z, -sigma_b), None),
        'cfo_rx_+2sd': (shift_cfo(z, 2 * sigma_b), None), 'cfo_rx_-2sd': (shift_cfo(z, -2 * sigma_b), None),
        'cfo_tx_swap': (shift_cfo(z, cfo_tx), other_tx),
        'tone_link_swap': (reshape_tones(z, pick('tone', 'i', t_i, r2_i) - pick('tone', 'i', t_i, r_i)), None),
        'tone_rx_swap': (reshape_tones(z, pick('tone', 'b', r2_i) - pick('tone', 'b', r_i)), None),
        'tone_tx_swap': (reshape_tones(z, pick('tone', 'a', t2_i) - pick('tone', 'a', t_i)), other_tx),
    }
    # SNR of the same TX at another source RX (cell mean = grand + TX + RX + link), per packet
    target = np.array([eff[('snr', day[k])]['g'][0] + eff[('snr', day[k])]['a'][t_i[k], 0] + eff[('snr', day[k])]['b'][r2_i[k], 0]
                       + eff[('snr', day[k])]['i'][t_i[k], r2_i[k], 0] for k in range(n)])
    interventions['snr_link_swap'] = (add_noise_to_snr(z, d['snr_db'], target, rng), None)
    x = {k: (to_cache(v), tgt) for k, (v, tgt) in interventions.items()}
    # source references for attribution: TX means over source RX (date 2 uses date 1 statistics)
    ref = {}
    for dd in np.unique(day):
        c, t_ = eff[('cfo', dd)], eff[('tone', dd)]
        ref[dd] = dict(txs=np.array(c['txs']), cfo=c['g'][0] + c['a'][:, 0] + c['b'][:, 0].mean(),
                       tone=t_['g'] + t_['a'] + t_['b'].mean(0))
    return dict(sigma_b_hz=sigma_b, interventions=x, ref=ref,
                snr_changed_fraction=float((target < d['snr_db']).mean()))


def attribution(pred, data, ref, rng):
    """Do CNN errors follow a pure-CFO or pure-tone-shape nearest-TX rule built from source stats?"""
    d = stf_descriptors(to_complex(data['x'])); ok = d['periodicity'] >= MIN_PERIODICITY
    rows = []
    for cue in ('cfo', 'tone'):
        guess = np.empty(len(pred), int)
        for dd in np.unique(data['day']):
            m = data['day'] == dd; R = ref[min(dd, 1)]
            if cue == 'cfo':
                dist = np.abs(d['cfo_hz'][m, None] - R['cfo'][None])
            else:
                dist = ((d['tone_shape'][m, None, :] - R['tone'][None]) ** 2).sum(-1)
            guess[m] = R['txs'][dist.argmin(1)]
        wrong = (pred != data['y']) & ok; right = (pred == data['y']) & ok
        agree_wrong = float((pred[wrong] == guess[wrong]).mean()) if wrong.any() else np.nan
        null = []
        for _ in range(ATTRIBUTION_PERMUTATIONS):
            g2 = guess.copy()
            for r in np.unique(data['rx']):
                idx = np.flatnonzero(wrong & (data['rx'] == r)); g2[idx] = guess[rng.permutation(idx)]
            null.append((pred[wrong] == g2[wrong]).mean() if wrong.any() else np.nan)
        rows.append(dict(cue=cue, cue_rule_accuracy=float((guess[ok] == data['y'][ok]).mean()),
                         agree_given_cnn_wrong=agree_wrong, agree_given_cnn_wrong_null=float(np.nanmean(null)),
                         agree_given_cnn_right=float((pred[right] == guess[right]).mean()) if right.any() else np.nan,
                         cnn_errors=int(wrong.sum()), excluded_low_periodicity=int((~ok).sum())))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='R0/R1 worktree with outputs/receiver_r01')
    p.add_argument('--r2-root', help='optional R2-A worktree with outputs/receiver_r2 (h200 runs)')
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, help='smoke test: first N jobs only')
    a = p.parse_args()
    root = Path(a.root); out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(root / 'src'), str(root / 'scripts')]
    import torch
    import pandas as pd
    from receiver_r01.data import wisig, shen
    from train_receiver_r01 import build

    reg = json.loads((root / 'configs/receiver_r01/registration.json').read_text())
    jobs = [dict(j, group=f"{j['dataset']}/{j['backbone']}/r01", ckpt=root / 'outputs/receiver_r01/formal' / j['run_id'])
            for j in reg['formal_jobs']]
    if a.r2_root:
        r2 = json.loads((Path(a.r2_root) / 'configs/receiver_r2/registration.json').read_text())
        jobs += [dict(j, group=f"{j['dataset']}/{j['backbone']}/r2a_h200", ckpt=Path(a.r2_root) / 'outputs/receiver_r2' / j['run_id'])
                 for j in r2['jobs'] if j['stage'] == 'horizon']
    jobs = jobs[:a.limit] if a.limit else jobs
    (out / 'manifest.json').write_text(json.dumps(dict(
        files={f.name: sha(f) for f in sorted(HERE.glob('*.py'))}, erase_k=ERASE_K, kinds=KINDS, pca_q=PCA_Q,
        random_draws=RANDOM_DRAWS, refit=REFIT, refit_k=REFIT_K, seed=SEED, jobs=[j['run_id'] for j in jobs],
        root=str(root), r2_root=a.r2_root, network_fitting=False, final_RX_read=False, date3_read=False,
        equalized_read=False, started_unix=time.time()), indent=2))

    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cache = {}

    def pools(job, scaler):
        if job['dataset'] == 'wisig':
            if 'wisig' not in cache:
                src, internal = reg['wisig']['roles']['source'], reg['wisig']['roles']['internal_rx']
                cache['wisig'] = dict(source=wisig(reg, src, [0, 1], [160, 200]),
                                      internal_seen_date=wisig(reg, internal, [0, 1], [0, 200]),
                                      internal_unseen_date=wisig(reg, internal, [2], [0, 200]),
                                      source_fit=wisig(reg, src, [0, 1], [0, 120]))
                cache['wisig_physics'] = wisig_physics(cache['wisig'])
            return cache['wisig']
        fold = reg['shen_folds'][job['fold']]; keys = fold['inner_train'] + fold['inner_validation']
        key = ('shen', job['fold'], scaler['mean'], scaler['std'])  # scaler is a fold property
        if key not in cache:
            cache[key] = dict(source=shen(reg, keys, scaler, 'val')[0], outer_RX=shen(reg, fold['outer_evaluation'], scaler, 'all')[0],
                              source_fit=shen(reg, keys, scaler, 'train')[0])
        return cache[key]

    runs, erasures, sens, attrib, probs, run_pool = [], [], [], [], {}, {}
    eval_roles = lambda data: [r for r in data if r != 'source_fit']
    tick = time.perf_counter()
    for job in jobs:
        status = json.loads((job['ckpt'] / 'status.json').read_text())
        assert status['state'] == 'completed' and sha(job['ckpt'] / 'selected.pt') == status['checkpoint_sha256'], job['run_id']
        cp = torch.load(job['ckpt'] / 'selected.pt', map_location='cpu', weights_only=False)
        model = build(job).to(device); model.load_state_dict(cp['model']); model.eval()
        enc, head, make_head = split(model)
        data = pools(job, cp.get('scaler')); run_pool[job['run_id']] = data
        x0 = torch.as_tensor(data['source']['x'][:256], device=device)
        with torch.no_grad():
            gap = float((model(x0) - head(enc(x0))).abs().max())
        assert gap < 1e-4, (job['run_id'], gap)
        feats = {r: run_batches(enc, d['x'], device) for r, d in data.items()}
        logits = lambda f, h=head: run_batches(h, f, device, 4096)
        src = data['source']; ident = dict(run_id=job['run_id'], group=job['group'], dataset=job['dataset'],
                                           backbone=job['backbone'], fold=job['fold'], method=job['method'], seed=job['seed'])
        row = dict(**ident, feature_dim=feats['source'].shape[1], split_gap=gap,
                   **{f'feat_{k}': v for k, v in source_metrics(feats['source'], src).items()})
        row.update({f'logit_{k}': v for k, v in source_metrics(logits(feats['source']), src).items() if k in ('link_share', 'link_ratio')})
        preds = {}
        for role in eval_roles(data):
            lg = logits(feats[role]); d = data[role]; preds[role] = lg.argmax(1)
            row[f'acc_{role}'] = rx_equal_accuracy(preds[role], d['y'], d['rx'], d['day'])
            row[f'classes_{role}'] = int(len(np.unique(d['y'])))
            if role != 'source':
                e = np.exp(lg - lg.max(1, keepdims=True)); probs[(job['run_id'], role)] = (e / e.sum(1, keepdims=True)).astype(np.float32)

        # erasure: directions from the source pool only; cells TX x (RX,date); frozen head and refit head
        dk = domain_key(src); m = np.ones(len(dk), bool); txs, dks = complete_grid(src['y'], dk, m)
        c = feats['source'].mean(0); dim = feats['source'].shape[1]; fit = data['source_fit']
        def score(transform, kind, k, draw=None):
            rec = dict(**ident, kind=kind, k=k, draw=draw)
            for head_name in ('frozen', 'refit'):
                if head_name == 'refit' and k != REFIT_K and kind != 'none':
                    continue
                h = head if head_name == 'frozen' else refit_head(make_head, transform(feats['source_fit']), fit['y'], device)
                for role in eval_roles(data):
                    lg = logits(transform(feats[role]), h); dd = data[role]
                    rec[f'{head_name}_acc_{role}'] = rx_equal_accuracy(lg.argmax(1), dd['y'], dd['rx'], dd['day'])
                    if role == 'internal_unseen_date' or (role == 'outer_RX'):
                        rec.update({f'{head_name}_{s}': v for s, v in logit_stats(lg).items()})
            return rec
        erasures.append(score(lambda f: f, 'none', 0))
        for kind in KINDS:
            for k in ERASE_K:
                if k > min(PCA_Q, dim): continue
                V, G, share = effect_directions(feats['source'], src['y'], dk, m, txs, dks, k, kind, PCA_Q)
                rec = score(lambda f, V=V, G=G: erase_oblique(f, V, G, c), kind, k); rec['mean_share'] = float(share.mean())
                erasures.append(rec)
        for k in ERASE_K:
            if k > min(PCA_Q, dim): continue
            for draw in range(RANDOM_DRAWS):
                Q = random_basis(dim, k, 1000 * k + draw)
                erasures.append(score(lambda f, Q=Q: erase(f, Q, c), 'random', k, draw))

        if job['dataset'] == 'wisig':
            phys = cache['wisig_physics']; base = preds['source']
            for name, (xp, target) in phys['interventions'].items():
                pr = logits(run_batches(enc, xp, device)).argmax(1)
                sens.append(dict(**ident, intervention=name, sigma_b_hz=phys['sigma_b_hz'],
                                 acc=rx_equal_accuracy(pr, src['y'], src['rx'], src['day']),
                                 acc_drop_pp=100 * (row['acc_source'] - rx_equal_accuracy(pr, src['y'], src['rx'], src['day'])),
                                 flip_rate=float((pr != base).mean()),
                                 moved_to_target=float((pr == target).mean()) if target is not None else np.nan))
            rng = np.random.default_rng(SEED)
            for role in ('internal_seen_date', 'internal_unseen_date'):
                for r_ in attribution(preds[role], data[role], phys['ref'], rng):
                    attrib.append(dict(**ident, role=role, **r_))

        runs.append(row); del model
        for name, rows in (('run_metrics', runs), ('erasure', erasures), ('sensitivity', sens), ('attribution', attrib)):
            pd.DataFrame(rows).to_csv(out / f'{name}.csv', index=False)
        print(json.dumps({k: row[k] for k in ('run_id', 'feat_link_share', 'feat_link_ratio', 'acc_source')}),
              f'{time.perf_counter() - tick:.0f}s', flush=True)

    # ensembles and error overlap (packet order identical across runs: same loader calls, asserted)
    R = pd.DataFrame(runs); ens, overlap = [], []
    for (grp, fold, method), g in R.groupby(['group', 'fold', 'method']):
        for role in [c[4:] for c in R.columns if c.startswith('acc_') and c != 'acc_source']:
            ids = [i for i in g.run_id if (i, role) in probs]
            if len(ids) < 2: continue
            pool = run_pool[ids[0]][role]
            assert all(np.array_equal(run_pool[i][role]['position'], pool['position']) for i in ids)
            P = np.stack([probs[(i, role)] for i in ids]); pr = P.argmax(2); y = pool['y']
            vote = np.array([np.bincount(col, minlength=10).argmax() for col in pr.T])
            singles = [rx_equal_accuracy(q, y, pool['rx'], pool['day']) for q in pr]
            ens.append(dict(group=grp, fold=fold, method=method, role=role, members=len(ids), single_mean=float(np.mean(singles)),
                            single_min=float(np.min(singles)), single_max_posthoc_oracle=float(np.max(singles)),
                            prob_average=rx_equal_accuracy(P.mean(0).argmax(1), y, pool['rx'], pool['day']),
                            majority_vote=rx_equal_accuracy(vote, y, pool['rx'], pool['day'])))
            for (i, pi), (j, pj) in itertools.combinations(zip(ids, pr), 2):
                overlap.append(dict(group=grp, fold=fold, method=method, role=role, run_a=i, run_b=j, **error_overlap(pi != y, pj != y)))
    pd.DataFrame(ens).to_csv(out / 'ensemble.csv', index=False)
    pd.DataFrame(overlap).to_csv(out / 'error_overlap.csv', index=False)
    (out / 'timing.json').write_text(json.dumps(dict(seconds=time.perf_counter() - tick, jobs=len(jobs),
                                                    device=str(device)), indent=2))
    print('done', f'{time.perf_counter() - tick:.0f}s')


if __name__ == '__main__':
    main()
