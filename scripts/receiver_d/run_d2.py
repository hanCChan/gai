#!/usr/bin/env python3
"""D2: Shen frozen representations (30 R0/R1 models: 2 folds x 5 methods x 3 seeds).

Rules: docs/receiver_d/PROTOCOL_D.md (descriptive, no verdict thresholds). Inference plus the
documented source-only refit heads (r1 recipe, every refit recorded). Strict CIS cache only.

Checks before any model is scored: checkpoint status hash; checkpoint scaler equals the scaler
recomputed from the fold's training keys (first 160 per TX); every Strict CIS / labels / rows file
hash equals the registration's input assets.

Erasure is run twice per model:
  dir_pool='fit'    r1 L0 behaviour: directions and refit heads from the training packets
                    (first 160 per TX), identity on the selection pool (positions 160..199)
  dir_pool='half'   protocol fix: the selection pool split into halves by within-cell order;
                    directions from one half, identity on the other, both ways, averaged
Both report frozen and refit heads, source/outer accuracy, per-outer-RX accuracy, per-TX outer
recall, feature change, energy retention, effective rank and condition numbers.

    python3 run_d2.py --root /workspace/rl1/gai_retrain --out OUT [--shard i/n] [--limit N]
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'interaction_l0')]
from link_core import (complete_grid, cell_means, crossfit_energies, effect_directions, erase_oblique,  # noqa: E402
                       random_basis, erase, rx_equal_accuracy)
import run_l0 as R1  # noqa: E402
import d_common as C  # noqa: E402

ERASE_K, KINDS, PCA_Q, RANDOM_DRAWS, REFIT_K = R1.ERASE_K, R1.KINDS, R1.PCA_Q, R1.RANDOM_DRAWS, R1.REFIT_K
N_CLASSES = 10


def halves(data):
    """Split a pool into two halves by alternating packets within each (TX, RX) cell (by position)."""
    h = np.zeros(len(data['y']), int)
    for t in np.unique(data['y']):
        for r in np.unique(data['rx']):
            idx = np.flatnonzero((data['y'] == t) & (data['rx'] == r))
            h[idx[np.argsort(data['position'][idx], kind='stable')][1::2]] = 1
    return h


def sub(data, m):
    return {k: v[m] for k, v in data.items()}


def energies(f, data):
    """Cross-fitted TX/RX/link energies of a single-date pool (alternate packets per cell)."""
    m = np.ones(len(data['y']), bool)
    txs, rxs = complete_grid(data['y'], data['rx'], m)
    A, B = (cell_means(f, data['y'], data['rx'], m, txs, rxs, h, data['position']) for h in (0, 1))
    e = crossfit_energies(A, B)
    tot = e['tx'] + e['rx'] + e['link']
    e.update(rx_share=e['rx'] / tot, link_share=e['link'] / tot, tx_share=e['tx'] / tot)
    return e


def effective_rank(f):
    ev = np.clip(np.linalg.eigvalsh(np.cov(f.T)), 0, None)
    return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30))


def per_tx_recall(pred, y):
    return {int(t): float((pred[y == t] == t).mean()) for t in range(N_CLASSES)}


def verify_inputs(reg):
    bad = [a['path'] for a in reg['input_assets'] if 'shen_p2a8' in a['path'] and C.sha(a['path']) != a['sha256']]
    n = sum('shen_p2a8' in a['path'] for a in reg['input_assets'])
    assert not bad, f'Shen input hash mismatch: {bad}'
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--shard', default='0/1')
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    si, sn = map(int, a.shard.split('/'))
    root = C.add_root(a.root)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    import pandas as pd
    import torch
    from receiver_r01.data import shen
    from train_receiver_r01 import build

    reg = json.loads((root / 'configs/receiver_r01/registration.json').read_text())
    n_assets = verify_inputs(reg)
    jobs = [dict(j, ckpt=root / 'outputs/receiver_r01/formal' / j['run_id'])
            for j in reg['formal_jobs'] if j['dataset'] == 'shen'][si::sn]
    jobs = jobs[:a.limit] if a.limit else jobs
    assert jobs
    (out / 'manifest.json').write_text(json.dumps(dict(
        files={f.name: C.sha(f) for f in sorted(HERE.glob('*.py'))}, code_sha256_l0=R1.code_shas(),
        jobs=[j['run_id'] for j in jobs], shard=a.shard, root=str(root), shen_input_assets_verified=n_assets,
        erase_k=ERASE_K, kinds=KINDS, pca_q=PCA_Q, random_draws=RANDOM_DRAWS, refit=R1.REFIT, refit_k=REFIT_K,
        network_parameters_changed=False, head_refits=True, raw_iq_read=False, started_unix=time.time()), indent=1, default=str))

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    cache, scaler_ref = {}, {}
    runs, eras, per_rx_rows, per_tx_rows, checks = [], [], [], [], []
    n_refits = 0
    tick = time.perf_counter()
    for job in jobs:
        cp, _ = C.load_model(job['ckpt'])
        fold = reg['shen_folds'][job['fold']]
        keys = fold['inner_train'] + fold['inner_validation']
        if job['fold'] not in scaler_ref:
            _, st = shen(reg, keys)
            scaler_ref[job['fold']] = st
        st = scaler_ref[job['fold']]
        sc = cp['scaler']
        scaler_ok = bool(abs(sc['mean'] - st['mean']) < 1e-9 and abs(sc['std'] - st['std']) < 1e-9
                         and sc['fit_keys'] == st['fit_keys'])
        assert scaler_ok, (job['run_id'], sc, st)
        ck = job['fold']
        if ck not in cache:
            cache[ck] = dict(source_fit=shen(reg, keys, sc, 'train')[0], source=shen(reg, keys, sc, 'val')[0],
                             outer=shen(reg, fold['outer_evaluation'], sc, 'all')[0],
                             rx_keys=dict(source=keys, outer=fold['outer_evaluation']))
        data = cache[ck]
        model = build(job).to(device)
        model.load_state_dict(cp['model'])
        model.eval()
        enc, head, make_head = R1.split(model)
        x0 = torch.as_tensor(data['source']['x'][:256], device=device)
        with torch.no_grad():
            gap = float((model(x0) - head(enc(x0))).abs().max())
        assert gap < 1e-4, (job['run_id'], gap)
        feats = {r: R1.run_batches(enc, data[r]['x'], device) for r in ('source_fit', 'source', 'outer')}
        logits = lambda f, h=head: R1.run_batches(h, f, device, 4096)
        ident = dict(run_id=job['run_id'], fold=job['fold'], method=job['method'], seed=job['seed'])
        src, outer, fit = data['source'], data['outer'], data['source_fit']
        base_pred = {r: logits(feats[r]).argmax(1) for r in ('source', 'outer')}
        row = dict(**ident, split_gap=gap, scaler_ok=scaler_ok,
                   acc_source=rx_equal_accuracy(base_pred['source'], src['y'], src['rx'], src['day']),
                   acc_outer=rx_equal_accuracy(base_pred['outer'], outer['y'], outer['rx'], outer['day']),
                   eff_rank_source=effective_rank(feats['source']), eff_rank_outer=effective_rank(feats['outer']))
        for pool, f in (('source', feats['source']), ('outer', feats['outer'])):
            row.update({f'{pool}_{k}': v for k, v in energies(f, data[pool]).items()})
        # the outer-RX pool carries 5 *other* receivers; energies there are a diagnostic only
        runs.append(row)

        def score(transform, kind, k, pool_tag, eval_src, fit_feats, fit_y, draw=None, extra=None, refit=False):
            nonlocal n_refits
            rec = dict(**ident, dir_pool=pool_tag, kind=kind, k=k, draw=draw, **(extra or {}))
            fe = transform(eval_src['f'])
            spread = float(np.mean(np.sum((eval_src['f'] - eval_src['f'].mean(0)) ** 2, 1)))
            rec['feature_change_DT'] = float(np.mean(np.sum((fe - eval_src['f']) ** 2, 1)) / spread)
            b0, b1 = energies(eval_src['f'], eval_src['d']), energies(fe, eval_src['d'])
            rec.update({f'{c}_energy_retained': b1[c] / b0[c] if b0[c] > 0 else np.nan for c in ('tx', 'rx', 'link')})
            heads = [('frozen', head)]
            if refit:
                h, info = R1.refit_head(make_head, transform(fit_feats), fit_y, device)
                n_refits += 1
                rec.update(info)
                heads.append(('refit', h))
            for hn, h in heads:
                ps = logits(fe, h).argmax(1)
                po = logits(transform(feats['outer']), h).argmax(1)
                rec[f'{hn}_acc_source'] = rx_equal_accuracy(ps, eval_src['d']['y'], eval_src['d']['rx'], eval_src['d']['day'])
                rec[f'{hn}_acc_outer'] = rx_equal_accuracy(po, outer['y'], outer['rx'], outer['day'])
                if hn == 'frozen' or refit:
                    for rv, acc in C.per_rx_accuracy(po, outer).items():
                        per_rx_rows.append(dict(**ident, dir_pool=pool_tag, kind=kind, k=k, draw=draw, head=hn, rx=rv,
                                                rx_key=reg['shen_allowed'][rv], acc=acc))
                    if kind in ('none',) or (k == REFIT_K and draw in (None, 0)):
                        for t, rc in per_tx_recall(po, outer['y']).items():
                            per_tx_rows.append(dict(**ident, dir_pool=pool_tag, kind=kind, k=k, head=hn, tx=t, recall=rc))
            return rec

        # ---- dir_pool = fit: r1 L0 (directions/refit on training packets, identity on selection pool)
        dk = C.domain_key(fit)
        m = np.ones(len(dk), bool)
        txs, dks = complete_grid(fit['y'], dk, m)
        c = feats['source_fit'].mean(0)
        dim = feats['source'].shape[1]
        ev_src = dict(f=feats['source'], d=src)
        conds = []
        eras.append(score(lambda f: f, 'none', 0, 'fit', ev_src, feats['source_fit'], fit['y'], refit=True))
        for kind in KINDS:
            for k in ERASE_K:
                info = {}
                V, G, share = effect_directions(feats['source_fit'], fit['y'], dk, m, txs, dks, k, kind, PCA_Q, info=info)
                conds.append(info['cond_Bq'])
                eras.append(score(lambda f, V=V, G=G: erase_oblique(f, V, G, c), kind, k, 'fit', ev_src,
                                  feats['source_fit'], fit['y'], extra=dict(mean_share=float(share.mean()), cond_Bq=info['cond_Bq']),
                                  refit=(k == REFIT_K)))
        for k in ERASE_K:
            for draw in range(RANDOM_DRAWS):
                Q = random_basis(dim, k, 1000 * k + draw)
                eras.append(score(lambda f, Q=Q: erase(f, Q, c), 'random', k, 'fit', ev_src, feats['source_fit'],
                                  fit['y'], draw=draw, refit=(k == REFIT_K)))
        # ---- dir_pool = half: protocol fix (selection pool halves, both ways); frozen head only
        hh = halves(src)
        for a_ in (0, 1):
            A, B = sub(src, hh == a_), sub(src, hh != a_)
            fA, fB = feats['source'][hh == a_], feats['source'][hh != a_]
            dkA = C.domain_key(A)
            mA = np.ones(len(dkA), bool)
            txA, dkAs = complete_grid(A['y'], dkA, mA)
            cA = fA.mean(0)
            evB = dict(f=fB, d=B)
            eras.append(score(lambda f: f, 'none', 0, f'half{a_}', evB, None, None))
            for kind in KINDS:
                for k in ERASE_K:
                    info = {}
                    V, G, share = effect_directions(fA, A['y'], dkA, mA, txA, dkAs, k, kind, PCA_Q, info=info)
                    eras.append(score(lambda f, V=V, G=G, cA=cA: erase_oblique(f, V, G, cA), kind, k, f'half{a_}', evB,
                                      None, None, extra=dict(mean_share=float(share.mean()), cond_Bq=info['cond_Bq'])))
            for k in ERASE_K:
                for draw in range(RANDOM_DRAWS):
                    Q = random_basis(dim, k, 1000 * k + draw)
                    eras.append(score(lambda f, Q=Q, cA=cA: erase(f, Q, cA), 'random', k, f'half{a_}', evB, None, None,
                                      draw=draw))
        checks.append(dict(**ident, split_gap=gap, scaler_ok=scaler_ok, cond_Bq_max=float(max(conds)),
                           finite=bool(all(np.isfinite(v).all() for v in feats.values()))))
        del model
        for nm, rws in (('run_metrics', runs), ('erasure', eras), ('per_rx', per_rx_rows), ('per_tx', per_tx_rows),
                        ('checks', checks)):
            pd.DataFrame(rws).to_csv(out / f'{nm}.csv', index=False)
        print(json.dumps(dict(run_id=job['run_id'], acc_source=round(row['acc_source'], 4),
                              acc_outer=round(row['acc_outer'], 4), rx_share=round(row['source_rx_share'], 3),
                              link_share=round(row['source_link_share'], 3))) + f' {time.perf_counter() - tick:.0f}s',
              flush=True)
    timing = dict(seconds=time.perf_counter() - tick, jobs=len(jobs), refits=n_refits, device=str(device),
                  peak_allocated_MiB=float(torch.cuda.max_memory_allocated() / 2 ** 20) if device.type == 'cuda' else 0.0)
    (out / 'timing.json').write_text(json.dumps(timing, indent=1))
    print('done', json.dumps(timing))


if __name__ == '__main__':
    main()
