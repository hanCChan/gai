"""End-to-end CPU test of the real run_l0.py (preflight -> full) and summarize_l0.py.

Builds a temporary root: a copy of this repository's src/scripts/configs, a trimmed R0/R1
registration (WiSig official x 3 methods x 3 seeds, WiSig ResNet x 1, Shen fold 0 x 2), synthetic
caches in the exact cache formats (WiSig npz with the real internal coverage gap; Shen strict_cis
2000x1x52x126) and randomly initialised checkpoints with valid status SHAs. Models are untrained,
so only engineering behaviour is tested, never scientific outcomes.
"""
import hashlib, json, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from physics import to_cache, TONE_BINS, PERIOD, FS

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def packets(n, tone_db, cfo, snr_db, rng, L=256):
    phase = np.exp(2j * np.pi * rng.uniform(size=12)); g = 10 ** (tone_db / 20)
    t = rng.integers(0, PERIOD, size=n)[:, None] + np.arange(L)[None]
    s = (g * phase * np.exp(2j * np.pi * t[..., None] * TONE_BINS / PERIOD)).sum(-1)
    s = s * np.exp(2j * np.pi * cfo * np.arange(L) / FS)
    p = np.mean(np.abs(s) ** 2)
    return s + (rng.normal(size=s.shape) + 1j * rng.normal(size=s.shape)) * np.sqrt(p / 10 ** (snr_db / 10) / 2)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_root(tmp):
    root = tmp / 'root'
    for sub in ('src', 'scripts', 'configs'):
        shutil.copytree(REPO / sub, root / sub, ignore=shutil.ignore_patterns('__pycache__'))
    reg = json.loads((REPO / 'configs/receiver_r01/registration.json').read_text())
    keep = [j for j in reg['formal_jobs'] if j['dataset'] == 'wisig' and j['backbone'] == 'official' and j['method'] in ('ce', 'mldg_rx_date', 'pcgrad')]
    keep += [j for j in reg['formal_jobs'] if j['backbone'] == 'resnet'][:1]
    keep += [j for j in reg['formal_jobs'] if j['dataset'] == 'shen' and j['fold'] == 0][:2]
    reg['formal_jobs'] = keep; reg['wisig_cache'] = str(tmp / 'wcache'); reg['shen_cache'] = str(tmp / 'scache')
    (root / 'configs/receiver_r01/registration.json').write_text(json.dumps(reg))
    rng = np.random.default_rng(0); roles = reg['wisig']['roles']
    tx_cfo = rng.normal(scale=20e3, size=10); rx_cfo = rng.normal(scale=3e3, size=32)
    tx_t = rng.normal(scale=1.5, size=(10, 12)); rx_t = rng.normal(size=(32, 12)); link = rng.normal(size=(10, 32, 12))
    (tmp / 'wcache').mkdir()
    for r in roles['source'] + roles['internal_rx']:
        source = r in roles['source']
        pos = np.r_[np.arange(0, 120, 4), np.arange(160, 200, 2)] if source else np.arange(20)
        for d in (0, 1, 2):
            xs, ys, ps = [], [], []
            for t in range(10):
                if not source and r == roles['internal_rx'][0] and d < 2 and t not in (1, 4, 5, 7):
                    continue
                xs.append(to_cache(packets(len(pos), tx_t[t] + rx_t[r] + link[t, r], tx_cfo[t] + rx_cfo[r], 25.0, rng)))
                ys += [t] * len(pos); ps += pos.tolist()
            np.savez(tmp / 'wcache' / f'rx{r:02d}_day{d}.npz', x=np.concatenate(xs), y=np.array(ys), position=np.array(ps))
    for key in reg['shen_allowed']:
        k = tmp / 'scache' / key; k.mkdir(parents=True)
        y = np.repeat(np.arange(10), 200)
        np.save(k / 'strict_cis.npy', (rng.normal(size=(2000, 1, 52, 126)) + 0.3 * y[:, None, None, None]).astype(np.float32))
        np.save(k / 'labels.npy', y); np.save(k / 'rows.npy', np.arange(2000))
        (k / 'manifest.json').write_text(json.dumps({'source': {'role': 'train'}}))
    sys.path[:0] = [str(root / 'src'), str(root / 'scripts')]
    import torch
    from train_receiver_r01 import build
    from receiver_r01.data import shen
    for j in keep:
        run = root / 'outputs/receiver_r01/formal' / j['run_id']; run.mkdir(parents=True)
        torch.manual_seed(j['seed']); model = build(j)
        scaler = None
        if j['dataset'] == 'shen':
            fold = reg['shen_folds'][j['fold']]; scaler = shen(reg, fold['inner_train'] + fold['inner_validation'])[1]
        torch.save(dict(model=model.state_dict(), job=j, scaler=scaler), run / 'selected.pt')
        (run / 'status.json').write_text(json.dumps(dict(state='completed', checkpoint_sha256=sha(run / 'selected.pt'))))
    return root


def main():
    import os, torch
    n_gpu = torch.cuda.device_count(); shards = max(1, min(4, n_gpu))
    tmp = Path(tempfile.mkdtemp()); root = build_root(tmp)
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    run = lambda *a, gpu='0': subprocess.run([sys.executable, *map(str, a)], check=True, env=dict(env, CUDA_VISIBLE_DEVICES=gpu))
    pre, full = tmp / 'pre', tmp / 'full'
    run(HERE / 'run_l0.py', '--root', root, '--out', pre, '--preflight')
    P = json.loads((pre / 'preflight.json').read_text())
    print('preflight decision', P['decision'], P['reasons'], 'device', P['timing']['device'], 'peak MiB', round(P['timing']['peak_allocated_MiB']))
    assert set(P['per_architecture']) == {'wisig/official/r01', 'wisig/resnet/r01', 'shen/rank16/r01'}
    import pandas as pd
    C = pd.read_csv(pre / 'checks.csv'); assert C.finite_all.all() and C.refit_loss_decreased.all() and (C.refits == 9).all()
    print(C[['run_id', 'split_gap', 'cond_Bq_max', 'seconds_job']].to_string(index=False))
    V = pd.read_csv(pre / 'intervention_validation.csv'); print(V[['intervention', 'error_median_abs', 'error_rms', 'tone_leakage_rms']].round(4).to_string(index=False))
    if n_gpu: assert P['decision'] == 'GO_FULL_L0', P['reasons']
    else:  # CPU timing may exceed the GPU budget cap; allow the full-mode test only in that case
        assert all(r.startswith('budget') for r in P['reasons']), P['reasons']
        P['decision'] = 'GO_FULL_L0'; (pre / 'preflight.json').write_text(json.dumps(P))
    procs = [subprocess.Popen([sys.executable, str(HERE / 'run_l0.py'), '--root', str(root), '--out', str(full / f'shard{i}'), '--full',
                               '--preflight-dir', str(pre), '--shard', f'{i}/{shards}'], env=dict(env, CUDA_VISIBLE_DEVICES=str(i) if n_gpu else ''))
             for i in range(shards)]
    assert all(p.wait() == 0 for p in procs), 'a shard failed'
    run(HERE / 'merge_l0.py', '--shards', *[full / f'shard{i}' for i in range(shards)], '--out', full / 'merged')
    M = full / 'merged'
    for f in ('run_metrics', 'erasure', 'sensitivity', 'attribution', 'ensemble', 'error_overlap', 'checks'):
        assert (M / f'{f}.csv').exists(), f
    E = pd.read_csv(M / 'erasure.csv')
    assert {'none', 'link', 'rx', 'cond', 'random'} <= set(E.kind) and E.feature_change_DT.notna().all()
    assert (E[E.kind == 'none'].feature_change_DT == 0).all() and len(pd.read_csv(M / 'run_metrics.csv')) == 12
    assert len(pd.read_csv(M / 'ensemble.csv')) > 0
    run(HERE / 'summarize_l0.py', '--l0', M)
    S = json.loads((M / 'summary.json').read_text()); print('verdict on untrained models:', S['interaction_shortcut_verdict'])
    print(json.loads((M / 'timing.json').read_text()))
    shutil.rmtree(tmp); print(f'PASS run_l0 end-to-end (preflight, guarded full on {shards} shard(s), merge, summary)')


if __name__ == '__main__':
    main()
