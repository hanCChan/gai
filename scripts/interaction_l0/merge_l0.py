#!/usr/bin/env python3
"""Combine full-mode shards of run_l0.py; compute ensembles and error overlap. CPU only.

    python merge_l0.py --shards FULL/shard0 FULL/shard1 ... --out FULL/merged
"""
import argparse, itertools, json
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from link_core import rx_equal_accuracy, error_overlap

TABLES = ('run_metrics', 'erasure', 'sensitivity', 'attribution', 'checks')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--shards', nargs='+', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); shards = [Path(s) for s in a.shards]; out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    mans = [json.loads((s / 'manifest.json').read_text()) for s in shards]
    assert all(m['mode'] == 'full' for m in mans) and len({json.dumps(m['code_sha256'], sort_keys=True) for m in mans}) == 1
    jobs = [j for m in mans for j in m['jobs']]
    assert len(jobs) == len(set(jobs)) == mans[0]['all_jobs'], (len(jobs), len(set(jobs)), mans[0]['all_jobs'])
    for t in TABLES:
        frames = [pd.read_csv(s / f'{t}.csv') for s in shards if (s / f'{t}.csv').exists() and (s / f'{t}.csv').stat().st_size > 1]
        if frames: pd.concat(frames, ignore_index=True).to_csv(out / f'{t}.csv', index=False)
    vals = [pd.read_csv(s / 'intervention_validation.csv') for s in shards if (s / 'intervention_validation.csv').exists()]
    if vals:
        assert all(v.round(9).equals(vals[0].round(9)) for v in vals), 'model-free validation differs across shards'
        vals[0].to_csv(out / 'intervention_validation.csv', index=False)
    R = pd.read_csv(out / 'run_metrics.csv'); files = {f.name[:-4]: f for s in shards for f in (s / 'probs').glob('*.npz')}
    ens, overlap = [], []
    for (grp, fold, method), g in R.groupby(['group', 'fold', 'method']):
        roles = sorted({k.split('__')[1] for k in files if k.split('__')[0] in set(g.run_id)})
        for role in roles:
            ids = [i for i in g.run_id if f'{i}__{role}' in files]
            if len(ids) < 2: continue
            arrs = [np.load(files[f'{i}__{role}']) for i in ids]
            for k in ('y', 'rx', 'day', 'position'):
                assert all(np.array_equal(x[k], arrs[0][k]) for x in arrs), (grp, method, role, k)
            P = np.stack([x['prob'] for x in arrs]); pr = P.argmax(2); y, rx, day = arrs[0]['y'], arrs[0]['rx'], arrs[0]['day']
            vote = np.array([np.bincount(col, minlength=10).argmax() for col in pr.T])
            singles = [rx_equal_accuracy(q, y, rx, day) for q in pr]
            ens.append(dict(group=grp, fold=fold, method=method, role=role, members=len(ids), single_mean=float(np.mean(singles)),
                            single_min=float(np.min(singles)), single_max_posthoc_oracle=float(np.max(singles)),
                            prob_average=rx_equal_accuracy(P.mean(0).argmax(1), y, rx, day),
                            majority_vote=rx_equal_accuracy(vote, y, rx, day)))
            for (i, pi), (j, pj) in itertools.combinations(zip(ids, pr), 2):
                overlap.append(dict(group=grp, fold=fold, method=method, role=role, run_a=i, run_b=j, **error_overlap(pi != y, pj != y)))
    pd.DataFrame(ens).to_csv(out / 'ensemble.csv', index=False)
    pd.DataFrame(overlap).to_csv(out / 'error_overlap.csv', index=False)
    tim = [json.loads((s / 'timing.json').read_text()) for s in shards]
    (out / 'timing.json').write_text(json.dumps(dict(shards=len(shards), jobs=len(jobs),
        reserved_gpuh_sum=sum(t['seconds'] for t in tim) / 3600, wall_seconds_max=max(t['seconds'] for t in tim),
        peak_allocated_MiB_max=max(t['peak_allocated_MiB'] for t in tim)), indent=2))
    (out / 'manifest.json').write_text(json.dumps(dict(mode='merged', shards=[str(s) for s in shards], code_sha256=mans[0]['code_sha256'],
                                                       jobs=jobs), indent=2))
    print(json.dumps(json.loads((out / 'timing.json').read_text()), indent=2))


if __name__ == '__main__':
    main()
