#!/usr/bin/env python3
"""Register the WiSig-only subset of the frozen R0/R1 matrix for the environment re-run.

This machine has no Shen caches (dataset must be downloaded separately by the user), so the
Shen arms of the frozen matrix are registered separately once that data arrives. The WiSig
arms below are copied VERBATIM from the published configs/receiver_r01/registration.json
(commit 9f53579..84a360b, run on the original server): same seeds, LRs, epochs, inner split,
methods, and source-only selection. Nothing is re-chosen.

Shen arms are intentionally absent: train_receiver_r01.py reads shen jobs from the same
registration, so Shen runs will use a Shen-only registration with the same provenance rules.

    python3 prepare_receiver_r01_wisig_only.py
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from wisig_w01_common import sha, dump

PUBLISHED = Path('/workspace/rl1/gai/configs/receiver_r01/registration.json')


def main():
    out = ROOT / 'outputs/receiver_r01'; public = ROOT / 'reports/receiver_r01/2026-10-20_local'
    public.mkdir(parents=True, exist_ok=False)
    # The repository ships the historical registration (committed); preserve it as provenance,
    # then write the local WiSig-only registration in its place. The historical file is untouched
    # in the published branch; this copy lives on experiment/l0-retrain-wisig only.
    hist = ROOT / 'configs/receiver_r01/registration_published.json'
    if not hist.exists():
        (ROOT / 'configs/receiver_r01/registration.json').rename(hist)
    old = json.loads(PUBLISHED.read_text())
    assert json.loads(hist.read_text()) == old, 'published registration drifted'
    w_reg = json.loads((ROOT / 'configs/wisig_w01/registration.json').read_text())
    assert w_reg['roles'] == old['wisig']['roles'], 'WiSig split changed'
    assert [a['sha256'] for a in json.loads((ROOT / 'outputs/wisig_w01/preflight.json').read_text())['assets']]

    reg = dict(
        registered_utc=None,  # filled below
        provenance=dict(original_registration=str(PUBLISHED), original_commit='84a360b (R0/R1 completed)',
                        rerun_reason='original server released; frozen WiSig arms re-fit on this machine',
                        wisig_arms_copied_verbatim=True, shen_arms_deferred_until_dataset_download=True),
        base_remote_commit=old['base_remote_commit'], branch='experiment/l0-retrain-wisig',
        reserved_gpu_hours_cap=old['reserved_gpu_hours_cap'],
        formal_seeds=old['formal_seeds'], search_seed=old['search_seed'],
        wisig=old['wisig'],
        wisig_cache=str(ROOT / 'outputs/wisig_w01/cache'),
        wisig_inner_train=old['wisig_inner_train'], wisig_inner_held=old['wisig_inner_held'],
        shen_cache=None, shen_allowed=[], shen_folds=old['shen_folds'],  # not used by wisig jobs
        training_positions=old['training_positions'], source_validation_positions=old['source_validation_positions'],
        epochs=old['epochs'], selection=old['selection'], methods=old['methods'], diagnostics=old['diagnostics'],
        final_data_access=False, extra_downloads=False, r2_fit_authorized=False,
        search_jobs=[j for j in old['search_jobs'] if j['dataset'] == 'wisig'],
        formal_jobs=[j for j in old['formal_jobs'] if j['dataset'] == 'wisig'])
    import datetime
    reg['registered_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    assert len(reg['search_jobs']) == 14 and len(reg['formal_jobs']) == 21, (len(reg['search_jobs']), len(reg['formal_jobs']))
    # verify every job equals the published one (lr/epochs/seed/run_id)
    pub = {(j['run_id']): j for j in old['search_jobs'] + old['formal_jobs']}
    for j in reg['search_jobs'] + reg['formal_jobs']:
        assert j == pub[j['run_id']], j['run_id']
    # freeze inputs: development cache SHAs only (no final RX/date3/equalized exist in this cache)
    pre = json.loads((ROOT / 'outputs/wisig_w01/preflight.json').read_text())
    reg['input_assets'] = [dict(path=str((ROOT / 'outputs/wisig_w01/cache') / Path(a['path']).name), sha256=a['sha256'])
                           for a in pre['assets']]
    for a in reg['input_assets']:
        assert sha(a['path']) == a['sha256'], a['path']
    # data preflight on the fitting split
    from receiver_r01.data import wisig as load_wisig
    import numpy as np
    data = load_wisig(reg, reg['wisig']['roles']['source'], [0, 1, 2], [0, 200])
    n = sum(int(((data['rx'] == r) & (data['day'] == d) & (data['y'] == t)).sum()) >= 160
            for r in reg['wisig']['roles']['source'] for d in (0, 1, 2) for t in range(10))
    assert n == 420 and np.isfinite(data['x']).all()
    dump(ROOT / 'configs/receiver_r01/registration.json', reg)
    dump(public/'data_preflight.json', dict(wisig_source_cells=n, all_source_ten_classes_valid=True, final_rx_read=False,
        final_date_read=False, equalized_read=False, archive_provenance=reg['wisig']['archive_sha256'],
        note='WiSig-only re-run registration; jobs copied verbatim from the frozen matrix'))
    import csv
    with (public/'source_10class_coverage.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=['rx_index', 'rx', 'date_index', 'tx_index', 'valid_packets'], lineterminator='\n')
        w.writeheader()
        for r in reg['wisig']['roles']['source']:
            for d in (0, 1, 2):
                for t in range(10):
                    w.writerow(dict(rx_index=r, rx=reg['wisig']['rx_names'][r], date_index=d, tx_index=t,
                                    valid_packets=int(((data['rx'] == r) & (data['day'] == d) & (data['y'] == t)).sum())))
    print(json.dumps(dict(search_jobs=len(reg['search_jobs']), formal_jobs=len(reg['formal_jobs']),
                          inner_train=reg['wisig_inner_train'], inner_held=reg['wisig_inner_held'],
                          source_rx=reg['wisig']['roles']['source'])))


if __name__ == '__main__':
    main()
