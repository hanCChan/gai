#!/usr/bin/env python3
"""Freeze the entire finite matrix and validate existing authorized data before fitting."""
import datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from wisig_w01_common import sha,dump
from receiver_r01.data import wisig


def main():
    out=ROOT/'outputs/receiver_r01';public=ROOT/'reports/receiver_r01/2026-10-13'
    assert not (ROOT/'configs/receiver_r01/registration.json').exists()
    old=json.loads((ROOT/'configs/wisig_w01/registration.json').read_text())
    b1=json.loads((ROOT/'configs/b1_baselines/registration.json').read_text())
    source=sorted(old['roles']['source']);hold=source[::4];train=[r for r in source if r not in hold]
    reg=dict(registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        base_remote_commit='e64c5a3fe855684b082ff29183001f01b74730be',branch='experiment/r0-r1-receiver-held-out',
        authorization_sha256=sha('/root/.codex/attachments/0b3ad324-db83-4689-a518-d546725c4844/已粘贴的文本.txt'),
        reserved_gpu_hours_cap=6.,formal_seeds=[20261013,20261014,20261015],search_seed=20261016,
        wisig=old,wisig_cache='/home/keyan/code-y-w0w1/outputs/wisig_w01/cache',
        wisig_inner_train=train,wisig_inner_held=hold,
        shen_cache=b1['history_root']+'/outputs/shen_p2a8/g1_cache_v1',
        shen_allowed=['shen_'+r+'_train' for r in b1['source_receivers']],shen_folds=b1['matrix']['source_splits'],
        training_positions=dict(wisig=[0,120],shen='first160 cached packets of each TX per fitting RX'),
        source_validation_positions=dict(wisig=[120,160],shen='remaining40 cached packets per TX per fitting RX'),
        epochs=dict(wisig_search=50,wisig_formal=100,shen_search=10,shen_formal=20),
        selection='min receiver-equal CE on source held RX; tie lower LR; checkpoint uses source-only CE (SWAD published LossValley)',
        methods=dict(mldg='full second-order (L_train + beta L_held(theta-alpha*g_train))/(1+beta); inner plain SGD',
            alpha=.01,beta=1.,swad=dict(n_converge=3,n_tolerance=6,tolerance_ratio=.3,dense='every outer step, epoch segments',no_early_training_stop=True),
            groupdro='RXxTX multiplicative weights, eta=.01, uniform initialization',
            pcgrad='standard symmetric PCGrad on two random RX partitions; no new algorithm'),
        diagnostics=dict(wisig_interval_epochs=5,shen_interval_epochs=1,batch='first balanced training batch; held RX excluded from inner gradient',
            includes='deterministic eval-mode inner SGD and actual outer update per held RX/TX losses; gradient cosine',interpretation='source simulation only, not target causal proof'),
        final_data_access=False,extra_downloads=False,r2_fit_authorized=False,search_jobs=[],formal_jobs=[])
    methods=['ce','swad','mldg_rx','groupdro','pcgrad']
    specs=[('wisig','official',0,methods+['mldg_rx_date']),('wisig','resnet',0,['ce'])]
    specs += [('shen','rank16',f,methods) for f in range(2)]
    for dataset,backbone,fold,ms in specs:
        for method in ms:
            common=dict(dataset=dataset,backbone=backbone,fold=fold,method=method,alpha=.01,beta=1.)
            for lr in ([.0003,.0005] if dataset=='wisig' else [.0001,.0003]):
                reg['search_jobs'].append(dict(common,stage='search',seed=reg['search_seed'],lr=lr,
                    epochs=reg['epochs'][dataset+'_search'],run_id=f'{dataset}_{backbone}_CV{fold}_{method}_search_lr{lr}'))
            for seed in reg['formal_seeds']:
                reg['formal_jobs'].append(dict(common,stage='formal',seed=seed,lr='source_selected',
                    epochs=reg['epochs'][dataset+'_formal'],run_id=f'{dataset}_{backbone}_CV{fold}_{method}_seed{seed}'))
    assert len(reg['search_jobs'])==34 and len(reg['formal_jobs'])==51
    assets=[]
    pre=json.loads(Path('/home/keyan/code-y-w0w1/outputs/wisig_w01/preflight.json').read_text())
    for a in pre['assets']:
        assert a['rx_index'] not in old['roles']['final_rx'] and a['date_index'] in [0,1,2]
        assert sha(a['path'])==a['sha256'];assets.append(a)
    for rel,a in b1['feature_assets'].items():
        p=Path(b1['history_root'])/rel
        assert p.name in ['strict_cis.npy','labels.npy','rows.npy','manifest.json'] and p.parent.name in reg['shen_allowed']
        assert sha(p)==a['sha256'];assets.append(dict(path=str(p),sha256=a['sha256']))
    reg['input_assets']=assets
    data=wisig(reg,source,[0,1,2],[0,200]);coverage=[]
    for r in source:
        for d in [0,1,2]:
            for t in range(10):
                mask=(data['rx']==r)&(data['day']==d)&(data['y']==t)
                n=int(mask.sum());assert n>=160
                coverage.append(dict(rx_index=r,rx=old['rx_names'][r],date_index=d,tx_index=t,valid_packets=n))
    assert np.isfinite(data['x']).all() and len(coverage)==420
    import csv
    with (public/'source_10class_coverage.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(coverage[0]),lineterminator='\n');w.writeheader();w.writerows(coverage)
    dump(ROOT/'configs/receiver_r01/registration.json',reg)
    dump(public/'data_preflight.json',dict(assets_checked=len(assets),wisig_source_cells=420,all_source_ten_classes_valid=True,
        no_new_raw_iq_access=True,final_rx_read=False,final_date_read=False,equalized_read=False,shen_development_locked_read=False,
        archive_provenance=old['archive_sha256'],official_byte_equivalence_proven=False))
    print(json.dumps(dict(search_jobs=34,formal_jobs=51,source_hold_rx=hold,inputs=len(assets))))

if __name__=='__main__':main()
