#!/usr/bin/env python3
"""Freeze G1 roles and sampling before opening the multi-receiver signals."""
import datetime,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shen_p2a.cache import write_json
from shen_p2a8.g1_data import build_shen
if __name__=='__main__':
    out=ROOT/'configs/shen_p2a8';path=out/'g1_registration.json';assert not path.exists()
    historical=json.loads((ROOT/'configs/shen_p2a/p3_history_review.candidate.json').read_text())
    prior=json.loads(Path('/home/keyan/code-y/inventory/shen_stage1_audit_20261011/audit.json').read_text())['files']
    reg={k:historical[k] for k in ('train_receivers','development_receivers','final_locked_receivers')}
    reg.update(stage='G1_first_baselines',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_samples_per_TX_RX=200,sample_selection_seed=20261012,methods=['full_iq','strict_iq','full_cis','strict_cis'],
        confirmation_seeds=[20261012,20261013,20261014],cv_seed=20261012,lr=.0003,weight_decay=0.,batch_size=64,max_cv_epochs=20,
        source_receiver_folds=[reg['train_receivers'][::2],reg['train_receivers'][1::2]],final_epoch_rule='ceil(mean(best epoch across two source receiver folds)); same epoch for all final seeds',
        osu_receivers=['RX1','RX2'],osu_epochs=20,osu_periodic_coherence_min=.8,osu_minimum_windows_per_class=20,
        dates_sessions_status='root multi-RX files unknown, drift dates are named day indices; OSU SigMF dates known',
        local_RF_status='Shen Fs/BW/SF paper verified but local provenance unverified; OSU Fs verified/BW and SF assumed',
        research_unseen_external_holdout=False,locked_access_enabled=False,G2_G3_automatic_start=False,P2_B_P3_started=False)
    items=[]
    for rx in reg['train_receivers']:
        items.append(dict(receiver=rx,relative=rx+'_train.h5',role='train',key='shen_'+rx+'_train'))
    for rx in reg['train_receivers']+reg['development_receivers']:
        items.append(dict(receiver=rx,relative=rx+'_test.h5',role='evaluation',key='shen_'+rx+'_test'))
    for rx in ('rtl_6','n210_1'):
        for day in (1,2):items.append(dict(receiver=rx,relative=f'Test/{rx}_day{day}_wireless_test.h5',role='drift_evaluation',key=f'shen_{rx}_day{day}'))
    for item in items:
        matched=[x for x in prior if x['path'].endswith('/'+item['relative'])];assert len(matched)==1
        item['expected_sha256']=matched[0]['sha256']
    reg['files']=items;write_json(path,reg)
    for item in items:build_shen(ROOT,item,reg)
    write_json(ROOT/'outputs/shen_p2a8/g1_shen_cache_status.json',dict(status='completed',registered_sources=len(items),source_hashes_match=True,locked_files_opened=False))
