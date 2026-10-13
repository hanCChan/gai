#!/usr/bin/env python3
"""C0 provenance/availability audit. No IQ, prediction-value or model inference.

OSU Table II numbers are re-aggregated from historical per-seed summaries,
never claimed as reconstruction of missing per-window/per-capture outputs.
Shen NPZ headers and source HDF5 metadata are inspected without decoding IQ
or evaluating the cached logits. Missing chronology stops C1/C2.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OSU=Path('/home/keyan/code-y')
HISTORY=Path('/home/keyan/code-y-p2a')
SOURCE=['b200_mini_1','b210_1','pluto_1','rtl_1','rtl_2','rtl_3','rtl_5','rtl_6','rtl_7','rtl_9']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def npz_headers(path):
    result={}
    with zipfile.ZipFile(path) as f:
        for name in f.namelist():
            if not name.endswith('.npy'):raise ValueError('Unexpected NPZ member')
            with f.open(name) as g:
                version=np.lib.format.read_magic(g)
                shape,fortran,dtype=np.lib.format._read_array_header(g,version)
                if dtype.hasobject:raise ValueError('Object arrays not allowed')
                result[name[:-4]]=dict(shape=list(shape),dtype=str(dtype),fortran=fortran)
    return result


def main(out):
    if out.exists():raise ValueError('New output directory required')
    out.mkdir(parents=True)
    sources=[]
    def source(path,role,private=False):
        item={'path':str(path),'role':role,'exists':path.is_file()}
        if item['exists']:item.update(bytes=path.stat().st_size,sha256=sha(path))
        item['private_content_not_published']=private;sources.append(item)
        return item
    original=Path('/home/keyan/private/p0_sync_20261013/rejected_manuscript_15.pdf')
    source(original,'rejected_manuscript',True)
    text=Path('/home/keyan/private/p0_sync_20261013/rejected_manuscript_15.txt').read_text()
    assert 'OOB-Guided Cross-Attentive RF-HSTU Hybrid' in text and '41.5' in text and '43.5' in text
    source(Path('/home/keyan/private/p0_sync_20261013/rejection.txt'),'rejection_letter',True)
    asset=OSU/'inventory/asset_read_audit_current.json';source(asset,'historical_asset_inventory')
    old_inventory={x['path']:x for x in json.loads(asset.read_text())['files']}
    main_root=OSU/'outputs/paper_ready_v3/step1_phase7_clean'
    metric=main_root/'metrics_all.csv';source(metric,'formal_crossday_seed_summaries')
    source(main_root/'multiseed_summary.json','historical_aggregate_crosscheck')
    old_rows=list(csv.DictReader(metric.open()))
    wanted=['A_cnn_iq','B_linear_no_oob','F_cross_attn_chirp_plain']
    seed_rows=[];aggregates=[];checkpoint_info=[]
    for model in wanted:
        subset=[r for r in old_rows if r['model_id']==model]
        for r in subset:
            seed=r['seed'];checkpoint=main_root/'runs'/model/('seed_'+seed)/'best.pt'
            folder=main_root/'outputs'/model/('seed_'+seed)
            available={name:(folder/name).is_file() for name in ['predictions.csv','file_predictions.csv','metrics.json','run_config.json']}
            ck=source(checkpoint,'formal_crossday_checkpoint')
            old=old_inventory.get(str(checkpoint.relative_to(OSU)),{})
            if ck['exists']:
                assert ck['sha256']==old.get('sha256'),'Historical checkpoint hash mismatch'
                checkpoint_info.append(dict(model_id=model,seed=int(seed),sha256=ck['sha256'],
                                            historical_inventory_hash_matches=True,config=old.get('config',{}),
                                            live_deserialization=False))
            seed_rows.append(dict(model_id=model,seed=int(seed),window_acc=float(r['window_acc']),
                                  file_acc=float(r['file_acc']),window_macro_f1=float(r['window_macro_f1']),
                                  file_macro_f1=float(r['file_macro_f1']),n_files=int(r['n_files']),
                                  n_windows=int(r['n_windows']),checkpoint_currently_present=ck['exists'],
                                  checkpoint_historically_reported_present=r['has_ckpt'],
                                  per_window_csv_present=available['predictions.csv'],
                                  per_capture_csv_present=available['file_predictions.csv'],
                                  historical_metrics_path=r['metrics_path'],window_order_available=False,
                                  full_score_vectors_available=False))
        aggregate={'model_id':model,'n_seeds':len(subset),'live_checkpoints':sum(r['checkpoint_currently_present'] for r in seed_rows if r['model_id']==model),
                   'origin':'historical per-seed summaries; NOT logits/capture reconstruction'}
        for k in ['window_acc','file_acc','window_macro_f1','file_macro_f1']:
            a=[float(r[k]) for r in subset]
            aggregate[k+'_mean']=float(np.mean(a))
            aggregate[k+'_std_population_ddof0']=float(np.std(a,ddof=0))
            aggregate[k+'_std_sample_ddof1']=float(np.std(a,ddof=1))
        aggregate['aggregation_gain_pp']=100*(aggregate['file_acc_mean']-aggregate['window_acc_mean'])
        aggregates.append(aggregate)
    write_csv(out/'OSU_per_seed_assets.csv',seed_rows);write_csv(out/'Table_II_historical_reaggregation.csv',aggregates)
    dump(out/'checkpoint_provenance.json',checkpoint_info)
    manifest=OSU/'data/paper/cross_day_day1to5_source_only.csv';source(manifest,'crossday_manifest')
    captures=[]
    for row in csv.DictReader(manifest.open()):
        path=OSU/row['path'];meta=path.with_suffix('.sigmf-meta');info={}
        if meta.is_file():
            source(meta,'archived_osu_metadata')
            m=json.loads(meta.read_text()).get('_metadata',{});cap=m.get('captures',[{}])[0]
            info={'metadata_date':cap.get('core:date'),'sample_rate_metadata':m.get('global',{}).get('core:sample_rate'),
                  'datatype':m.get('global',{}).get('core:datatype')}
        captures.append(dict(capture_id=row['relative_path'],day=int(row['day']),split=row['split'],
                             label=int(row['label']),device_field=int(row['device']),
                             raw_iq_path=str(path),raw_iq_present=path.is_file(),
                             raw_iq_bytes=path.stat().st_size if path.is_file() else 0,
                             metadata_present=meta.is_file(),**info))
    write_csv(out/'OSU_per_capture_assets.csv',captures)
    # Find all archived predictions/caches, but inspect text/headers only.
    candidates=[]
    for root in [OSU/'outputs',OSU/'legacy/day1_day2/outputs']:
        for p in sorted(root.rglob('*')):
            if p.is_file() and (p.name in ('predictions.csv','file_predictions.csv') or p.suffix in ('.npy','.npz') or 'logit' in p.name.lower()):
                d=dict(path=str(p),bytes=p.stat().st_size,formal_day1to3_day4_day5=False)
                if p.suffix=='.csv':
                    with p.open() as f:d['fields']=next(csv.reader(f))
                else:d['cache_header']=npz_headers(p) if p.suffix=='.npz' else 'not opened'
                candidates.append(d)
    dump(out/'OSU_prediction_search.json',candidates)
    # A source header read is not an IQ value read. No datetime/test HDF5 used.
    g1_path=HISTORY/'configs/shen_p2a8/g1_registration.json';source(g1_path,'frozen_g1_roles')
    g1=json.loads(g1_path.read_text());assert g1['train_receivers']==SOURCE
    raw_base=(OSU/'data/external/shen_receiver_agnostic').resolve()
    shen=[];pred_headers=[];access=[]
    for rx in SOURCE:
        path=raw_base/(rx+'_train.h5');assert path.resolve()==path and not path.is_symlink()
        access.append(dict(path=str(path),receiver=rx,role='source_train',mode='read_only',operation='HDF5_headers_and_attributes_only',iq_values_read=0))
        with h5py.File(path,'r') as f:
            fields={}
            for name in f.keys():
                if not isinstance(f.get(name,getlink=True),h5py.HardLink):raise ValueError('Unexpected external link')
                d=f[name];fields[name]=dict(shape=list(d.shape),dtype=str(d.dtype),attrs={k:str(v) for k,v in d.attrs.items()})
            attrs={k:str(v) for k,v in f.attrs.items()}
        names=' '.join(list(fields)+list(attrs)).lower()
        temporal_fields_present=any(k in names for k in ['timestamp','session','capture_id','sample_offset','packet_time'])
        shen.append(dict(receiver=rx,path=str(path),bytes=path.stat().st_size,attrs=attrs,fields=fields,
                         explicit_temporal_metadata_present=temporal_fields_present,
                         chronology_independently_verified=False,session_independently_verified=False,
                         expected_raw_sha256_from_g1=next(z['expected_sha256'] for z in g1['files'] if z['receiver']==rx and z['role']=='train'),
                         raw_sha256_recomputed_this_round=False))
    pred_base=HISTORY/'outputs/eisr/v1_20261012/E0_predictions'
    for fold in range(2):
        for method in ['full_cis','strict_cis']:
            folder=pred_base/f'CV{fold}_{method}';source(folder/'manifest.json','historical_prediction_provenance')
            meta=json.loads((folder/'manifest.json').read_text())
            assert {s['receiver'] for s in meta['sources']}==set(g1['source_receiver_folds'][fold])
            for entry in meta['sources']:
                rx=entry['receiver'];assert rx in SOURCE
                p=folder/(rx+'.npz');src=source(p,'source_heldout_prediction_cache')
                headers=npz_headers(p)
                assert set(headers)=={'logits','probabilities','labels','rows'}
                assert headers['logits']['shape']==[2000,10] and headers['probabilities']['shape']==[2000,10]
                pred_headers.append(dict(receiver=rx,fold=fold,method=method,path=str(p),sha256=src['sha256'],headers=headers,
                                         expected_packets_per_tx=200,chrono_verified=False,checkpoint_sha256=meta['checkpoint_sha256'],epoch=meta['epoch'],
                                         cached_scores_evaluated_this_round=False))
    dump(out/'Shen_source_headers.json',shen);dump(out/'Shen_prediction_asset_headers.json',pred_headers)
    dump(out/'access_log.json',access)
    for p in [OSU/'src/rfhstu/data.py',OSU/'scripts/evaluate.py',HISTORY/'scripts/run_eisr_v1.py',ROOT/'scripts/audit_c_evidence_assets.py',ROOT/'docs/c_evidence/EXECUTION_PLAN.md']:
        source(p,'source_code_or_plan')
    dump(out/'INPUT_MANIFEST.json',sources)
    decision={'decision':'STOP_INVALID_GROUPING','secondary_osu_decision':'STOP_ASSET_INSUFFICIENT',
              'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'C0':'completed','C1':'not_started; chronological grouping unverified','C2':'not_started; C1 precondition failed',
              'user_declined_unordered_fusion':True,'p0_reopened':False,'new_training_runs':0,'new_model_inference_runs':0,
              'raw_iq_values_read':0,'prediction_values_evaluated':0,'development_locked_or_day_signal_access':False,
              'main_osu_captures_in_manifest':len(captures),'main_osu_iq_files_present':sum(c['raw_iq_present'] for c in captures),
              'formal_osu_main_prediction_files_present':sum(r['per_window_csv_present'] for r in seed_rows),
              'full_rf_hstu_checkpoints_present':next(r['live_checkpoints'] for r in aggregates if r['model_id']=='F_cross_attn_chirp_plain'),
              'cnn_iq_checkpoints_present':next(r['live_checkpoints'] for r in aggregates if r['model_id']=='A_cnn_iq'),
              'no_oob_checkpoints_present':next(r['live_checkpoints'] for r in aggregates if r['model_id']=='B_linear_no_oob'),
              'shen_source_hdf5_metadata_files_inspected':len(shen),'shen_full_score_cache_files_available':len(pred_headers),
              'shen_chronology_verified_files':sum(r['chronology_independently_verified'] for r in shen),
              'historical_summary_agrees_with_table_ii_not_full_reproduction':True,
              'future_research_value':'Cannot assess correlation/sequential mechanism or baseline necessity until chronology/session and source predictions are verifiable',
              'next_training_authorized':False}
    assert decision['main_osu_iq_files_present']==0 and decision['formal_osu_main_prediction_files_present']==0
    assert not any(r['explicit_temporal_metadata_present'] for r in shen)
    dump(out/'decision.json',decision)
    dump(out/'artifact_sha256.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(decision,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output.resolve())
