#!/usr/bin/env python3
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from d0_class_pair.metrics import offdiag
from d05_b0.damage import damage_predictors,evaluate_damage
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_d0_class_pair import write_json,write_csv,sha,median
ROOT=Path(__file__).resolve().parents[1]

def main():
    start=time.perf_counter();r=json.loads((ROOT/'configs/d05_b0/registration.json').read_text())
    seal=json.loads((ROOT/'configs/d05_b0/pre_execution_seal.json').read_text())
    for path,digest in seal['files_sha256'].items():assert sha(ROOT/path)==digest,path
    original=json.loads((ROOT/r['d0_asset_registration']).read_text());assert sha(ROOT/r['d0_asset_registration'])==r['d0_asset_registration_sha256']
    matrices={};rows_by_rx={};prov=[]
    for a in original['prediction_assets']:
        assert sha(a['path'])==a['sha256'];rx=a['receiver'];method=a['method']
        assert rx in r['folds'][a['fold']]
        with np.load(a['path'],allow_pickle=False) as f:z=f['logits'];y=f['labels'];rows=f['rows']
        assert z.shape==(2000,10) and np.isfinite(z).all() and np.array_equal(np.bincount(y,minlength=10),np.full(10,200))
        c=np.zeros((10,10),int);np.add.at(c,(y,z.argmax(1)),1);matrices[rx,method]=c/200
        rows_by_rx[rx,method]=(rows.copy(),y.copy());prov.append(dict(receiver=rx,method=method,fold=a['fold'],input_sha256=a['sha256']))
    for rx in r['source_receivers']:
        assert all(np.array_equal(a,b) for a,b in zip(rows_by_rx[rx,'full_cis'],rows_by_rx[rx,'strict_cis']))
    pair_rows=[];profiles=[];vectors=[]
    for fold,rxs in enumerate(r['folds']):
        for probe in rxs:
            target=offdiag(matrices[probe,'strict_cis']-matrices[probe,'full_cis'])
            for edge,(i,j) in enumerate((i,j) for i in range(10) for j in range(10) if i!=j):
                pair_rows.append(dict(receiver=probe,fold=fold,hardware_family=r['hardware_families'][probe],tx_i=i+31,tx_j=j+31,full_confusion=matrices[probe,'full_cis'][i,j],strict_confusion=matrices[probe,'strict_cis'][i,j],signed_delta=target[edge],positive_delta=max(0,target[edge])))
            for control in ('same_paired_cnn_loro','different_family_reference','cross_paired_cnn_transfer'):
                refs=[rx for rx in (r['folds'][1-fold] if control=='cross_paired_cnn_transfer' else rxs) if rx!=probe]
                if control=='different_family_reference':refs=[rx for rx in refs if r['hardware_families'][rx]!=r['hardware_families'][probe]]
                if not refs:continue
                pp=damage_predictors([matrices[rx,'full_cis'] for rx in refs],[matrices[rx,'strict_cis'] for rx in refs])
                result={name:evaluate_damage(score,target,r['d05']['top_k']) for name,score in pp.items()}
                for name,score in pp.items():
                    meta=dict(fold=fold,probe_receiver=probe,hardware_family=r['hardware_families'][probe],control=control,predictor=name,reference_count=len(refs),reference_receivers=';'.join(refs))
                    ev=result[name];profile=dict(**meta,**ev)
                    for baseline in pp:
                        a=ev['positive_mass_capture'];b=result[baseline]['positive_mass_capture']
                        profile['capture_gain_over_'+baseline]=None if a is None or b is None else a-b
                    profiles.append(profile)
                    for edge,(i,j) in enumerate((i,j) for i in range(10) for j in range(10) if i!=j):
                        vectors.append(dict(**meta,tx_i=i+31,tx_j=j+31,predicted_score=float(score[edge]),observed_signed_delta=float(target[edge])))
    summaries=[]
    for fold in range(2):
        for control in ('same_paired_cnn_loro','different_family_reference','cross_paired_cnn_transfer'):
            ss=[v for v in profiles if v['fold']==fold and v['control']==control and v['predictor']=='signed_delta_mean']
            summaries.append(dict(fold=fold,control=control,probe_receivers=len(ss),median_signed_spearman=median([v['signed_spearman'] for v in ss]),median_positive_mass_capture=median([v['positive_mass_capture'] for v in ss]),**{'median_capture_gain_over_'+b:median([v['capture_gain_over_'+b] for v in ss]) for b in r['d05']['predictors'] if b!='signed_delta_mean'}))
    primary=[v for v in summaries if v['control']=='same_paired_cnn_loro']
    support=all(v['median_signed_spearman'] is not None and v['median_signed_spearman']>0 and v['median_capture_gain_over_uniform'] is not None and v['median_capture_gain_over_uniform']>0 for v in primary)
    incremental=all(v['median_capture_gain_over_'+b] is not None and v['median_capture_gain_over_'+b]>0 for v in primary for b in ('recall_drop_only','class_difficulty_full','factorized_delta'))
    decision=dict(status='EXPLORATORY_INCREMENTAL_DAMAGE_SIGNAL' if support and incremental else 'DAMAGE_SIGNAL_WITHOUT_STABLE_BASELINE_GAIN' if support else 'ABSOLUTE_RISK_ONLY_DAMAGE_SCREEN_FAILED',signed_damage_predictive_screen=support,incremental_primary_baseline_screen=incremental,primary_summaries=primary,seed=20261012,independent_holdout=False,causal_over_alignment_evidence=False,b0_still_authorized=True,b1_training_authorized=False)
    out=ROOT/'reports/d05_b0/2026-10-13/d05';out.mkdir(parents=True,exist_ok=False)
    for name,records in [('all_directed_deltas',pair_rows),('profile_evaluation',profiles),('prediction_vectors',vectors),('summary',summaries),('input_provenance',prov)]:write_csv(out/(name+'.csv'),records)
    write_json(out/'decision.json',decision);write_json(out/'execution.json',dict(seconds=time.perf_counter()-start,new_training=0,new_inference=0,packet_pairs=20000,source_receivers=10,all_d0_input_hashes_verified=True,seal_sha256=sha(ROOT/'configs/d05_b0/pre_execution_seal.json')))
    print(json.dumps(decision,indent=2))

if __name__=='__main__':main()
