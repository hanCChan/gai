#!/usr/bin/env python3
"""D0 archived predictions only. No raw IQ, model inference, or training."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from d0_class_pair.metrics import pair_metrics, predictors, offdiag, evaluate_profile

ROOT = Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def write_csv(path, rows):
    assert rows, path
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def median(values):
    vals=[v for v in values if v is not None]
    return float(np.median(vals)) if vals else None

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='reports/d0_class_pair/2026-10-13_D0');args=p.parse_args()
    start=time.perf_counter()
    output=ROOT / args.output
    if output.exists(): raise RuntimeError('Refusing to overwrite an existing audit')
    reg=json.loads((ROOT/'configs/d0_class_pair/registration.json').read_text())
    seal=json.loads((ROOT/'configs/d0_class_pair/pre_execution_seal.json').read_text())
    for rel, expected in seal['files_sha256'].items(): assert sha(ROOT/rel)==expected, rel
    history=Path(reg['read_only_history_root'])
    provenance=[]
    for group in ('expected_checkpoint_and_config_sha256','expected_index_sha256'):
        for rel,expected in reg[group].items():
            actual=sha(history/rel);assert actual==expected,rel
            provenance.append(dict(path=rel,sha256=actual,kind=group))
    risks={}; datasets={}; pairs=[]; per_rx=[]; per_tx=[]; conf=[]
    checked_manifests=set()
    for asset in reg['prediction_assets']:
        path=Path(asset['path']);rx=asset['receiver'];method=asset['method'];fold=asset['fold']
        assert rx in reg['source_receiver_folds'][fold] and method in reg['methods']
        assert sha(path)==asset['sha256']
        run=f'CV{fold}_{method}'
        cfg=json.loads((history/f'outputs/shen_p2a8/g1_shen_v2/runs/{run}/config.json').read_text())
        expected_fit=[f'shen_{r}_train' for r in reg['source_receiver_folds'][1-fold]]
        expected_eval=[f'shen_{r}_train' for r in reg['source_receiver_folds'][fold]]
        assert cfg['train_keys']==expected_fit and cfg['validation_keys']==expected_eval
        assert cfg['scaler']['fit_source_keys']==expected_fit
        assert cfg['seed']==reg['seed'] and cfg['epochs']==20
        assert cfg['initial_model_payload_sha256']=='3a326536b92db79f2fdd541b5918e2aa44644c56ca2701dbff525a4a99da8376'
        manifest_path=path.parent/'manifest.json'
        manifest=json.loads(manifest_path.read_text())
        assert manifest['checkpoint_sha256']==asset['checkpoint_sha256'] and manifest['epoch']==20
        assert manifest['scaler']==cfg['scaler']
        assert [s['receiver'] for s in manifest['sources']]==reg['source_receiver_folds'][fold]
        if str(manifest_path) not in checked_manifests:
            provenance.append(dict(path=str(manifest_path.relative_to(history)),sha256=sha(manifest_path),kind='prediction_manifest'))
            checked_manifests.add(str(manifest_path))
        with np.load(path,allow_pickle=False) as archive:
            z=archive['logits'];prob=archive['probabilities'];y=archive['labels'];rows=archive['rows']
        assert z.shape==(2000,10) and prob.shape==z.shape
        assert y.shape==rows.shape==(2000,) and len(np.unique(rows))==2000
        assert np.array_equal(np.bincount(y,minlength=10),np.full(10,200))
        assert np.isfinite(z).all() and np.isfinite(prob).all()
        soft=np.exp(z-z.max(axis=1,keepdims=True));soft/=soft.sum(axis=1,keepdims=True)
        assert np.allclose(soft,prob,rtol=1e-6,atol=1e-7)
        cache=history/f'outputs/shen_p2a8/g1_cache_v1/shen_{rx}_train'
        assert np.array_equal(rows,np.load(cache/'rows.npy',allow_pickle=False))
        assert np.array_equal(y,np.load(cache/'labels.npy',allow_pickle=False))
        datasets[(rx,method)]=(rows.copy(),y.copy())
        counts,risk,records=pair_metrics(z,y);risks[(rx,method)]=risk
        meta=dict(receiver=rx,fold=fold,hardware_family=reg['hardware_families'][rx],method=method)
        per_rx.append(dict(**meta,packets=2000,correct=int(np.trace(counts)),accuracy=float(np.trace(counts)/2000),worst_tx_recall=float(np.diag(risk).min()),errors=int(2000-np.trace(counts))))
        pairs.extend(dict(**meta,**r) for r in records)
        for i in range(10):
            per_tx.append(dict(**meta,tx=i+31,packets=200,correct=int(counts[i,i]),recall=float(risk[i,i])))
            for j in range(10):conf.append(dict(**meta,tx_i=i+31,tx_j=j+31,count=int(counts[i,j]),conditional_probability=float(risk[i,j])))
        provenance.append(dict(path=str(path.relative_to(history)),sha256=asset['sha256'],kind='prediction_npz'))
    for rx in reg['source_receivers']:
        a,b=datasets[(rx,'full_cis')],datasets[(rx,'strict_cis')]
        assert all(np.array_equal(x,y) for x,y in zip(a,b)),rx
    # Every archived RX enters exactly one independently fitted CNN per method.
    assert len(risks)==20
    lookup={(r['receiver'],r['method'],r['i'],r['j']):r for r in pairs}
    changes=[]
    fields=['confusion_risk','pair_order_violation','raw_margin_mean','normalized_margin_mean','normalized_margin_q05']
    for rx in reg['source_receivers']:
        for i in range(10):
            for j in range(10):
                if i==j:continue
                a=lookup[rx,'full_cis',i,j];b=lookup[rx,'strict_cis',i,j]
                row={k:b[k] for k in ('receiver','fold','hardware_family','i','j','tx_i','tx_j')}
                row.update({'delta_'+k:b[k]-a[k] for k in fields});changes.append(row)
    replication=[]
    for i in range(10):
        for j in range(10):
            if i==j:continue
            rows=[r for r in changes if (r['i'],r['j'])==(i,j)]
            harmful=[r for r in rows if r['delta_confusion_risk']>0]
            replication.append(dict(tx_i=i+31,tx_j=j+31,harmful_receivers=len(harmful),harmful_fold0=sum(r['fold']==0 for r in harmful),harmful_fold1=sum(r['fold']==1 for r in harmful),harmful_non_rtl=sum(r['hardware_family']!='RTL-SDR' for r in harmful),mean_delta_confusion=float(np.mean([r['delta_confusion_risk'] for r in rows])),both_folds_harm=sorted(set(r['fold'] for r in harmful))==[0,1]))
    unordered=[]
    for rx,method in risks:
        for i in range(10):
            for j in range(i+1,10):
                a=lookup[rx,method,i,j];b=lookup[rx,method,j,i]
                unordered.append(dict(receiver=rx,fold=a['fold'],hardware_family=a['hardware_family'],method=method,tx_a=i+31,tx_b=j+31,balanced_confusion_risk=(a['confusion_risk']+b['confusion_risk'])/2,min_direction_mean_normalized_margin=min(a['normalized_margin_mean'],b['normalized_margin_mean'])))
    worst=[]
    # RX-balanced summaries; do not treat margins from different scales as equivalent.
    for method in reg['methods']:
        groupings={'all':reg['source_receivers'],**{f'fold{f}':rs for f,rs in enumerate(reg['source_receiver_folds'])},**{family:[rx for rx in reg['source_receivers'] if reg['hardware_families'][rx]==family] for family in set(reg['hardware_families'].values())}}
        for group,rxs in sorted(groupings.items()):
            for i in range(10):
                for j in range(10):
                    if i==j:continue
                    rr=[lookup[rx,method,i,j] for rx in rxs]
                    worst.append(dict(method=method,group=group,receivers=len(rxs),tx_i=i+31,tx_j=j+31,mean_confusion=float(np.mean([r['confusion_risk'] for r in rr])),max_rx_confusion=max(r['confusion_risk'] for r in rr),min_rx_mean_raw_margin=min(r['raw_margin_mean'] for r in rr),min_rx_mean_normalized_margin=min(r['normalized_margin_mean'] for r in rr)))
    profiles=[]; predictions=[]
    for method in reg['methods']:
        for fold,rxs in enumerate(reg['source_receiver_folds']):
            for probe in rxs:
                for control in ('same_cnn_loro','different_family_reference','cross_cnn_transfer'):
                    references=[r for r in (reg['source_receiver_folds'][1-fold] if control=='cross_cnn_transfer' else rxs) if r!=probe]
                    if control=='different_family_reference': references=[r for r in references if reg['hardware_families'][r]!=reg['hardware_families'][probe]]
                    if not references:continue
                    pp=predictors([risks[r,method] for r in references]);observed=offdiag(risks[probe,method])
                    result={name:evaluate_profile(scores,observed,reg['predictability']['top_k_pairs']) for name,scores in pp.items()}
                    factor=result['source_class_and_destination_factorized'][1]
                    for name,scores in pp.items():
                        rho,capture=result[name]
                        meta=dict(method=method,fold=fold,probe_receiver=probe,hardware_family=reg['hardware_families'][probe],control=control,predictor=name,reference_count=len(references),reference_receivers=';'.join(references))
                        profiles.append(dict(**meta,spearman=rho,top10_error_capture=capture,gain_over_uniform=None if capture is None else capture-10/90,gain_over_factorized=None if capture is None or factor is None else capture-factor,probe_error_count=int(observed.sum()*200+0.5)))
                        edge=0
                        for i in range(10):
                            for j in range(10):
                                if i==j:continue
                                predictions.append(dict(**meta,tx_i=i+31,tx_j=j+31,predicted_risk=float(scores[edge]),observed_risk=float(observed[edge])));edge+=1
    cells=[]
    for method in reg['methods']:
        for fold in (0,1):
            rows=[r for r in profiles if r['method']==method and r['fold']==fold and r['control']=='same_cnn_loro' and r['predictor']=='pairwise_mean']
            cells.append(dict(method=method,fold=fold,receivers=len(rows),defined_receivers=sum(r['spearman'] is not None for r in rows),median_spearman=median([r['spearman'] for r in rows]),median_gain_over_uniform=median([r['gain_over_uniform'] for r in rows]),median_gain_over_factorized=median([r['gain_over_factorized'] for r in rows])))
    accuracies={m:float(np.mean([r['accuracy'] for r in per_rx if r['method']==m])) for m in reg['methods']}
    structural=accuracies['strict_cis']>accuracies['full_cis'] and any(r['both_folds_harm'] for r in replication)
    predictive=all(r['median_spearman'] is not None and r['median_spearman']>0 and r['median_gain_over_uniform'] is not None and r['median_gain_over_uniform']>0 for r in cells)
    incremental=all(r['median_gain_over_factorized'] is not None and r['median_gain_over_factorized']>0 for r in cells)
    decision=dict(status='LIMITED_SUPPORT_CONDITIONAL_BASELINE_POC_ONLY' if structural and predictive else 'STOP_CANDIDATE_SCREEN_FAILED',repeated_structure_support=structural,predictive_support=predictive,mechanism_specific_support=incremental,mean_accuracy=accuracies,mean_accuracy_delta=accuracies['strict_cis']-accuracies['full_cis'],harmful_edges_both_folds=sum(r['both_folds_harm'] for r in replication),predictability_cells=cells,strong_method_sufficiency='UNKNOWN: no archived strong-method scores; literature coverage is not empirical sufficiency',seed_replication='UNAVAILABLE: one seed',new_training_runs=0,new_inference_runs=0,downloaded_dataset_bytes=0,os_u_or_development_or_locked_data_reads=0,next_training_authorized=False)
    output.mkdir(parents=True)
    for name,rows in [('input_provenance',provenance),('per_RX',per_rx),('per_RX_TX',per_tx),('confusion_counts',conf),('directed_pairs',pairs),('unordered_pairs',unordered),('strict_full_pair_changes',changes),('replicated_harmful_edges',replication),('worst_RX_pairs',worst),('risk_prediction_evaluation',profiles),('risk_prediction_vectors',predictions),('predictability_cells',cells)]:write_csv(output/(name+'.csv'),rows)
    write_json(output/'decision.json',decision)
    write_json(output/'execution.json',dict(elapsed_seconds=time.perf_counter()-start,python=sys.version,numpy=np.__version__,prediction_assets=20,expert_predictions=40000,distinct_rx_packets=20000,full_strict_exact_rows_labels_pairing_verified=True,source_fit_eval_rx_disjoint_verified=True,all_frozen_input_hashes_verified=True,pre_execution_seal_sha256=sha(ROOT/'configs/d0_class_pair/pre_execution_seal.json')))
    print(json.dumps(decision,indent=2));print('Output:',output)

if __name__=='__main__':main()
