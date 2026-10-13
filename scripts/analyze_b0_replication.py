#!/usr/bin/env python3
"""All three seeds, no model selection or inference; keep every fit status."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from d0_class_pair.metrics import pair_metrics
from audit_d0_class_pair import sha,write_json,write_csv
def main():
    r=json.loads((ROOT/'configs/d05_b0/registration.json').read_text());h=Path(r['history_root']);out=ROOT/'reports/d05_b0/2026-10-13/b0';out.mkdir(parents=True,exist_ok=False)
    original=json.loads((ROOT/'configs/d0_class_pair/registration.json').read_text());old={(a['fold'],a['method'],a['receiver']):a for a in original['prediction_assets']}
    runs=[];history_rows=[];per_rx=[];per_tx=[];pairs=[];confusion=[];provenance=[];risks={};labels_rows={};collapse_by_run={}
    for seed in [r['original_seed']]+r['new_seeds']:
        for fold,rxs in enumerate(r['folds']):
            for method in r['b0_methods']:
                run=f'B0_seed{seed}_CV{fold}_{method}'
                folder=h/f'outputs/shen_p2a8/g1_shen_v2/runs/CV{fold}_{method}' if seed==r['original_seed'] else ROOT/'outputs/d05_b0/b0/runs'/run
                status=json.loads((folder/'status.json').read_text());success=status.get('status')=='completed'
                if not success:
                    runs.append(dict(run_id=run,seed=seed,fold=fold,method=method,status=status['status'],collapse_flag=True,final_train_accuracy=None,final_train_loss=None,final_validation_accuracy=None,train_loss_last5_mean=None,seconds=status.get('seconds'),checkpoint_sha256=None,initial_sha256=None));collapse_by_run[seed,fold,method]=True;continue
                cfg=json.loads((folder/'config.json').read_text());hist=pd.read_csv(folder/'history.csv');assert hist.epoch.tolist()==list(range(1,21))
                assert cfg['seed']==seed and cfg['epochs']==status['final_epoch']==20
                assert cfg['train_keys']==[f'shen_{rx}_train' for rx in r['folds'][1-fold]]
                assert cfg['validation_keys']==[f'shen_{rx}_train' for rx in rxs]
                cp_sha=sha(folder/'final.pt')
                final_train=float(hist.iloc[-1].train_accuracy);collapse=final_train<=.20 or not np.isfinite(hist[['train_loss','train_accuracy']].values).all()
                collapse_by_run[seed,fold,method]=bool(collapse)
                runs.append(dict(run_id=run,seed=seed,fold=fold,method=method,status='completed',collapse_flag=bool(collapse),final_train_accuracy=final_train,final_train_loss=float(hist.iloc[-1].train_loss),final_validation_accuracy=float(hist.iloc[-1].validation_accuracy),train_loss_last5_mean=float(hist.tail(5).train_loss.mean()),seconds=status['seconds'],checkpoint_sha256=cp_sha,initial_sha256=cfg['initial_model_payload_sha256']))
                for row in hist.to_dict('records'):history_rows.append(dict(run_id=run,seed=seed,fold=fold,method=method,epoch=int(row['epoch']),train_loss=row['train_loss'],train_accuracy=row['train_accuracy'],training_seconds=row['training_seconds'],validation_accuracy=row['validation_accuracy'],validation_selected=False))
                for rx in rxs:
                    if seed==r['original_seed']:
                        asset=old[fold,method,rx];path=Path(asset['path']);assert sha(path)==asset['sha256'] and cp_sha==asset['checkpoint_sha256']
                    else:
                        path=folder/'predictions'/(rx+'.npz');manifest=json.loads((folder/'predictions/manifest.json').read_text());assert manifest['epoch']==20 and manifest['checkpoint_sha256']==cp_sha and sha(path)==manifest['assets'][rx+'.npz']
                    with np.load(path,allow_pickle=False) as f:z=f['logits'];y=f['labels'];rows=f['rows'];prob=f['probabilities']
                    assert np.array_equal(np.bincount(y,minlength=10),np.full(10,200)) and z.shape==(2000,10) and np.isfinite(z).all()
                    cache=h/f'outputs/shen_p2a8/g1_cache_v1/shen_{rx}_train'
                    assert np.array_equal(rows,np.load(cache/'rows.npy')) and np.array_equal(y,np.load(cache/'labels.npy'))
                    exp=np.exp(z-z.max(1,keepdims=True));assert np.allclose(prob,exp/exp.sum(1,keepdims=True),rtol=1e-6,atol=1e-7)
                    labels_rows[seed,rx,method]=(y.copy(),rows.copy());c,risk,pp=pair_metrics(z,y);risks[seed,rx,method]=risk
                    meta=dict(seed=seed,fold=fold,receiver=rx,hardware_family=r['hardware_families'][rx],method=method)
                    per_rx.append(dict(**meta,accuracy=float(np.trace(c)/2000),worst_tx_recall=float(np.diag(risk).min()),errors=int(2000-np.trace(c)),macro_f1=float(np.mean(2*np.diag(c)/np.maximum(c.sum(0)+c.sum(1),1)))))
                    for tx in range(10):
                        per_tx.append(dict(**meta,tx=tx+31,recall=float(risk[tx,tx]),correct=int(c[tx,tx]),packets=200))
                        for other in range(10):confusion.append(dict(**meta,tx_i=tx+31,tx_j=other+31,count=int(c[tx,other]),confusion_risk=float(risk[tx,other])))
                    pairs.extend(dict(**meta,**v) for v in pp)
                    provenance.append(dict(**meta,path=str(path),sha256=sha(path),checkpoint_sha256=cp_sha))
    for seed in [r['original_seed']]+r['new_seeds']:
        for rx in r['source_receivers']:
            if (seed,rx,'full_cis') in labels_rows and (seed,rx,'strict_cis') in labels_rows:
                assert all(np.array_equal(a,b) for a,b in zip(labels_rows[seed,rx,'full_cis'],labels_rows[seed,rx,'strict_cis']))
    summary=[];changes=[];pair_changes=[]
    for seed in [r['original_seed']]+r['new_seeds']:
        fr=[x for x in per_rx if x['seed']==seed and x['method']=='full_cis'];sr=[x for x in per_rx if x['seed']==seed and x['method']=='strict_cis']
        complete=len(fr)==len(sr)==10
        seed_cells=[]
        for fold,rxs in enumerate(r['folds']):
            for rx in rxs:
                if (seed,rx,'full_cis') not in risks or (seed,rx,'strict_cis') not in risks:continue
                f=risks[seed,rx,'full_cis'];s=risks[seed,rx,'strict_cis'];collapsed=collapse_by_run[seed,fold,'full_cis'] or collapse_by_run[seed,fold,'strict_cis']
                for tx in range(10):
                    delta=float(s[tx,tx]-f[tx,tx]);row=dict(seed=seed,fold=fold,receiver=rx,hardware_family=r['hardware_families'][rx],tx=tx+31,full_recall=float(f[tx,tx]),strict_recall=float(s[tx,tx]),delta_recall=delta,severe_drop=delta<=-.10+1e-12,paired_model_collapse=collapsed);changes.append(row);seed_cells.append(row)
                    for j in range(10):
                        if tx!=j:pair_changes.append(dict(seed=seed,fold=fold,receiver=rx,hardware_family=r['hardware_families'][rx],tx_i=tx+31,tx_j=j+31,full_confusion=float(f[tx,j]),strict_confusion=float(s[tx,j]),signed_delta=float(s[tx,j]-f[tx,j]),positive_delta=float(max(0,s[tx,j]-f[tx,j]))))
        summary.append(dict(seed=seed,complete=complete,full_mean_accuracy=float(np.mean([x['accuracy'] for x in fr])) if complete else None,strict_mean_accuracy=float(np.mean([x['accuracy'] for x in sr])) if complete else None,mean_accuracy_delta=float(np.mean([x['accuracy'] for x in sr])-np.mean([x['accuracy'] for x in fr])) if complete else None,full_worst_RX_TX_recall=min(x['worst_tx_recall'] for x in fr) if complete else None,strict_worst_RX_TX_recall=min(x['worst_tx_recall'] for x in sr) if complete else None,serious_cells=sum(x['severe_drop'] for x in seed_cells),declining_cells=sum(x['delta_recall']<0 for x in seed_cells),worst_recall_delta=min(x['delta_recall'] for x in seed_cells) if seed_cells else None,collapse_models=sum(v for (seed0,_,_),v in collapse_by_run.items() if seed0==seed)))
    repetition=[]
    for fold,rxs in enumerate(r['folds']):
        for rx in rxs:
            for tx in range(31,41):
                cells=[x for x in changes if x['receiver']==rx and x['tx']==tx];bad=[x for x in cells if x['severe_drop']];valid_bad=[x for x in bad if not x['paired_model_collapse']]
                repetition.append(dict(receiver=rx,fold=fold,hardware_family=r['hardware_families'][rx],tx=tx,seeds_observed=len(cells),severe_seeds=len(bad),noncollapsed_severe_seeds=len(valid_bad),severe_seed_ids=';'.join(str(x['seed']) for x in bad),mean_delta_recall=float(np.mean([x['delta_recall'] for x in cells])) if cells else None,min_delta_recall=min(x['delta_recall'] for x in cells) if cells else None,max_delta_recall=max(x['delta_recall'] for x in cells) if cells else None))
    all_complete=len(runs)==12 and all(s['complete'] for s in summary) and all(x['status']=='completed' for x in runs)
    repeated=[x for x in repetition if x['noncollapsed_severe_seeds']>=2]
    support=all_complete and all(s['mean_accuracy_delta']>0 for s in summary) and bool(repeated) and len({x['receiver'] for x in repeated})>=2
    decision=dict(status='REPLICATED_LOCAL_DAMAGE_B1_REVIEW_ONLY' if support else 'STOP_OR_REFRAME_SEED_STABILITY',screen_passed=support,all_three_seeds_complete=all_complete,repeated_severe_RX_TX_cells=len(repeated),repeated_severe_physical_RX=len({x['receiver'] for x in repeated}),repeated_severe_folds=sorted({x['fold'] for x in repeated}),seed_summaries=summary,flagged_collapse_models=sum(x['collapse_flag'] for x in runs),b1_training_authorized=False,b2_training_authorized=False,new_algorithm_authorized=False)
    for name,records in [('optimization_status',runs),('training_history',history_rows),('per_RX',per_rx),('per_RX_TX',per_tx),('confusion_counts',confusion),('directed_pairs',pairs),('recall_changes',changes),('all_pair_changes',pair_changes),('cross_seed_repetition',repetition),('seed_summary',summary),('input_provenance',provenance)]:
        if records:write_csv(out/(name+'.csv'),records)
    write_json(out/'decision.json',decision)
    write_json(out/'execution.json',json.loads((ROOT/'outputs/d05_b0/b0/execution.json').read_text()))
    print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
