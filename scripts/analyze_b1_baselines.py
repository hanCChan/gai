#!/usr/bin/env python3
"""All approved jobs and all formal class risks, without selection or inference."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from d0_class_pair.metrics import pair_metrics
from audit_d0_class_pair import write_json as dump,write_csv as csv_write,sha

def main():
    r=json.loads((ROOT/'configs/b1_baselines/registration.json').read_text());p=ROOT/'outputs/b1_baselines';execution=json.loads((p/'execution.json').read_text())
    if execution['state'] not in ('completed','completed_with_retained_failures'):raise ValueError('Wait for complete fixed matrix')
    dst=ROOT/'reports/b1_baselines/2026-10-13';dst.mkdir(parents=True,exist_ok=False)
    jobs=[];history=[];per_rx=[];per_tx=[];pairs=[];cm_rows=[];provenance=[];source_profiles=[]
    for stage,key in [('search','search_jobs'),('formal','formal_jobs')]:
        for job in r['matrix'][key]:
            folder=p/stage/'runs'/job['run_id'];status=json.loads((folder/'status.json').read_text())
            meta=dict(stage=stage,run_id=job['run_id'],fold=job['fold'],seed=job['seed'],method=job['method'])
            if status['status']!='completed':
                jobs.append(dict(**meta,status=status['status'],epochs=status.get('epoch'),lr=job.get('lr'),base_parameters=None,aux_parameters=None,train_accuracy=None,loss=None,seconds=status.get('seconds'),optimizer_steps=status.get('optimizer_steps'),checkpoint_sha256=None,initial_sha256=None,schedule_sha256=None,selection_sha256=None));continue
            config=json.loads((folder/'config.json').read_text());hist=pd.read_csv(folder/'history.csv');metrics=json.loads((folder/'metrics.json').read_text());cp_sha=sha(folder/'final.pt')
            assert status['final_epoch']==job['epochs'] and cp_sha==status['checkpoint_sha256'] and hist.epoch.tolist()==list(range(1,job['epochs']+1))
            jobs.append(dict(**meta,status='completed',epochs=job['epochs'],lr=config['lr'],base_parameters=config['base_parameters'],aux_parameters=config['aux_parameters'],train_accuracy=float(hist.iloc[-1].train_accuracy),loss=float(hist.iloc[-1].classification_CE),seconds=status['seconds'],optimizer_steps=status['optimizer_steps'],checkpoint_sha256=cp_sha,initial_sha256=config['initial_base_payload_sha256'],schedule_sha256=config['schedule_sha256'],selection_sha256=config.get('selection_sha256')))
            for row in hist.to_dict('records'):
                history.append(dict(**meta,epoch=int(row['epoch']),classification_CE=float(row['classification_CE']),objective_loss=float(row['objective_loss']),auxiliary_loss=float(row['auxiliary_loss']),train_accuracy=float(row['train_accuracy']),optimizer_steps=int(row['optimizer_steps']),training_seconds=float(row['training_seconds']),q_min=None if pd.isna(row['q_min']) else float(row['q_min']),q_max=None if pd.isna(row['q_max']) else float(row['q_max']),q_entropy=None if pd.isna(row['q_entropy']) else float(row['q_entropy'])))
            if stage=='search':
                source_profiles.append(dict(**meta,lr=config['lr'],validation_keys=';'.join(config['validation_keys']),macro_f1=metrics['macro_f1'],accuracy=metrics['accuracy'],worst_tx_recall=min(x['worst_tx_recall'] for x in metrics['per_RX']),macro_f1_numerator=metrics['macro_f1_fraction']['numerator'],macro_f1_denominator=metrics['macro_f1_fraction']['denominator']));continue
            assert config['selection_sha256']==sha(p/'selection.frozen.json')
            for key in config['evaluation_keys']:
                rx=key.removeprefix('shen_').removesuffix('_train');file=folder/'predictions'/(rx+'.npz');assert sha(file)==metrics['assets'][file.name]
                with np.load(file,allow_pickle=False) as a:logits=a['logits'];y=a['labels'];rows=a['rows']
                assert logits.shape==(2000,10) and np.array_equal(np.bincount(y,minlength=10),np.full(10,200))
                counts,risk,pp=pair_metrics(logits,y);m=dict(method=job['method'],seed=job['seed'],fold=job['fold'],receiver=rx,hardware_family=r['hardware_families'][rx])
                f1=float(np.mean(2*np.diag(counts)/np.maximum(counts.sum(0)+counts.sum(1),1)))
                per_rx.append(dict(**m,accuracy=float(np.trace(counts)/2000),macro_f1=f1,worst_tx_recall=float(np.diag(risk).min()),errors=int(2000-np.trace(counts))))
                for tx in range(10):
                    per_tx.append(dict(**m,tx=tx+31,recall=float(risk[tx,tx]),correct=int(counts[tx,tx]),packets=200))
                    for other in range(10):cm_rows.append(dict(**m,tx_i=tx+31,tx_j=other+31,count=int(counts[tx,other]),conditional_probability=float(risk[tx,other])))
                pairs.extend(dict(**m,**v) for v in pp)
                provenance.append(dict(**m,path=str(file),npz_sha256=sha(file),checkpoint_sha256=cp_sha,labels_sha256=hashlib.sha256(y.tobytes()).hexdigest(),rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest()))
    assert len(jobs)==32
    def summary(rows,txrows,method,group):
        return dict(method=method,group=group,receiver_seed_cells=len(rows),mean_accuracy=float(np.mean([x['accuracy'] for x in rows])),mean_RX_macro_f1=float(np.mean([x['macro_f1'] for x in rows])),minimum_RX_TX_recall=min(x['recall'] for x in txrows),mean_min_TX_recall=float(np.mean([x['worst_tx_recall'] for x in rows])))
    summaries=[]
    for method in r['methods']:
        rr=[x for x in per_rx if x['method']==method];tt=[x for x in per_tx if x['method']==method]
        if not rr:continue
        summaries.append(summary(rr,tt,method,'all'))
        for seed in (20261013,20261014):
            sr=[x for x in rr if x['seed']==seed];st=[x for x in tt if x['seed']==seed]
            if sr:summaries.append(summary(sr,st,method,f'seed{seed}'))
        for fold in (0,1):
            sr=[x for x in rr if x['fold']==fold];st=[x for x in tt if x['fold']==fold]
            if sr:summaries.append(summary(sr,st,method,f'fold{fold}'))
        for family in sorted(set(r['hardware_families'].values())):
            sr=[x for x in rr if x['hardware_family']==family];st=[x for x in tt if x['hardware_family']==family]
            if sr:summaries.append(summary(sr,st,method,family))
    recall_index={(x['method'],x['seed'],x['receiver'],x['tx']):x for x in per_tx}
    pair_index={(x['method'],x['seed'],x['receiver'],x['i'],x['j']):x for x in pairs}
    rx_index={(x['method'],x['seed'],x['receiver']):x for x in per_rx}
    changes=[];rx_changes=[];pair_changes=[]
    for method in r['methods'][1:]:
        for seed in (20261013,20261014):
            for fold,rxs in enumerate(r['folds']):
                for rx in rxs:
                    if (method,seed,rx) not in rx_index or ('ce',seed,rx) not in rx_index:continue
                    c=rx_index['ce',seed,rx];other=rx_index[method,seed,rx];meta=dict(method=method,seed=seed,fold=fold,receiver=rx,hardware_family=r['hardware_families'][rx])
                    rx_changes.append(dict(**meta,delta_accuracy=other['accuracy']-c['accuracy'],delta_macro_f1=other['macro_f1']-c['macro_f1'],delta_worst_TX_recall=other['worst_tx_recall']-c['worst_tx_recall']))
                    for tx in range(31,41):
                        a=recall_index['ce',seed,rx,tx];b=recall_index[method,seed,rx,tx];delta=b['recall']-a['recall']
                        changes.append(dict(**meta,tx=tx,CE_recall=a['recall'],method_recall=b['recall'],delta_recall=delta,severe_CE_relative_loss=delta<=-.1+1e-12))
                    for i in range(10):
                        for j in range(10):
                            if i==j:continue
                            a=pair_index['ce',seed,rx,i,j];b=pair_index[method,seed,rx,i,j]
                            pair_changes.append(dict(**meta,tx_i=i+31,tx_j=j+31,delta_confusion=b['confusion_risk']-a['confusion_risk'],delta_raw_margin_mean=b['raw_margin_mean']-a['raw_margin_mean'],delta_normalized_margin_mean=b['normalized_margin_mean']-a['normalized_margin_mean'],delta_pair_order_violation=b['pair_order_violation']-a['pair_order_violation']))
    comparison=[];summary_index={(x['method'],x['group']):x for x in summaries}
    for method in r['methods'][1:]:
        if (method,'all') not in summary_index or ('ce','all') not in summary_index:continue
        a=summary_index['ce','all'];b=summary_index[method,'all'];rows=[x for x in changes if x['method']==method]
        seed_delta={str(seed):summary_index[method,f'seed{seed}']['mean_accuracy']-summary_index['ce',f'seed{seed}']['mean_accuracy'] for seed in (20261013,20261014) if (method,f'seed{seed}') in summary_index and ('ce',f'seed{seed}') in summary_index}
        comparison.append(dict(method=method,mean_accuracy_delta=b['mean_accuracy']-a['mean_accuracy'],mean_RX_macro_f1_delta=b['mean_RX_macro_f1']-a['mean_RX_macro_f1'],minimum_RX_TX_recall_delta=b['minimum_RX_TX_recall']-a['minimum_RX_TX_recall'],mean_min_TX_recall_delta=b['mean_min_TX_recall']-a['mean_min_TX_recall'],severe_CE_relative_losses=sum(x['severe_CE_relative_loss'] for x in rows),declining_RX_TX_cells=sum(x['delta_recall']<0 for x in rows),paired_RX_TX_cells=len(rows),pooled_joint_improvement=b['mean_accuracy']>=a['mean_accuracy'] and b['minimum_RX_TX_recall']>a['minimum_RX_TX_recall'],mean_accuracy_preserved_both_seeds=len(seed_delta)==2 and all(v>=0 for v in seed_delta.values())))
    # The B0 repeated cells are descriptive history context, never training weights.
    original_repeated=pd.read_csv(ROOT/'reports/d05_b0/2026-10-13/b0/cross_seed_repetition.csv');original_repeated=original_repeated[original_repeated.noncollapsed_severe_seeds>=2]
    context=[]
    for cell in original_repeated.itertuples():
        for seed in (20261013,20261014):
            for method in r['methods']:
                key=method,seed,cell.receiver,int(cell.tx)
                if key in recall_index:context.append(dict(method=method,seed=seed,receiver=cell.receiver,fold=int(cell.fold),tx=int(cell.tx),recall=recall_index[key]['recall'],historical_B0_severe_seeds=int(cell.severe_seeds)))
    complete=all(x['status']=='completed' for x in jobs) and len(per_rx)==80 and len(per_tx)==800
    joint=[x['method'] for x in comparison if x['pooled_joint_improvement']]
    strict_joint=[x['method'] for x in comparison if x['pooled_joint_improvement'] and x['mean_accuracy_preserved_both_seeds'] and x['severe_CE_relative_losses']==0]
    decision=dict(status='B1_COMPLETE_STANDARD_BASELINES_REVIEW_REQUIRED' if complete else 'B1_INCOMPLETE_RETAIN_FAILURES',all32_complete=complete,fitting_jobs=execution['fitting_jobs_launched'],reserved_gpu_hours=execution['reserved_gpu_hours'],pooled_average_and_minimum_improvement_methods=joint,methods_with_joint_both_seed_accuracy_and_no_new_10pp_losses=strict_joint,comparison=comparison,B2_authorized=False,ArcFace_authorized=False,new_algorithm_necessity_proven=False,independent_dataset_validated=False,stop_after_B1=True)
    for name,values in [('all_job_status',jobs),('training_history',history),('source_search_metrics',source_profiles),('per_RX',per_rx),('per_RX_TX',per_tx),('directed_pairs',pairs),('confusion_counts',cm_rows),('input_provenance',provenance),('method_summaries',summaries),('paired_RX_changes',rx_changes),('paired_TX_changes',changes),('paired_pair_changes',pair_changes),('CE_comparison',comparison),('B0_repeated_cells_context',context)]:
        if values:csv_write(dst/(name+'.csv'),values)
    dump(dst/'decision.json',decision);dump(dst/'execution.json',execution)
    print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
