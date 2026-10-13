#!/usr/bin/env python3
"""Apply the predeclared sharedness stop rule; never fit on held TX."""
import csv
import json
from pathlib import Path
import numpy as np
from receiver_transform_common import dump, sha, mappings, distances
ROOT=Path(__file__).resolve().parents[1]

def table(path, records):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

def main():
    out=ROOT/'outputs/receiver_transform';reg=json.loads((ROOT/'configs/receiver_transform/registration.json').read_text())
    exe=json.loads((out/'execution.json').read_text());assert exe['state']=='completed' and exe['completed_jobs']==20
    report=ROOT/'reports/receiver_transform/2026-10-13';report.mkdir(parents=True,exist_ok=False)
    rows=[];train=[]
    for fold in range(2):
        for partition in range(5):
            held=[x-31 for x in reg['held_tx_partitions'][partition]]
            fitted=[x for x in range(10) if x not in held]
            for seed in reg['seeds']:
                run=out/'runs'/f'gate_CV{fold}_TX{partition}_seed{seed}'
                status=json.loads((run/'status.json').read_text());cfg=json.loads((run/'config.json').read_text())
                assert status['state']=='completed' and status['optimizer_steps']==1200
                assert cfg['outer_rx_opened']==[] and cfg['fitting_held_tx_access'] is False
                assert cfg['held_tx_ids']==[x+31 for x in held] and not set(cfg['fitted_tx_ids'])&set(cfg['held_tx_ids'])
                assert sha(run/'features.npy')==status['features_sha256']
                train.append(dict(run=run.name,fold=fold,partition=partition,seed=seed,
                    train_accuracy=status['final_train_accuracy'],seconds=status['seconds'],
                    checkpoint_sha256=status['checkpoint_sha256'],features_sha256=status['features_sha256'],
                    initial_sha256=cfg['initial_payload_sha256'],schedule_sha256=status['schedule_sha256']))
                feature=np.load(run/'features.npy',allow_pickle=False).astype('float64')
                fitting=feature[:,fitted,:40]
                center=fitting.mean(axis=(0,1,2));scale=float(np.sqrt(np.mean((fitting-center)**2)))
                if scale<=1e-10:raise ValueError('Collapsed encoder representation')
                feature=(feature-center)/scale
                for src in range(5):
                    for dst in range(5):
                        if src==dst:continue
                        transforms=mappings(feature[src,fitted,:40],feature[dst,fitted,:40],
                            rank=reg['rank'],ridge=reg['ridge_relative'],shrink=reg['covariance_shrinkage'])
                        reference=feature[dst,:,40:].mean(1)
                        for tx in held:
                            source=feature[src,tx,40:];target=feature[dst,tx,40:]
                            for method,(a,b) in transforms.items():
                                transformed=source@a+b
                                mean,w2=distances(transformed,target)
                                query=transformed.mean(0)
                                retrieval=int(np.argmin(np.sum((reference-query)**2,axis=1)))
                                rows.append(dict(fold=fold,partition=partition,seed=seed,
                                    source_rx=cfg['source_rx_opened'][src],target_rx=cfg['source_rx_opened'][dst],
                                    tx=tx+31,method=method,mean_squared_distance=mean,gaussian_w2_squared=w2,
                                    identity_retrieval_correct=int(retrieval==tx),predicted_tx=retrieval+31))
    table(report/'held_TX_RX_pair_distances.csv',rows);table(report/'training_jobs.csv',train)
    summaries=[];strata=[]
    controls=['identity','translation','coral'];all_pass=True
    for fold in range(2):
        for seed in reg['seeds']:
            grouped={}
            for method in controls+['lowrank']:
                relevant=[x for x in rows if x['fold']==fold and x['seed']==seed and x['method']==method]
                assert len(relevant)==200
                avg={metric:float(np.mean([x[metric] for x in relevant])) for metric in
                    ['mean_squared_distance','gaussian_w2_squared','identity_retrieval_correct']}
                grouped[method]=avg;summaries.append(dict(fold=fold,seed=seed,method=method,**avg))
            criteria={};passed=True
            for metric in ['mean_squared_distance','gaussian_w2_squared']:
                best=min(controls,key=lambda m:grouped[m][metric])
                improvement=1-grouped['lowrank'][metric]/max(grouped[best][metric],1e-30)
                successes=0
                for partition in range(5):
                    means={m:float(np.mean([x[metric] for x in rows if x['fold']==fold and x['seed']==seed and x['partition']==partition and x['method']==m])) for m in controls+['lowrank']}
                    successes+=means['lowrank']<min(means[m] for m in controls)
                yes=improvement>=reg['gate']['relative_distance_reduction'] and successes>=3
                criteria[metric]=dict(best_control=best,relative_improvement=improvement,
                    improving_TX_partitions=successes,passed=bool(yes));passed=passed and yes
            best_identity=max(grouped[m]['identity_retrieval_correct'] for m in controls)
            retention=grouped['lowrank']['identity_retrieval_correct']>=best_identity-.05
            encoder=min(x['train_accuracy'] for x in train if x['fold']==fold and x['seed']==seed)>=.5
            passed=bool(passed and retention and encoder);all_pass=all_pass and passed
            strata.append(dict(fold=fold,seed=seed,criteria=criteria,
                lowrank_identity_retrieval=grouped['lowrank']['identity_retrieval_correct'],
                strongest_control_identity_retrieval=best_identity,identity_gate=bool(retention),
                encoder_gate=bool(encoder),passed=passed))
    table(report/'stratum_summary.csv',summaries)
    result=dict(status='PASS_SHAREDNESS_CONDITIONAL_POC_REQUIRED' if all_pass else 'STOP_SHAREDNESS_FAILED_NO_CLASSIFICATION_POC',
        sharedness_passed=all_pass,strict_held_TX_exclusion=True,strata=strata,
        fitting_jobs=20,optimizer_steps=24000,classification_poc_jobs=0,
        reserved_gpu_hours=exe['reserved_gpu_hours'],fitting_retries=0,
        outer_receiver_evaluation_performed=False,wisig_used_for_research=False,
        scope='Fixed rank2 linear residual mappings in one preregistered IQ encoder family; no causal hardware attribution, no general rejection of nonlinear mapping hypotheses',
        uncertainty='Directed RX pairs share receivers; TX partitions share fitted identities. Engineering gate, not independent significance test.')
    dump(report/'decision.json',result);dump(report/'execution.json',exe)
    dump(report/'pre_execution_seal.json',json.loads((out/'pre_execution_seal.json').read_text()))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
