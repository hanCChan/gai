#!/usr/bin/env python3
"""Frozen exploratory cross-receiver/date decision transfer; CPU only."""
import itertools,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from receiver_r2.controller import choose_scale


def rankcorr(x,y):
    if len(x)<3 or len(set(x))<2 or len(set(y))<2:return np.nan
    return float(np.corrcoef(pd.Series(x).rank(),pd.Series(y).rank())[0,1])


def auc(scores,labels):
    labels=np.asarray(labels,bool);p=int(labels.sum());n=len(labels)-p
    if not p or not n:return np.nan
    ranks=pd.Series(scores).rank().to_numpy();return float((ranks[labels].sum()-p*(p+1)/2)/(p*n))


def main():
    pub=ROOT/'reports/receiver_c0';frame=pd.read_csv(pub/'snapshot_RX_date_risks.csv');choices=[];predictions=[]
    for (run,epoch),g in frame.groupby(['run_id','epoch']):
        common={k:g.iloc[0][k] for k in ['run_id','dataset','method','fold','epoch','phase']};rxs=sorted(g.rx.unique());split_n=len(rxs)//2;cases=[]
        cases.append(('same_RX_pool',rxs,rxs,-999,-999))
        for rx in rxs:cases.append(('same_RX_individual',[rx],[rx],-999,-999))
        for left in itertools.combinations(rxs,split_n):
            right=[r for r in rxs if r not in left];cases.append(('cross_RX',left,right,-999,-999))
        if common['dataset']=='wisig':
            for d1,d2 in [(0,1),(1,0)]:
                cases.append(('cross_date_same_RX',rxs,rxs,d1,d2))
                for left in itertools.combinations(rxs,split_n):cases.append(('cross_RX_and_date',left,[r for r in rxs if r not in left],d1,d2))
        for kind,left,right,d1,d2 in cases:
            src=g[(g.pool=='C1')&g.rx.isin(left)&(g.date==d1)];target=g[(g.pool=='C2')&g.rx.isin(right)&(g.date==d2)]
            assert len(src)==len(left)*4 and len(target)==len(right)*4
            inputs={s:[dict(delta=r.delta,paired_SE=r.paired_SE,resolution=r.resolution,n=int(r.n)) for r in sub.itertuples()] for s,sub in src.groupby('scale') if s>0}
            full=target[target.scale==1];sfull=src[src.scale==1];mu=float(full.delta.mean());noise=float((2*full.paired_SE+full.resolution).mean());source_mu=float(sfull.delta.mean());source_noise=float((2*sfull.paired_SE+sfull.resolution).mean())
            source_id=','.join(map(str,left));target_id=','.join(map(str,right));base=dict(**common,kind=kind,decision_RX=source_id,target_RX=target_id,decision_date=d1,target_date=d2,decision_n=int(sfull.n.sum()),target_n=int(full.n.sum()),decision_RX_n=len(left),target_RX_n=len(right),source_full_delta=source_mu,source_full_noise=source_noise,target_full_delta=mu,target_full_noise=noise,source_full_harm=source_mu>source_noise,target_full_harm=mu>noise,sign_agree=np.sign(source_mu)==np.sign(mu),near_zero_source=abs(source_mu)<=source_noise,near_zero_target=abs(mu)<=noise)
            predictions.append(base)
            for policy,scale in [('always_full',1.),('fixed_half',.5),('mean_backtracking',choose_scale(inputs,False)),('multi_RX_guard',choose_scale(inputs,True))]:
                q=target[target.scale==scale];delta=float(q.delta.mean());margin=float((2*q.paired_SE+q.resolution).mean());worst=float(q.delta.max())
                choices.append(dict(**base,policy=policy,chosen_scale=scale,target_chosen_mean_delta=delta,target_chosen_worst_RX_delta=worst,target_chosen_noise=margin,
                    false_accept_raw=scale>0 and delta>0,false_accept_above_noise=scale>0 and delta>margin,
                    false_reject_beneficial=scale==0 and mu< -noise,complete_reject=scale==0,accepted=scale>0,
                    target_any_RX_harm=bool(np.any(q.delta>2*q.paired_SE+q.resolution))))
    pred=pd.DataFrame(predictions);choice=pd.DataFrame(choices);pred.to_csv(pub/'transfer_predictions.csv',index=False);choice.to_csv(pub/'transfer_choices.csv',index=False)
    summaries=[]
    for key,g in pred.groupby(['dataset','method','kind']):
        labels=g.target_full_harm;positive=g.source_full_harm;tp=int((positive&labels).sum());fp=int((positive&~labels).sum());fn=int((~positive&labels).sum());tn=int((~positive&~labels).sum());conf=(~g.near_zero_source)&(~g.near_zero_target)
        summaries.append(dict(dataset=key[0],method=key[1],kind=key[2],split_observations=len(g),unique_snapshots=len(g[['run_id','epoch']].drop_duplicates()),unique_runs=g.run_id.nunique(),source_harm_n=int(positive.sum()),target_harm_n=int(labels.sum()),target_harm_base_rate=float(labels.mean()),sign_agreement=float(g.sign_agree.mean()),confident_both_n=int(conf.sum()),confident_sign_agreement=float(g[conf].sign_agree.mean()) if conf.any() else np.nan,
            source_near_zero_rate=float(g.near_zero_source.mean()),harm_precision=tp/(tp+fp) if tp+fp else np.nan,harm_recall=tp/(tp+fn) if tp+fn else np.nan,harm_specificity=tn/(tn+fp) if tn+fp else np.nan,balanced_accuracy=((tp/(tp+fn))+(tn/(tn+fp)))/2 if tp+fn and tn+fp else np.nan,
            raw_source_risk_AUC=auc(g.source_full_delta,labels),source_noise_normalized_risk_AUC=auc(g.source_full_delta/g.source_full_noise,labels),pooled_rank=rankcorr(g.source_full_delta,g.target_full_delta)))
    pd.DataFrame(summaries).to_csv(pub/'transfer_prediction_summary.csv',index=False)
    keys=['dataset','method','kind','policy'];behavior=choice.groupby(keys).agg(split_observations=('epoch','count'),mean_scale=('chosen_scale','mean'),complete_rejection_rate=('complete_reject','mean'),accepted_rate=('accepted','mean'),target_mean_delta=('target_chosen_mean_delta','mean'),target_worst_RX_delta=('target_chosen_worst_RX_delta','mean'),false_accept_raw_rate=('false_accept_raw','mean'),false_accept_above_noise_rate=('false_accept_above_noise','mean'),false_reject_beneficial_rate=('false_reject_beneficial','mean'),target_any_RX_harm_rate=('target_any_RX_harm','mean')).reset_index();behavior.to_csv(pub/'transfer_behavior_summary.csv',index=False)
    # Equal snapshot weight, not a confidence interval treating rotations as independent.
    choice.groupby(['dataset','method','kind','policy','run_id','epoch']).agg(mean_target_delta=('target_chosen_mean_delta','mean'),mean_target_worst_RX_delta=('target_chosen_worst_RX_delta','mean'),mean_scale=('chosen_scale','mean'),false_accept_rate=('false_accept_above_noise','mean')).reset_index().to_csv(pub/'snapshot_weighted_behavior.csv',index=False)
    choice.groupby(keys+['phase']).agg(observations=('epoch','count'),target_mean_delta=('target_chosen_mean_delta','mean'),mean_scale=('chosen_scale','mean'),false_accept_rate=('false_accept_above_noise','mean')).reset_index().to_csv(pub/'phase_behavior.csv',index=False)
    # Receiver participation is overlapping; stratification is descriptive, not independent sampling.
    strata=[]
    for key,g in pred.groupby(['dataset','method','kind']):
        for label in ['decision_RX','target_RX','decision_date','target_date','phase']:
            for val,sub in g.groupby(label):strata.append(dict(dataset=key[0],method=key[1],kind=key[2],stratum=label,value=str(val),observations=len(sub),snapshots=sub.epoch.nunique(),sign_agreement=float(sub.sign_agree.mean()),rank=rankcorr(sub.source_full_delta,sub.target_full_delta),harm_AUC=auc(sub.source_full_delta,sub.target_full_harm)))
    pd.DataFrame(strata).to_csv(pub/'transfer_strata.csv',index=False)
    dependence=[]
    for key,g in pred.groupby(['dataset','method','kind','run_id']):
        steps=g.groupby('epoch')[['source_full_delta','target_full_delta']].mean().sort_index();changes=steps.diff().dropna()
        dependence.append(dict(dataset=key[0],method=key[1],kind=key[2],run_id=key[3],snapshots=len(steps),source_lag1=float(steps.source_full_delta.autocorr()),target_lag1=float(steps.target_full_delta.autocorr()),between_step_rank=rankcorr(steps.source_full_delta,steps.target_full_delta),first_difference_rank=rankcorr(changes.source_full_delta,changes.target_full_delta),first_difference_sign_agreement=float((np.sign(changes.source_full_delta)==np.sign(changes.target_full_delta)).mean())))
    pd.DataFrame(dependence).to_csv(pub/'trajectory_dependence.csv',index=False)
    delta=choice.pivot(index=['run_id','epoch','kind','decision_RX','target_RX','decision_date','target_date'],columns='policy',values='chosen_scale').reset_index();delta['guard_differs']=delta.mean_backtracking!=delta.multi_RX_guard;delta.to_csv(pub/'guard_vs_mean_choices.csv',index=False)
    print(pd.DataFrame(summaries).query("kind=='cross_RX'").to_string(index=False));print(behavior.query("kind=='cross_RX'").to_string(index=False))

if __name__=='__main__':main()
