#!/usr/bin/env python3
"""CPU-only complete horizon pairing and C1/C2 reliability analysis."""
import hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];PUB=ROOT/'reports/receiver_r2/r2_a';OUT=ROOT/'outputs/receiver_r2'


def write(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def corr(a,b):
    if len(a)<2 or len(set(a))<2 or len(set(b))<2:return None
    return float(np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1])


def main():
    reg=json.loads((ROOT/'configs/receiver_r2/registration.json').read_text());old=Path(reg['historical_worktree']);previous=old/'reports/receiver_r01/2026-10-13'
    h100=pd.read_csv(previous/'per_seed.csv');h100=h100[(h100.dataset=='wisig')&(h100.backbone=='official')&h100.method.isin(['ce','mldg_rx_date'])].copy();h100['horizon']=100
    h200=pd.read_csv(PUB/'per_seed200.csv');allresults=pd.concat([h100,h200],ignore_index=True,sort=False);allresults.to_csv(PUB/'per_seed100_200.csv',index=False)
    metrics=['accuracy','macro_f1','RX_accuracy_q10','mean_RX_worst_TX_recall'];pairing=[];aggregation=[]
    for (method,seed,role),g in allresults.groupby(['method','seed','role']):
        a=g[g.horizon==100].iloc[0];b=g[g.horizon==200].iloc[0]
        pairing.append(dict(method=method,seed=int(seed),role=role,**{f'{m}_100':float(a[m]) for m in metrics},**{f'{m}_200':float(b[m]) for m in metrics},**{f'{m}_delta_pp':float((b[m]-a[m])*100) for m in metrics}))
    pd.DataFrame(pairing).to_csv(PUB/'paired_horizon_changes.csv',index=False)
    for (horizon,method,role),g in allresults.groupby(['horizon','method','role']):
        aggregation.append(dict(horizon=int(horizon),method=method,role=role,seeds=len(g),**{m+'_mean':float(g[m].mean()) for m in metrics},accuracy_seed_SD_pp=float(g.accuracy.std(ddof=1)*100)))
    aggregate=pd.DataFrame(aggregation);aggregate.to_csv(PUB/'horizon_summary.csv',index=False)
    gains=[]
    for (horizon,seed,role),g in allresults.groupby(['horizon','seed','role']):
        a=g[g.method=='ce'].iloc[0];b=g[g.method=='mldg_rx_date'].iloc[0]
        gains.append(dict(horizon=int(horizon),seed=int(seed),role=role,**{m+'_dateMLDG_minus_CE_pp':float((b[m]-a[m])*100) for m in metrics}))
    gain=pd.DataFrame(gains);gain.to_csv(PUB/'paired_method_gains.csv',index=False)
    source=[];history=[];updates=[];risks=[]
    for job in reg['jobs']:
        run=OUT/job['run_id'];sel=json.loads((run/'selection.json').read_text());s=json.loads((run/'status.json').read_text())
        common={k:job[k] for k in ['run_id','dataset','method','seed','fold','stage','epochs']}
        source.append(dict(**common,selected_epoch=sel['epoch'],source_validation_CE=sel['source_validation']['CE'],source_validation_macro_f1=sel['source_validation']['macro_f1'],source_train_CE=sel['frozen_train_scores']['CE'],source_train_accuracy=sel['frozen_train_scores']['accuracy'],peak_memory_MiB=s['peak_allocated_bytes']/2**20,steps=s['steps'],effective_packet_gradient_uses=s['cumulative_effective_packet_gradient_uses'],cumulative_effective_delta_L2_sum=s['cumulative_effective_delta_L2_sum']))
        for file,container in [('history.csv',history),('updates.csv',updates),('risks.csv',risks)]:
            if (run/file).exists():
                frame=pd.read_csv(run/file)
                for k,v in common.items():frame[k]=v
                frame['phase']=np.where(frame.epoch<=job['epochs']/3,'early',np.where(frame.epoch<=2*job['epochs']/3,'middle','late'))
                container.append(frame)
    pd.DataFrame(source).to_csv(PUB/'source_selection_and_cost.csv',index=False)
    hist=pd.concat(history,ignore_index=True);upd=pd.concat(updates,ignore_index=True);risk=pd.concat(risks,ignore_index=True)
    hist.to_csv(PUB/'learning_curves.csv',index=False);upd.to_csv(PUB/'actual_update_statistics.csv',index=False);risk.to_csv(PUB/'all_finite_risks.csv',index=False)
    primary=risk[(risk.tx==-999)&(risk['mode']=='eval')&(risk.replicate==0)]
    keys=['run_id','dataset','method','fold','epoch','step','scale','rx','phase'];paired=primary[primary.pool=='C1'].merge(primary[primary.pool=='C2'],on=keys,suffixes=('_C1','_C2'))
    paired.to_csv(PUB/'C1_C2_RX_pairs.csv',index=False);reliability=[]
    for key,g in paired.groupby(['dataset','method','scale','phase']):
        confident=(~g.near_zero_C1)&(~g.near_zero_C2);detections=g.significant_harm_C1;hits=g.significant_harm_C2
        ranks=[]
        for _,step in g.groupby(['run_id','epoch']):
            val=corr(step.delta_C1,step.delta_C2)
            if val is not None:ranks.append(val)
        reliability.append(dict(dataset=key[0],method=key[1],scale=key[2],phase=key[3],RX_observations=len(g),measured_steps=len(g[['run_id','epoch']].drop_duplicates()),
            sign_agreement=float((np.sign(g.delta_C1)==np.sign(g.delta_C2)).mean()),C1_C2_pooled_rank=corr(g.delta_C1,g.delta_C2),within_step_RX_rank_mean=float(np.mean(ranks)) if ranks else None,
            confident_both_n=int(confident.sum()),confident_sign_agreement=float((np.sign(g[confident].delta_C1)==np.sign(g[confident].delta_C2)).mean()) if confident.any() else None,
            C1_harm_detections=int(detections.sum()),C2_harm_confirmations=int((detections&hits).sum()),C1_harm_precision=float(hits[detections].mean()) if detections.any() else None,
            C2_harm_recall=float(detections[hits].mean()) if hits.any() else None,C1_near_zero_rate=float(g.near_zero_C1.mean()),C2_near_zero_rate=float(g.near_zero_C2.mean()),
            C1_positive_delta_rate=float((g.delta_C1>0).mean()),C2_positive_delta_rate=float((g.delta_C2>0).mean()),median_C1_abs_delta_over_SE=float(np.median(np.abs(g.delta_C1)/(g.paired_delta_SE_C1+1e-12)))))
    pd.DataFrame(reliability).to_csv(PUB/'risk_reliability_by_phase_scale.csv',index=False)
    train=risk[(risk.tx==-999)&(risk['mode']=='train')&(risk.pool=='C1')].groupby(keys).agg(train_delta_mean=('delta','mean'),dropout_delta_SD=('delta','std')).reset_index()
    modepairs=primary[primary.pool=='C1'].merge(train,on=keys)
    modepairs['eval_train_sign_agree']=np.sign(modepairs.delta)==np.sign(modepairs.train_delta_mean)
    modepairs.to_csv(PUB/'mode_dropout_comparison.csv',index=False)
    # Update curves describe sampled steps; complete short-fit sum is in status.json.
    upd.groupby(['dataset','method','stage','phase']).agg(sampled_steps=('step','count'),candidate_delta_L2_mean=('candidate_delta_L2','mean'),preclip_gradient_norm_mean=('gradient_norm_preclip','mean'),clip_rate=('clip_applied','mean'),scale_mean=('scale','mean')).reset_index().to_csv(PUB/'update_behavior_by_phase.csv',index=False)
    # Audit review claims with their exact denominator; never modify old evidence.
    oldsource=pd.read_csv(previous/'source_validation_vs_development.csv');of=oldsource[(oldsource.dataset=='wisig')&(oldsource.backbone=='official')].drop_duplicates('run_id');ordinary=of[of.method!='swad']
    probe=pd.read_csv(previous/'update_diagnostics.csv');aud=[]
    for (stage,method),g in probe[probe.dataset=='shen'].groupby(['stage','method']):
        u=g.drop_duplicates(['run_id','epoch','step']);aud.append(dict(stage=stage,method=method,unique_probe_steps=len(u),artificial_SGD_training_side_positive_rate=float((u.meta_train_inner_delta>0).mean())))
    pd.DataFrame(aud).to_csv(PUB/'historical_probe_audit.csv',index=False)
    write(PUB/'historical_review_audit.json',dict(official_total_models=len(of),ordinary_nonSWAD_models=len(ordinary),ordinary_selected_epoch99_or100=int(ordinary.selected_epoch.isin([99,100]).sum()),review_denominator18_includes_SWAD=True))
    # Standalone scientific figures, every seed and both horizons visible.
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for col,method in enumerate(['ce','mldg_rx_date']):
        for seed in reg['formal_seeds']:
            g=hist[(hist.stage=='horizon')&(hist.method==method)&(hist.seed==seed)]
            axes[0,col].plot(g.epoch,g.source_val_CE,label=str(seed));axes[1,col].plot(g.epoch,g.source_val_macro_f1*100,label=str(seed))
        for ax in axes[:,col]:ax.axvline(100,color='black',linestyle='--',alpha=.5);ax.legend(fontsize=8);ax.set_xlabel('Epoch')
        axes[0,col].set_title(method);axes[0,col].set_ylabel('Source validation CE');axes[1,col].set_ylabel('Source validation Macro-F1 (%)')
    fig.tight_layout();fig.savefig(PUB/'convergence_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ds,marker in [('wisig','o'),('shen','x')]:
        for method,g in paired[(paired.scale==1)&(paired.dataset==ds)].groupby('method'):
            axes[0].scatter(g.delta_C1,g.delta_C2,alpha=.55,s=20,marker=marker,label=ds+' '+method)
            axes[1].scatter(g.epoch,np.abs(g.delta_C1)/(g.paired_delta_SE_C1+1e-12),alpha=.5,s=20,marker=marker)
    axes[0].axhline(0,color='grey');axes[0].axvline(0,color='grey');axes[0].set_xlabel('C1 actual Adam delta CE');axes[0].set_ylabel('C2 actual Adam delta CE');axes[0].legend(fontsize=8)
    axes[1].axhline(2,color='grey',linestyle='--');axes[1].set_xlabel('Short-fit epoch');axes[1].set_ylabel('|C1 delta| / paired SE');fig.tight_layout();fig.savefig(PUB/'risk_remeasurement.png',dpi=160);plt.close(fig)
    write(PUB/'analysis_environment.json',dict(python=sys.version,pandas=pd.__version__,numpy=np.__version__,matplotlib=matplotlib.__version__,CPU_only_analysis=True))
    print(aggregate[aggregate.role=='internal_unseen_date'].to_string(index=False));print(pd.DataFrame(reliability).query('scale==1').to_string(index=False));print('sources',pd.DataFrame(source).query("stage=='horizon'").to_string(index=False))

if __name__=='__main__':main()
