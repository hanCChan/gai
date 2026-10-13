#!/usr/bin/env python3
"""Frozen C1-only controller choices on unmodified-Adam R2-A snapshots."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from receiver_r2.controller import choose_scale


def main():
    pub=ROOT/'reports/receiver_r2/r2_a';risk=pd.read_csv(pub/'all_finite_risks.csv');upd=pd.read_csv(pub/'actual_update_statistics.csv')
    primary=risk[(risk.tx==-999)&(risk['mode']=='eval')&(risk.replicate==0)];rows=[]
    keys=['run_id','dataset','method','fold','epoch','step','phase']
    for key,g in primary.groupby(keys):
        a=g[g.pool=='C1'];inputs={s:[dict(delta=r.delta,paired_SE=r.paired_delta_SE,resolution=r.resolution,n=r.n) for r in h.itertuples()] for s,h in a.groupby('scale') if s>0}
        norm=float(upd[(upd.run_id==key[0])&(upd.step==key[5])].candidate_delta_L2.iloc[0])
        for policy,multi in [('mean_source_backtracking',False),('multi_RX_guard',True)]:
            scale=choose_scale(inputs,multi_rx=multi);b=g[(g.pool=='C2')&(g.scale==scale)]
            rows.append(dict(**dict(zip(keys,key)),policy=policy,chosen_scale=scale,accepted=scale>0,full_accept=scale==1,complete_reject=scale==0,
                sampled_effective_delta_L2_proxy=scale*norm,C2_mean_delta=float(b.delta.mean()),C2_any_positive=bool((b.delta>0).any()),C2_any_above_noise=bool(b.significant_harm.any()),
                kind='offline_on_standard_Adam_trajectory_not_controlled_fit'))
    frame=pd.DataFrame(rows);frame.to_csv(pub/'offline_controller_choices.csv',index=False)
    summary=frame.groupby(['dataset','method','policy','phase']).agg(sampled_steps=('step','count'),acceptance_rate=('accepted','mean'),full_acceptance_rate=('full_accept','mean'),complete_rejection_rate=('complete_reject','mean'),mean_scale=('chosen_scale','mean'),sampled_effective_delta_L2_proxy_sum=('sampled_effective_delta_L2_proxy','sum'),C2_mean_delta=('C2_mean_delta','mean'),C2_repeated_harm_rate=('C2_any_above_noise','mean')).reset_index();summary.to_csv(pub/'offline_controller_behavior.csv',index=False)
    compare=frame.pivot_table(index=keys,columns='policy',values='chosen_scale').reset_index();compare['choices_differ']=compare.mean_source_backtracking!=compare.multi_RX_guard;compare.to_csv(pub/'offline_guard_difference.csv',index=False)
    print(summary.to_string(index=False));print('guard_differences',int(compare.choices_differ.sum()),'of',len(compare))

if __name__=='__main__':main()
