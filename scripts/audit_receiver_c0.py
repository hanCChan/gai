#!/usr/bin/env python3
"""CPU source/index/correlation audit; no raw-IQ or target values are loaded."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from receiver_r2.data import diagnostic_data,identities
from receiver_r01.core import mldg_loss


def correlation(x,y):
    if np.std(x)<1e-12 or np.std(y)<1e-12:return None
    return float(np.corrcoef(x,y)[0,1])


def write_csv(path,rows):
    with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    torch.set_num_threads(2);reg=json.loads((ROOT/'configs/receiver_c0/registration.json').read_text());old=json.loads((ROOT/'configs/receiver_r01/registration.json').read_text());pub=ROOT/'reports/receiver_c0';posrows=[];checks=[];packet_risks=[];gradient_rows=[]
    # One data audit per dataset/fold; metadata uses only previously permitted source caches.
    for ds,fold in [('wisig',0),('shen',0),('shen',1)]:
        entry=next(e for e in reg['diagnostic_runs'] if e['job']['dataset']==ds and e['job']['fold']==fold);pools,stats=diagnostic_data(old,entry['job'])
        for a in pools:
            for b in pools:
                if a!=b:assert not identities(pools[a])&identities(pools[b])
        for rx in np.unique(pools['fit']['rx']):
            for day in np.unique(pools['fit']['day'][pools['fit']['rx']==rx]):
                for tx in range(10):
                    cell={n:p[(pools[n]['rx']==rx)&(pools[n]['day']==day)&(pools[n]['y']==tx)] for n,p in {k:v['position'] for k,v in pools.items()}.items()}
                    positions=np.concatenate(list(cell.values()));assert len(positions)==len(set(positions))
                    feats=[]
                    for name,data in pools.items():
                        mask=(data['rx']==rx)&(data['day']==day)&(data['y']==tx);x=data['x'][mask].reshape(int(mask.sum()),-1).astype(np.float64)
                        # WiSig normalized power is fixed, use I/Q means and fourth moment.
                        if ds=='wisig':
                            shaped=data['x'][mask];feat=np.stack([shaped[...,0].mean(axis=(1,2)),shaped[...,1].mean(axis=(1,2)),np.mean(x**4,axis=1)],axis=1)
                        else:feat=np.stack([x.mean(1),x.std(1),np.mean(x*x,axis=1)],axis=1)
                        feats.append(feat)
                    features=np.concatenate(feats);order=np.argsort(positions);ordered=features[order];ordered_pos=positions[order]
                    for fi in range(3):
                        rank_x=np.argsort(np.argsort(ordered[:,fi]));rank_pos=np.arange(len(order))
                        posrows.append(dict(dataset=ds,fold=fold,rx=int(rx),date=int(day),tx=tx,feature_index=fi,packets=len(order),position_min=int(ordered_pos.min()),position_max=int(ordered_pos.max()),C1_position_min=int(cell['C1'].min()),C1_position_max=int(cell['C1'].max()),C2_position_min=int(cell['C2'].min()),C2_position_max=int(cell['C2'].max()),C1_C2_original_index_boundary_gap=int(cell['C2'].min()-cell['C1'].max()),feature_lag1=correlation(ordered[:-1,fi],ordered[1:,fi]),feature_position_rank=correlation(rank_x,rank_pos),timestamp_or_session_available=False))
        checks.append(dict(dataset=ds,fold=fold,pools_disjoint=True,fit_packets=len(pools['fit']['y']),C1_packets=len(pools['C1']['y']),C2_packets=len(pools['C2']['y']),scaler_fit_only=stats is not None if ds=='shen' else True))
    # Risk-sequence dependence, descriptive block means only; policy constants unchanged.
    for entry in reg['diagnostic_runs']:
        job=entry['job'];run=Path(entry['run_path']);parts=np.load(run/'RX_partitions.npy');schedule=np.load(run/'schedule.npy');nA,nB,nC=(5,5,4) if job['dataset']=='wisig' else (2,1,2)
        for path in sorted(run.glob('actual_adam_epoch*.pt')):
            s=torch.load(path,map_location='cpu',weights_only=False);part=parts[(s['epoch']-1)*schedule.shape[1]];assert len(set(part))==len(part);A=part[:nA];B=part[nA:nA+nB];C=part[-nC:];assert not set(A)&set(B) and not (set(A)|set(B))&set(C)
            for risk in s['risk_packet_measurements']:
                if risk['mode']!='eval' or risk['replicate']!=0 or risk['scale']!=1:continue
                assert set(risk['rx'])==set(C)
                for rx in C:
                    for day in np.unique(risk['date'][risk['rx']==rx]):
                        for tx in range(10):
                            d=risk['delta'][(risk['rx']==rx)&(risk['date']==day)&(risk['tx']==tx)];assert len(d)==20;naive=float(d.std(ddof=1)/np.sqrt(20));block=float(d.reshape(5,4).mean(1).std(ddof=1)/np.sqrt(5))
                            packet_risks.append(dict(dataset=job['dataset'],method=job['method'],fold=job['fold'],epoch=s['epoch'],pool=risk['pool'],rx=int(rx),date=int(day),tx=tx,n=20,risk_delta_lag1=correlation(d[:-1],d[1:]),naive_paired_SE=naive,block4_descriptive_SE=block,block4_over_naive_SE=block/naive if naive else None,block_independence_confirmed=False))
    # Clip summary is observational and sampled: first batch of each epoch only.
    base=Path(reg['historical_worktree'])/'reports/receiver_r2/r2_a/actual_update_statistics.csv'
    with base.open() as f:
        for r in csv.DictReader(f):
            if r['stage']=='horizon':gradient_rows.append({k:r[k] for k in ['run_id','method','seed','epoch','phase','gradient_norm_preclip','gradient_norm_postclip','clip_applied','candidate_delta_L2','moment1_L2','moment2_L2']})
    write_csv(pub/'packet_index_feature_correlations.csv',posrows);write_csv(pub/'packet_risk_dependence.csv',packet_risks);write_csv(pub/'sampled_gradient_clipping.csv',gradient_rows)
    (pub/'source_pool_audit.json').write_text(json.dumps(dict(checks=checks,original_positions_are_indices_not_verified_timestamps=True,independent_sessions_proven=False,original_pregradient_Dropout_RNG_saved=False,saved_clipped_gradient_and_actual_Adam_states_available=True,formal_fits=0),indent=2)+'\n')
    print('source pool checks',len(checks),'feature cells',len(posrows),'packet risk cells',len(packet_risks),'sampled gradient rows',len(gradient_rows))

if __name__=='__main__':main()
