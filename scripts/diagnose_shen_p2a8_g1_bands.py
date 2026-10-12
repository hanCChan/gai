#!/usr/bin/env python3
"""Descriptive frozen equal-width bands; targets never select weights or frequencies."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from shen_p2a8.g1_data import cache
from report_shen_p2a import write_csv
from shen_p2a.cache import write_json

def load(key):
    folder=cache(ROOT,key);x=np.asarray(np.load(folder/'psd128.npy',mmap_mode='r'));y=np.load(folder/'labels.npy');return np.log10(np.maximum(x,1e-20)),y

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['shen','osu'],required=True);a=parser.parse_args();out=ROOT/f'outputs/shen_p2a8/g1_{a.dataset}_bands_v1';out.mkdir(parents=True,exist_ok=False)
    reg=json.loads((ROOT/'configs/shen_p2a8/g1_registration.json').read_text());metrics=[];shifts=[];source_scores=[]
    if a.dataset=='shen':
        tasks=[('10_training_RX',['shen_'+rx+'_train' for rx in reg['train_receivers']],[x['key'] for x in reg['files'] if x['role']!='train'])];classes=10
    else:
        # One source RX cannot estimate source cross-RX variance: record it as unavailable.
        tasks=[(rx,['osu_'+rx],['osu_RX1','osu_RX2']) for rx in ('RX1','RX2')];classes=25
    for direction,sourcekeys,evalkeys in tasks:
        source=[load(key) for key in sourcekeys];x=np.concatenate([v[0] for v in source]);y=np.concatenate([v[1] for v in source]);mean=x.mean(0);std=np.maximum(x.std(0),1e-8);z=(x-mean)/std
        for band in range(8):
            sl=slice(band*16,(band+1)*16);centres=np.stack([z[y==tx,sl].mean(0) for tx in range(classes)])
            between=float(2*classes/(classes-1)*np.var(centres,axis=0).sum())
            rxcentres=np.stack([np.stack([((v[0]-mean)/std)[v[1]==tx,sl].mean(0) for tx in range(classes)]) for v in source])
            rx_variance=float(np.var(rxcentres,axis=0).sum(-1).mean()) if len(source)>1 else None
            within=float(np.mean([np.var(z[y==tx,sl],axis=0).sum() for tx in range(classes)]))
            source_scores.append(dict(direction=direction,band=band,normalized_frequency_low=-.5+band/8,normalized_frequency_high=-.5+(band+1)/8,D_between=between,V_RX_given_TX=rx_variance,within_TX_packet_variance=within,discriminability_ratio=between/max(within,1e-8),candidate_stability_ratio=between/max(rx_variance,1e-8) if rx_variance is not None else None,source_domains=len(source),score_validated=False))
            for key in sourcekeys+evalkeys:
                xx,yy=load(key);zz=((xx-mean)/std)[:,sl];dist=np.square(zz[:,None,:]-centres[None,:,:]).sum(-1);pred=dist.argmin(1)
                metrics.append(dict(direction=direction,domain=key,band=band,source_resubstitution=key in sourcekeys,packets=len(yy),nearest_source_TX_centroid_accuracy=float((pred==yy).mean())))
                for tx in range(classes):
                    ix=yy==tx;target=zz[ix].mean(0)
                    shifts.append(dict(direction=direction,domain=key,band=band,raw_label=tx+(31 if a.dataset=='shen' else 1),squared_mean_shift=float(np.square(target-centres[tx]).sum()),nearest_centroid_recall=float((pred[ix]==tx).mean())))
    write_csv(out/'source_only_band_statistics.csv',source_scores);write_csv(out/'per_RX_band_centroid_accuracy.csv',metrics);write_csv(out/'per_TX_cross_RX_shift.csv',shifts)
    fig,axes=plt.subplots(1,len(tasks),figsize=(9*len(tasks),5),squeeze=False)
    for ax,(direction,sourcekeys,evalkeys) in zip(axes[0],tasks):
        scores=[r for r in source_scores if r['direction']==direction];ax.plot(range(8),[r['discriminability_ratio'] for r in scores],label='Source discriminability ratio',marker='o')
        if len(sourcekeys)>1:ax.plot(range(8),[r['V_RX_given_TX'] for r in scores],label='Source conditional RX variance',marker='s')
        ax.set_xlabel('Frozen equal-width normalized frequency band');ax.set_ylabel('Source-only statistic');ax.set_title(direction);ax.legend()
    fig.tight_layout();fig.savefig(out/'discriminability_and_source_RX_variation.png',dpi=180);plt.close(fig)
    write_json(out/'status.json',dict(status='completed',frequency_bands_frozen=8,target_used_for_weights=False,source_variance_available=a.dataset=='shen',G2_established=False,models_trained=0))
    (out/'REPORT.md').write_text('# G1 无训练频带机制统计\n\n八个等宽归一化频带，各16个PSD聚合bin，规则不依赖开发准确率。标准化与类中心只用源域，目标标签仅评价。数表同时给出辨识度、同TX跨源RX均值方差、类内packet方差、逐RX最近源类中心分类和逐TX均值移位。没有从目标挑频率、没有学习新方法。\n\nOSU每方向仅一个源RX，不能估计源跨RX稳定性方差；该字段为空而不是伪造为零。Shen多个源RX允许描述此统计，尚未证明它能预测未见环境。source分类为重代入描述，有乐观偏差；不能用它与目标下降之差直接证明新通用机制。G1需要结合固定CNN强基线和两数据集进行决策，G2仍需独立预登记。\n')
