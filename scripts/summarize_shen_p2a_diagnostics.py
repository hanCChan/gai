#!/usr/bin/env python3
"""Join frozen P1 errors to TRAIN-fitted clusters and write a bounded explanation."""
import csv,json,sys
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shen_p2a.cache import cache_dir,write_json
from report_shen_p2a import read_csv,write_csv
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--diagnostics',type=Path,required=True);a=p.parse_args();out=a.diagnostics
    assert json.loads((out/'status.json').read_text())['status']=='completed'
    target=out/'interpretation';target.mkdir(exist_ok=False)
    errors=read_csv(out/'P1_Day2_errors_with_quality.csv');table=[]
    for repr,sl in [('spec',slice(0,104)),('cis',slice(104,208))]:
        state=np.load(out/f'rtl_6_{repr}_train_fitted_cluster.npz');v=np.load(cache_dir(ROOT,'rtl_6','validation')/'diagnostic_vectors.npy')[:,sl]
        z=(v-state['mean'])/state['scale'];center=state['cluster_centers'];cluster=((z[:,None,:]-center[None,:,:])**2).sum(2).argmin(1)
        for seed in (20261012,20261013,20261014):
            sample=[r for r in errors if int(r['seed'])==seed]
            for c in range(10):
                for k in range(10):
                    chosen=[r for r in sample if int(r['raw_label'])==31+c and cluster[int(r['row'])]==k]
                    if not chosen:continue
                    count=lambda pred:sum(int(r['raw_prediction'])==pred for r in chosen)
                    table.append({'representation':repr,'seed':seed,'raw_label':31+c,'train_fitted_cluster':k,'packets':len(chosen),'errors':sum(r['correct']=='False' for r in chosen),'predicted31':count(31),'predicted34':count(34)})
    write_csv(target/'P1_error_clusters.csv',table)
    drifts=read_csv(out/'centroid_shifts.csv');quality=read_csv(out/'per_device_feature_quality.csv');recheck=read_csv(out/'P1_CIS_rechecked.csv')
    selected=[r for r in quality if r['receiver']=='rtl_6' and r['raw_label']=='34' and r['role'] in ('train','same_day_diagnostic','validation') and r['metric'] in ('raw_rms','cis_abs_q99','denominator_fraction_relative_lt_1e2','CFO_metadata_raw_unknown_units','SNR_metadata_raw_unknown_units')]
    selected_d=[r for r in drifts if r['receiver']=='rtl_6' and r['raw_label']=='34'];write_json(target/'observable_factors.json',{'P1_model_recheck':recheck,'quality':selected,'feature_centroid_shifts':selected_d,'label_error_or_hardware_causality_established':False,'centroids_use_104d_frequency_means_std_not_full_maps_or_network_embeddings':True})
    notes='''# RTL设备34与频带诊断复盘

P1三seed模型重新读取固定环境缓存：Day1训练设备34均800/800，同日测试均200/200，Day2均0/200。三次Day2预测与冻结P1保存预测逐元素一致。Day2误判到31为177、121、195包。类标签范围、计数、行对应及六源文件SHA一致；这些检查不验证采集端设备身份，不能据此排除采集标签错误，也没有证据据此宣称标签错误。

RTL34原始RMS均值Day1Train 0.33094、Day1Test 0.31297、Day2 0.64070。CFO元数据均值-788.63→-798.24，SNR元数据31.735→37.010；单位和提取算法未知，不能直接解释Hz/dB，更不能重复补偿。CIS绝对值99分位的均值2.22000→2.17565，未出现明显极端值扩大。分母低于该包平均功率1e-2的比例均值0.67357→0.68813，增幅1.46个百分点；该统计在所有类本就偏高，需结合每类表，不能用高比例直接解释零召回。

104维描述向量为每频率沿时间的均值/标准差，仅用于描述分布。以Day1训练拟合StandardScaler：RTL34 Day1同日测试最近训练中心仍是34；Day2 Spec最近中心为31（距34为5.947，距31为1.212），CIS最近中心为35（距34为6.226，距31为2.124）。CIS最近中心并非31，不能把粗统计当成神经网络判别规则。Spec也发生偏移，不能称失效是CIS独有机制。KMeans只拟合Day1训练，10簇、n_init10、固定seed；P1错误与这些训练簇的对应在P1_error_clusters.csv。尚无受控信道/接收链变化或采集身份校验，当前证据支持跨日期表示分布失配的观察，不能证实其物理原因。

所有环带的log能量描述向量、训练内散度与跨日期中心漂移见ring_descriptive_separability.csv；该描述包含原始幅度尺度，与正式独立RMS分类器不是同一输入，不作为控制了能量的分类结论。packet_ring_energy_window_probe.csv包含原包前Hann的非严格泄漏探针；窗变化混合原始FFT邻近频率，不能用于证明严格OOB硬件信息。synthetic_inband_tone_leakage.csv证明纯带内非整频点有限记录亦可落入OOB，并不证明真实OOB全部来自泄漏。正式projection_hann先投影再窗，通过排除频带不变性测试。

所有图频率轴都依赖未核实Fs=1MHz/BW=125kHz假设。未读Day3/4、六锁定接收机；旧历史使用审查只读PDF和已有缓存文件头。
'''
    (target/'REPORT.md').write_text(notes)
    print(json.dumps({'completed':True,'cluster_rows':len(table)}))
