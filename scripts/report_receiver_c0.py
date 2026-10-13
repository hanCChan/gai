#!/usr/bin/env python3
"""Cost forecast, dependence safeguards and factual research/engineering reports."""
import json,math,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];PUB=ROOT/'reports/receiver_c0';OUT=ROOT/'outputs/receiver_c0'


def main():
    prediction=pd.read_csv(PUB/'transfer_prediction_summary.csv');behavior=pd.read_csv(PUB/'transfer_behavior_summary.csv');pred=pd.read_csv(PUB/'transfer_predictions.csv');guard=pd.read_csv(PUB/'guard_vs_mean_choices.csv')
    timings=[];sparse=[];gradient=[]
    for card in range(4):
        r=json.loads((OUT/f'timing_gpu{card}.json').read_text());timings+=r['timing_cases'];sparse+=r['equivalence_cases'];gradient+=r['gradient_cases']
    t=pd.DataFrame(timings);s=pd.DataFrame(sparse);t.to_csv(PUB/'single_step_timings.csv',index=False);s.to_csv(PUB/'GPU_equivalence_and_sparse_diagnosis.csv',index=False)
    warm=t[t.epoch!=1];cost=[]
    for ds,epochs,steps,val_n in [('wisig',200,40,11200),('shen',20,60,2000)]:
        g=warm[warm.dataset==ds];gradmax=float(g.gradient_seconds.max());diagmax=float(s[(s.dataset==ds)&(s.epoch!=1)].sparse_diagnostic_seconds.max())
        percall=max(float(r.decision_seconds)/int(r.forward_calls) for r in g.itertuples() if r.mechanism.startswith('optimized') and r.forward_calls>0)
        validation_allowance=percall*math.ceil(val_n/256)+1.;cold_gradient=max(float(r['gradient_seconds']) for r in gradient if r['dataset']==ds)
        for mechanism in ['standard','optimized_mean','optimized_multi']:
            decisionmax=float(g[g.mechanism==mechanism].decision_seconds.max());opportunity=gradmax+decisionmax
            # Separate measured warm maxima from explicit conservative unmeasured allowances.
            raw=epochs*steps*opportunity+epochs*(validation_allowance+3.+diagmax)+90.+cold_gradient
            cost.append(dict(dataset=ds,mechanism=mechanism,seeds=3,epochs=epochs,steps_per_epoch=steps,total_opportunities=epochs*steps,gradient_warm_max_seconds=gradmax,decision_warm_max_seconds=decisionmax,full_opportunity_warm_max_seconds=opportunity,sparse_C2_full_transaction_allowance_seconds_per_epoch=diagmax,source_validation_estimated_seconds_per_epoch=validation_allowance,logging_checkpoint_allowance_seconds_per_epoch=3.,setup_allowance_seconds=90.,cold_gradient_measured_seconds=cold_gradient,margin_multiplier=1.5,three_seed_forecast_GPUh=raw*1.5*3/3600))
    costs=pd.DataFrame(cost);costs.to_csv(PUB/'conditional_18_fit_cost_forecast.csv',index=False);forecast=float(costs.three_seed_forecast_GPUh.sum())
    ledger=json.loads((OUT/'timing_execution.json').read_text());total=ledger['reserved_GPUh'];summary=dict(formal_fits=0,reset_single_step_timing_cases=len(t),additional_original_proposal_replays=24,sparse_diagnostic_transaction_replays=len(s),total_disposable_Adam_steps=len(t)+24+len(s),continuous_trajectory_updates=0,engineering_reserved_GPUh=total,watchdog_seconds=360,conditional_18_fit_forecast_GPUh=forecast,proposed_round_cap_if_later_eligible_GPUh=math.ceil(forecast)+1,forecast_is_conditional_not_approval=True,cold_start_not_charged_per_opportunity=True,short_snapshot_timings_not_guaranteed_full_training_bound=True,peak_by_GPU={str(card):{'allocated_decision_peak_MiB':float(t[t.gpu==card].peak_allocated_bytes.max()/2**20),'reserved_peak_including_gradient_cache_MiB':float(t[t.gpu==card].peak_reserved_bytes.max()/2**20)} for card in range(4)})
    (PUB/'COST.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Fixed RX/date-transfer pairs rarely recur: expose their support explicitly.
    fixed=pred.groupby(['dataset','method','kind','run_id','decision_RX','target_RX','decision_date','target_date']).agg(snapshots=('epoch','nunique')).reset_index();fixed.to_csv(PUB/'fixed_transfer_pair_support.csv',index=False)
    fixed_corr=[]
    for key,g in pred.groupby(['dataset','method','kind','run_id','decision_RX','target_RX','decision_date','target_date']):
        z=g.sort_values('epoch');n=len(z);corr=np.nan
        if n>=3 and z.source_full_delta.std()>0 and z.target_full_delta.std()>0:corr=float(z.source_full_delta.corr(z.target_full_delta,method='spearman'))
        fixed_corr.append(dict(zip(['dataset','method','kind','run_id','decision_RX','target_RX','decision_date','target_date'],key),snapshots=n,rank=corr,sufficient_three_points=n>=3))
    pd.DataFrame(fixed_corr).to_csv(PUB/'fixed_pair_ranks.csv',index=False)
    clipping=pd.read_csv(PUB/'sampled_gradient_clipping.csv');clip=clipping.groupby(['method','phase']).agg(observed_epoch_first_batches=('epoch','count'),clip_rate=('clip_applied','mean'),mean_preclip_norm=('gradient_norm_preclip','mean'),mean_candidate_delta=('candidate_delta_L2','mean')).reset_index();clip.to_csv(PUB/'clipping_summary.csv',index=False)
    packets=pd.read_csv(PUB/'packet_risk_dependence.csv');features=pd.read_csv(PUB/'packet_index_feature_correlations.csv');depend=packets.groupby('dataset').agg(cells=('n','count'),median_lag1=('risk_delta_lag1','median'),median_block4_ratio=('block4_over_naive_SE','median')).reset_index();depend.to_csv(PUB/'packet_dependence_summary.csv',index=False)
    source_audit=json.loads((PUB/'source_pool_audit.json').read_text())
    audit=['# 源码、数据与监督预算审计','',
        '旧R2-A/B源码、常数、报告及决策保持原提交字节。新优化代码位于src/receiver_c0，不替换旧执行器。','',
        'A/B/C在每次历史采样中互斥，C不进入MLDG候选梯度；WiSig互补日期episode覆盖A/B的两个源日期，α=.01、β=1、外层Adam及clip5保持原协议。80快照保存了真实已裁剪梯度、更新前后参数及Adam状态；旧源码保存的是梯度计算之后的RNG，不能宣称重建了原始Dropout掩码。计时的精确更新重放使用已保存梯度；单独全梯度计时使用新的固定掩码，结果不用于任何模型拟合。','',
        'CE与MLDG的裁剪率差异是优化混杂，未证明Bug或因果。下表每个epoch只观察首批，不代表所有训练步骤；Adam对恒定整体缩放近似不变，并不保证对随步骤变化的裁剪完全不变。没有新增裁剪训练消融，不能据此解释未知RX成绩。','',clip.to_markdown(index=False) if False else clip.to_string(index=False),'',
        'source_pool_audit验证了拟合/C1/C2/验证包身份互斥。WiSig风险池原position80:100与100:120连续相邻；Shen使用每类已排序缓存位置120:140与140:160，其原HDF5行索引也沿池递增。metadata没有可靠的包时间戳或session分割，位置编号不是已验证的采集时序。源验证40包未用于风险控制。','',
        '在缓存上只计算了I/Q均值/四阶矩（WiSig功率已归一化）及CIS均值/标准差/能量（Shen）等简单代理，不读取原始IQ或额外HDF5。特征与索引相关性逐RX/TX/date公开，但没有session独立性证明。20包风险差的lag1中位数及4包块SE比值如下；未发现强的普遍正相邻风险相关性，有限样本及这些代理也不能排除批次相关性。块SE只是描述，未替换控制器的固定2×SE。','',depend.to_string(index=False),'',
        '必须区分受限梯度池和总标签预算。三臂PoC的梯度只用A/B拟合包，但控制器还逐步读取C1真实TX标签。其收益包含额外监督反馈，不能全部归为优化创新。发表级比较须加入允许CE/MLDG直接训练fit+C1的同总标签预算对照，并披露步数、标签重复使用和计算量；C2始终不训练、不选阈值。','',
        'Shen跨RX1/1审计的判断端只有一个RX，原多RX规则必然退化为均值回溯：第一条件D≤ε成立时，第二条件D−ε≤max(0,−D)+ε自动成立。此设置能观察源RX风险转移，不能构成多RX机制的独立收益检验。']
    (PUB/'SOURCE_AUDIT.md').write_text('\n'.join(audit)+'\n')
    transfer=['# 跨RX/日期风险转移报告','',
        '**探索性分析，C2已被此前研发查看，不是新盲测。** 使用80份真实Adam快照，没有训练或新的候选方向。WiSig全部6个有向2/2互斥RX划分，Shen两个1/1划分；日期0->1、1->0两方向。每个选择仅使用判断侧C1，目标侧C2仅标记后果。','',
        '目标RX只是不参与本次A/B候选和C1判断；这些源RX可能在此前步骤已参与模型训练。该审计比同RX包重复更强，但仍没有验证整个训练过程从未见过的RX，更没有验证最终部署RX。','',
        '| 数据/方法 | 层次 | 快照/轮换观测 | 符号一致 | 有害判断精确率 | 召回 | 平衡准确率 | 风险AUC |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in prediction[prediction.method.str.startswith('mldg')].itertuples():transfer.append(f'| {r.dataset}/{r.method} | {r.kind} | {r.unique_snapshots}/{r.split_observations} | {r.sign_agreement:.3f} | {r.harm_precision:.3f} | {r.harm_recall:.3f} | {r.balanced_accuracy:.3f} | {r.raw_source_risk_AUC:.3f} |')
    transfer+=['','WiSig日期MLDG跨RX的精确率0.20等于目标有害率0.20，冻结有害判断的平衡准确率0.50；跨RX+日期约0.506。连续排序有部分信息，但早期AUC约0.882、短训练末段约0.556，缺乏稳定检测优势。Shen MLDG跨RX有较强探索性信号，但只有两条10轮轨迹（两个fold、同seed），每步目标仅一台RX。不能由此宣称跨数据集通用检测。','',
        '误选定义：接受非零倍率而目标C2均值为正（raw），以及超过目标描述性噪声幅度（above_noise）；错误完全拒绝要求full目标均值低于负噪声界。较差RX为目标RX损失变化最大值。零更新不能仅凭风险零被视为训练最优，mean_scale和拒绝率一并公开。','',
        '跨RX的行为结果：','',behavior[(behavior.kind=='cross_RX')&behavior.method.str.startswith('mldg')].to_string(index=False),'',
        '多RX与普通回溯跨RX不同选择仅4/320个轮换观测（其中含CE诊断；对应真实快照更少）；同RX整组重现原6/80差异。WiSig MLDG跨RX两规则的above-noise误接受率同为0.1583；固定半倍率的平均/较差目标损失下降更大，但其更新量更小，不能直接推出分类更好。Shen1/1两规则必然相同。','',
        '所有RX/日期/phase子组、支持数、风险排序和逐步实际C2损失均公开。轮换有共享与互补结构，不能把120个划分当120个训练seed。trajectory_dependence中的轮换均值塌缩为原同队列共同变化，不能当固定不同RX预测证明；fixed_pair_support/ranks进一步列出真正固定RX对的重复支持，通常稀少。first-difference和lag1均为描述，没有80独立采集的置信区间或packet级p值。','',
        '**判断：MODIFY。** 发现部分源风险排序信息，不支持当前多RX保护比均值回溯稳定地提供额外跨RX检测。保留研究问题，但不凭此自动启动18或54项训练，不针对C2搜索阈值。']
    (PUB/'TRANSFER_REPORT.md').write_text('\n'.join(transfer)+'\n')
    cost_report=['# 受控步骤等价优化与成本','',
        f'本次{len(t)}个计时case、24个原候选数值重放、{len(s)}个稀疏诊断case，共{len(t)+24+len(s)}次一次性Adam步骤；每次从旧状态重置并丢弃结果，连续训练轨迹0、正式拟合0。四卡工程占卡总额{total:.9f} GPUh，独立360秒watchdog，不扣历史3GPUh余额。','',
        '优化只做等价执行：每步缓存pre-C1，按1/.5/.25降序提前退出，复用已接受损失；C2只在固定稀疏时点读取pre和选中倍率，不遍历全部倍率。拒绝时目标损失直接为0。缓存不跨改变参数的步骤。四卡所有legacy/optimized对应case的倍率、参数和Adam状态精确一致，C2不改变决策；原代码及规则不修改。','',
        '原WiSig执行器每步112次分批forward、25600个包前向；Shen32次、6400包。新无C2控制最少2个全池pass，最多4个；实际WiSig14或28分批forward，Shen4。稀疏C2仅增加1或2个全池pass。','',
        'warm步骤统计（首case的CUDA冷启动另计，不按每步重复计入）：','',warm.groupby(['dataset','mechanism']).agg(max_gradient_seconds=('gradient_seconds','max'),max_decision_seconds=('decision_seconds','max'),max_total_seconds=('complete_opportunity_seconds','max')).to_string(),'','四卡显存：',json.dumps(summary['peak_by_GPU'],indent=2),'',
        'allocated峰值是decision阶段的计数；reserved峰值包含此前完整MLDG梯度的allocator缓存，是观测到的整步内存保守包络，不冒称精确梯度allocated峰值。时长另含真实梯度、源验证估算、每epoch3秒日志/checkpoint余量、稀疏C2完整事务余量、90秒setup与50%余量。三种方法的保守条件估算见CSV；不能把三份短轨迹样本视为未来完整训练的确定硬上界，正式预算与超时watchdog还需批准。','',
        f'如果以后科学条件满足，18项完整矩阵的条件估算为{forecast:.3f} reserved GPUh，建议独立上限{summary["proposed_round_cap_if_later_eligible_GPUh"]} GPUh；本次未申请或使用这笔正式训练预算。不得引用原R2-A1.03GPUh直接推算控制器训练。']
    (PUB/'COST_REPORT.md').write_text('\n'.join(cost_report)+'\n')
    decision=dict(status='MODIFY_CONTINUE_RESEARCH',engineering_equivalence_passed=True,cross_RX_signal='partial and dataset/phase dependent; WiSig frozen harm decision chance-level; Shen ranking stronger',independent_multi_RX_increment='cross_RX choices4/320; Shen1/1 degenerate',formal_R2_C1_gate_passed=False,formal_fits=0,formal_budget_requested=False,conditional_plan_only=True,engineering_reserved_GPUh=total,no_threshold_or_model_reselection=True)
    (PUB/'decision.json').write_text(json.dumps(decision,indent=2)+'\n');(PUB/'DECISION.md').write_text('# Decision\n\n**MODIFY_CONTINUE_RESEARCH.** Engineering optimization is equivalent and costed. Cross-source risk transfer has partial signal, but the frozen WiSig harm decision has balanced accuracy0.50 (crossRX) and0.506 (crossRX+date); multi-RX guard adds very few choices and Shen1/1 is mathematically identical to mean backtracking. This does not establish useful independent cross-RX protection.\n\nDo not automatically activate18 or54 formal fits or tune constants on C2. A complete conditional18-fit protocol and cost forecast are delivered for review, inactive until scientific reassessment and fresh explicit budget approval. No complete unseen-during-training RX or deployment risk has been verified. Preserve the broader research question; this is not a universal no-signal claim from one seed.\n\nPublish/verify this audit, then stop before fitting. No remaining historical allocation may fund extra jobs.\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for col,ds in enumerate(['wisig','shen']):
        g=pred[(pred.dataset==ds)&pred.method.str.startswith('mldg')&(pred.kind=='cross_RX')]
        axes[col].scatter(g.source_full_delta,g.target_full_delta,s=15,alpha=.4);axes[col].axhline(0,color='grey');axes[col].axvline(0,color='grey');axes[col].set_title(ds+' exploratory cross RX');axes[col].set_xlabel('Decision RX C1 deltaCE');axes[col].set_ylabel('Target RX C2 deltaCE')
    fig.tight_layout();fig.savefig(PUB/'cross_RX_risk_transfer.png',dpi=160);plt.close(fig)
    print(json.dumps(summary));print(json.dumps(decision))

if __name__=='__main__':main()
