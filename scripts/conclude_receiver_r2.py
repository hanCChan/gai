#!/usr/bin/env python3
"""Write factual stage decisions from frozen outcomes; no training/selection."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];PUB=ROOT/'reports/receiver_r2/r2_a';OUT=ROOT/'outputs/receiver_r2'


def main():
    reg=json.loads((ROOT/'configs/receiver_r2/registration.json').read_text());verify=json.loads((PUB/'verification.json').read_text());assert verify['passed']
    summary=pd.read_csv(PUB/'horizon_summary.csv');paired=pd.read_csv(PUB/'paired_method_gains.csv');source=pd.read_csv(PUB/'source_selection_and_cost.csv');reliability=pd.read_csv(PUB/'risk_reliability_by_phase_scale.csv');allpairs=pd.read_csv(PUB/'C1_C2_RX_pairs.csv');offline=pd.read_csv(PUB/'offline_guard_difference.csv')
    scale1=allpairs[allpairs.scale==1];risk_summary=[]
    for (ds,m),g in scale1.groupby(['dataset','method']):
        hits=g.significant_harm_C1;both=(~g.near_zero_C1)&(~g.near_zero_C2)
        risk_summary.append(dict(dataset=ds,method=m,observations=len(g),sign_agreement=float((np.sign(g.delta_C1)==np.sign(g.delta_C2)).mean()),confident_both_n=int(both.sum()),confident_agreement=float((np.sign(g[both].delta_C1)==np.sign(g[both].delta_C2)).mean()) if both.any() else None,C1_detected_harm_n=int(hits.sum()),confirmed_harm_n=int((hits&g.significant_harm_C2).sum()),precision=float(g.significant_harm_C2[hits].mean()) if hits.any() else None,near_zero_C1_rate=float(g.near_zero_C1.mean())))
    risks=pd.DataFrame(risk_summary);risks.to_csv(PUB/'risk_reliability_overall_scale1.csv',index=False)
    fit=pd.read_csv(PUB/'actual_fitting_batch_risks.csv');fit=fit[(fit.tx==-999)&(fit.scale==1)]
    # Equal RX, then step: do not overweight steps with different held group sizes.
    fit_summary=[]
    for (ds,m,mode),g in fit.groupby(['dataset','method','mode']):
        if mode=='train':g=g.groupby(['run_id','epoch','rx']).delta.mean().reset_index()
        steps=g.groupby(['run_id','epoch']).delta.mean()
        fit_summary.append(dict(dataset=ds,method=m,mode=mode,steps=len(steps),actual_Adam_AB_mean_loss_positive_step_rate=float((steps>0).mean()),mean_delta_CE=float(steps.mean())))
    fitting=pd.DataFrame(fit_summary);fitting.to_csv(PUB/'actual_fitting_side_summary.csv',index=False)
    mode=pd.read_csv(PUB/'mode_dropout_comparison.csv');mode1=mode[mode.scale==1]
    execution=json.loads((OUT/'execution.json').read_text());evalcost=json.loads((PUB/'evaluation_cost.json').read_text());remeasure=json.loads((PUB/'fitting_remeasurement_cost.json').read_text())
    # Evaluation/replay process startup and GPU context teardown allowance30 seconds.
    total=execution['reserved_gpu_hours']+evalcost['reserved_GPUh']+remeasure['reserved_GPUh']+30/3600
    assert total<3
    accounting=dict(fitting_reserved_GPUh=execution['reserved_gpu_hours'],evaluation_reserved_GPUh=evalcost['reserved_GPUh'],read_only_fitting_remeasurement_reserved_GPUh=remeasure['reserved_GPUh'],GPU_process_setup_teardown_allowance_seconds=30,total_conservative_reserved_GPUh=total,cap_GPUh=3.,no_historical_budget_reuse=True,CPU_analysis_GPUh=0.,remaining_allowance_authorizes_new_jobs=False)
    (PUB/'RESERVATION_ACCOUNTING.json').write_text(json.dumps(accounting,indent=2)+'\n')
    lines=['# R2-A 收敛与真实 Adam 风险复核','',
        '结论：**MODIFY，继续研发但尚未授权 R2-C**。真实更新测量的工程核验通过；当前观察来自有限收敛对照及一个 seed 的早期短训练，不足以证明跨未知 RX 的新机制。R2-B 最小控制器已有确定性实现及离线行为报告，受控轨迹仍未训练。','',
        f'新批准矩阵完整完成：6项 WiSig 200轮对照＋6项短诊断；12项拟合，0失败、0拟合重试、0替换种子。累计保守占卡成本 **{total:.6f} reserved V100 GPUh**，独立上限3 GPUh。使用四张V100，每卡独立模型；GPUh按各作业占卡时间求和。','',
        '## 100/200轮完整配对结果','',
        'WiSig官方CNN、相同训练数据/LR0.0003/初始化/采样/source-only checkpoint规则；仅改变训练时长。六项200轮作业的前100轮源验证CE/Accuracy/F1、采样顺序及源选模型参数与历史完全相同。CE目标日志采用mean与none后mean两种浮点归约，最大差异7.16e-8，在执行前登记的1e-7核验范围内；未重训或更改原始日志。100轮与200轮都完整列出，不按内部RX分数选择时长。','',
        '| Horizon | 方法 | 内部RX新日期Accuracy | seed SD(pp) | Macro-F1 | RX q10 | 平均RX最差TX召回 |','|---|---|---:|---:|---:|---:|---:|']
    for r in summary[summary.role=='internal_unseen_date'].itertuples():
        lines.append(f'| {r.horizon} | {r.method} | {r.accuracy_mean*100:.2f}% | {r.accuracy_seed_SD_pp:.2f} | {r.macro_f1_mean*100:.2f}% | {r.RX_accuracy_q10_mean*100:.2f}% | {r.mean_RX_worst_TX_recall_mean*100:.2f}% |')
    lines+=['','所有seed/RX/日期/TX的完整表在证据ZIP；per_seed100_200、paired_horizon_changes、paired_method_gains及horizon_summary也直接公布。RX先等权平均日期，再等权平均RX；无支持真类召回为NA。三seed SD用ddof1，q10采用线性插值。','']
    for horizon in [100,200]:
        g=paired[(paired.horizon==horizon)&(paired.role=='internal_unseen_date')].accuracy_dateMLDG_minus_CE_pp
        lines.append(f'{horizon}轮日期MLDG对CE：均值{g.mean():+.3f} pp，{int((g>0).sum())}/3 seed正向，范围[{g.min():+.3f},{g.max():+.3f}] pp。')
    panel=summary[summary.role=='internal_unseen_date']
    ce_change=(float(panel[(panel.horizon==200)&(panel.method=='ce')].accuracy_mean.iloc[0])-float(panel[(panel.horizon==100)&(panel.method=='ce')].accuracy_mean.iloc[0]))*100
    meta_change=(float(panel[(panel.horizon==200)&(panel.method=='mldg_rx_date')].accuracy_mean.iloc[0])-float(panel[(panel.horizon==100)&(panel.method=='mldg_rx_date')].accuracy_mean.iloc[0]))*100
    lines+=['',f'延长训练时，CE平均Accuracy变化{ce_change:+.3f} pp，日期MLDG变化{meta_change:+.3f} pp；二者seed标准差均变大。历史日期MLDG优势在200轮不再成立，不能将其直接作为稳定跨RX机制的证据。这里不据目标结果回选100轮，也不更换下降seed。']
    lines+=['','源验证最优checkpoint（200轮）：'+', '.join(f'{r.method}/seed{r.seed}={r.selected_epoch}' for r in source[source.stage=='horizon'].itertuples())+'。延长训练的收益及边界checkpoint只能说明配方敏感性；200轮仍不是充分收敛的证明。学习曲线、源CE/F1、实际梯度/更新幅度和裁剪统计完整保留。','',
        '## Adam测量与源包独立重测','',
        '短诊断使用WiSig每cell80/20/20拟合/C1/C2及Shen120/20/20；原源验证40包从未作训练控制。A/B/C按WiSig5/5/4、Shen2/1/2预冻结轮换。CE与MLDG在同一拟合池和排除日程上配对，C不参与候选梯度。Adam状态/参数增量留在本地；公开其SHA、范数、计数和损失统计。scale1为真正接受的更新，其余倍率为只读有限变化。','',
        f'独立核验{verify["actual_Adam_snapshots_verified"]}份完整Adam状态；最大参数方程残差{verify["maximum_Adam_parameter_equation_abs_error"]:.3g}，最大动量方程残差{verify["maximum_Adam_moment_equation_abs_error"]:.3g}。候选与接受后的损失一致，scale0精确零，测量不改变模型/优化器/训练RNG。','',
        '| 数据 | 方法 | RX观测数 | C1/C2符号一致率 | 双侧超噪声观测 | C1显著变差被C2确认 | C1近零比例 |','|---|---|---:|---:|---:|---:|---:|']
    for r in risks.itertuples():
        lines.append(f'| {r.dataset} | {r.method} | {r.observations} | {r.sign_agreement*100:.1f}% | {r.confident_both_n} | {r.confirmed_harm_n}/{r.C1_detected_harm_n} | {r.near_zero_C1_rate*100:.1f}% |')
    lines+=['','“显著”在这里仅指超过预设2×paired-SE加数值分辨率的描述性标记，不是适应性多重测试下的统计保证。近零、不一致、所有倍率、阶段和RX×TX×日期cell均保留；20包不保证独立同分布。跨步骤合并排序可能被训练阶段混淆，within-step RX排序单独报告，Shen每步仅2台C接收机。该源包重测不证明新接收机风险、未知RX泛化或硬件因果。','',
        f'eval与三个配对Dropout掩码的train均值符号一致率（C1、scale1）为{mode1.eval_train_sign_agree.mean()*100:.1f}%；完整模式差异见ZIP中mode_dropout_comparison。训练模式诊断不是原训练目标随机掩码的重建。','',
        '真实Adam在自身A/B拟合批次上的变化（等权RX再等权步骤）：','',
        '| 数据 | 方法 | 测量模式 | 均值损失增加步骤比例 | 平均deltaCE |','|---|---|---|---:|---:|']
    for r in fitting.itertuples():lines.append(f'| {r.dataset} | {r.method} | {r.mode} | {r.actual_Adam_AB_mean_loss_positive_step_rate*100:.1f}% | {r.mean_delta_CE:+.6f} |')
    lines+=['','此测量在新保存的实际轨迹上回读，未新增拟合或修改参数。历史eval-SGD人工探针的训练侧异常率另列historical_probe_audit；不同拟合池/阶段/seed使其不能成为同轨迹因果对照。历史selected.pt没有完整Adam状态，未宣称重建旧轨迹。','',
        '## R2-B与后续公平性','',
        '具体公式、包数、有限倍率、C1误差容差、逐步拒绝/缩放的Adam状态策略及公平矩阵在docs/receiver_r2/R2_B_DEFINITION.md。规则允许个别RX正向损失变化；拒绝完整恢复Adam，非零倍率保留候选动量、仅缩放参数增量。','',
        f'离线多RX保护与普通平均风险回溯选择不同的快照为{int(offline.choices_differ.sum())}/{len(offline)}。阶段接受率/拒绝率/平均倍率/抽样有效更新范数代理及C2复测均在offline_controller_behavior.csv。它们来自原Adam快照，**不是受控训练轨迹**；无法据此推断长期全部拒绝、泛化增益或完整有效更新总量。R2-A实际训练全部scale1，完整短训练累计增量和训练时长另列source_selection_and_cost。','',
        '按数据集分开，Shen40份快照上的两种策略选择完全相同，WiSig40份仅6份不同。当前多RX保护在Shen的离线增量为零，不能预设它在两个数据集具有额外机制。已实现普通回溯与固定较小LR对照是后续解释收益的必要条件。', '',
        '未来四个最小对照为标准MLDG、固定半LR、普通平均源风险回溯及多RX保护；为了公平比较最强已实现方法，应加同池SWAD，初始五臂方案为45项；本轮200轮CE已高于日期MLDG，因此当前公平计划再加入同池CE，形成六臂54项候选矩阵（WiSig18项、Shen36项），见docs/receiver_r2/R2_C_FAIR_MATRIX.md。旧Shen SWAD的85.81%来自较大拟合池，只作背景参考，不能作为同池受控比较。原R0–R1的MLDG/CAGrad等数学核查及本轮primary-source复核未证明创新，不能将有限倍率回溯包装成独立新算法。','',
        '不新增类别保护、骨干、预处理、特征生成或外部数据。C2作为开发诊断已被查看，后续不能反复据此修规则仍声称独立核验；任何修改须披露并登记新的核验协议。R2-C必须在技术评估后单独批准完整模型数、新预算、峰值显存和超时阈值，本轮R2-C/最终测试均0项。','',
        '## 审查口径与证据','',
        '历史WiSig官方模型共18项，其中SWAD3项，普通非SWAD15项；11项普通checkpoint落在99/100轮。审查的18项口径包含SWAD，边界集中这一发现仍成立，历史报告未改。完整历史人工SGD异常率按search/formal和唯一probe步骤分别重算，避免重复RX/TX行改变分母。','',
        f'独立从预测重建{verify["confusion_matrices_rebuilt"]}个RX/日期混淆矩阵，核验112输入资产哈希、全部epoch成员/配对哈希以及6份历史100轮前缀。WiSig最终8RX/日期3/等化版、Shen四开发和六锁定RX、OSU继续排除。证据ZIP只含派生统计和完整日志；不含IQ、完整特征、权重或优化器状态。','',
        'CPU报告脚本的首次语法编译发现一个缺失括号，执行前修正。独立核验首轮将全部目标日志误按bitwise精确比较，发现CE归约记录差异后，按执行前已有1e-7目标日志核验范围修正；源验证仍要求精确一致。保留失败核验日志；没有因此重训、替换seed或重评目标数据。GPU占卡核算、环境、前置冻结、状态日志和独立verification均随证据公开。']
    (PUB/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (PUB/'DECISION.md').write_text('# R2-A/B decision\n\n**MODIFY — continue staged research; no R2-C authorization.**\n\nThe approved12 fits completed with0 fitting failures/retries/replacement seeds. Engineering checks pass for actual Adam proposals, full-state rollback/scaling, RNG-isolated risk measurement and six exact100-epoch source-validation/checkpoint replays (CE objective logging differs by at most7.16e-8, within the original preexecution1e-7 bound). Complete100/200 results and source C1/C2 noise/disagreement evidence are retained. At200 epochs, CE51.74% exceeds date-MLDG50.78%; historical+3.54pp becomes-0.96pp,1/3 positive seeds. Both seed standard deviations increase. The horizon comparison does not support treating the old gain as stable mechanism evidence.\n\nC1/C2 sign agreement is88.75%-95%;35%-47.5% of C1 risks are near descriptive packet-noise margins. Actual Adam fitting-side positive loss-change steps are15%-25%, unlike the historical artificial SGD anomaly. These are one-seed early source diagnostics, not controlled stability or novel unseen-RX benefit.\n\nThe minimal source-error-aware finite controller and matched mean-risk control are implemented. They choose identically on all40 Shen snapshots and differ on only6/40 WiSig snapshots. Offline behavior is not a fitted controller trajectory; no added Shen mechanism has been demonstrated. Further PoC should include fair-pool SWAD as the strongest historical Shen alternative, alongside standard/date-MLDG and smaller LR. The current fair proposal includes matched CE as well: six arms,54 fits (18 WiSig,36 Shen), inactive and separately approved. Formal model count, horizon, cost, memory and timeout budget must be separately approved before any R2-C fit. No class-tail module or final test is authorized.\n\nConservative new allocation used: '+f'{total:.9f}'+' reserved V100 GPUh out of3. Unused approval must not start extra jobs. Publish this stage, verify remote file tree/bytes and report/decision, then stop.\n')
    b=ROOT/'reports/receiver_r2/r2_b';b.mkdir(exist_ok=True)
    (b/'REPORT.md').write_text('# R2-B minimal controller implementation\n\nConcrete definition: [R2_B_DEFINITION](../../../docs/receiver_r2/R2_B_DEFINITION.md). Source code: [controller.py](../../../src/receiver_r2/controller.py) and [measurement.py](../../../src/receiver_r2/measurement.py).\n\nDeterministic tests validate complete Adam rollback including the next update, nonzero-scaling moment retention, measurement RNG isolation and positive individual RX risk tolerance. Definition sealed before aggregate C2 analysis. Source-error noise constants are fixed; C2/target outcomes never choose a multiplier. No controlled fitting.\n\nOffline source behavior and all scale/phase tables are in [R2-A evidence](../r2_a/EVIDENCE_INDEX.md); they do not form a controlled training trajectory. Initial five-arm45-fit fair matrix included matched SWAD. The current [six-arm54-fit fair proposal](../../../docs/receiver_r2/R2_C_FAIR_MATRIX.md) also includes matched CE, following the frozen convergence assessment. Horizon and new budget require separate approval.\n')
    (b/'DECISION.md').write_text('# R2-B decision\n\n**GO for engineering readiness; MODIFY for the research claim.** Minimal controller implementation and deterministic semantics are reviewable. No independent-algorithm novelty or generalization gain has been established. Controlled R2-C jobs0; separate approval remains required. C2-informed revisions must disclose diagnostic reuse and freeze a new validation protocol.\n')
    print(json.dumps(accounting));print(risks.to_string(index=False));print(fitting.to_string(index=False))

if __name__=='__main__':main()
