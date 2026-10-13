#!/usr/bin/env python3
"""Layered receiver/date/seed analysis and derived-only evidence publication package."""
import csv,datetime,hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from wisig_w01_common import sha,dump


def main():
    out=ROOT/'outputs/receiver_r01';public=ROOT/'reports/receiver_r01/2026-10-13'
    reg=json.loads((ROOT/'configs/receiver_r01/registration.json').read_text());execution=json.loads((out/'execution.json').read_text());assert execution['state']=='completed'
    pd.DataFrame([dict(dataset='wisig',class_index=t,transmitter=reg['wisig']['tx_names'][t]) for t in range(10)]+
                 [dict(dataset='shen',class_index=t,transmitter='TX'+str(t+31)) for t in range(10)]).to_csv(public/'identity_labels.csv',index=False)
    pd.DataFrame([dict(date_index=d,capture_date=reg['wisig']['dates'][d],role='train_seen' if d<2 else 'development_unseen') for d in range(3)]).to_csv(public/'date_labels.csv',index=False)
    for name in ['execution.json','budget_calibration.json','initial_dispatcher_estimate_halt.json','shen_calibration_execution.json','method_specific_budget.json','resume_budget_hold.json']:
        p=out/name
        if p.exists():(public/name).write_bytes(p.read_bytes())
    costs=[];search=[];curves=[];risk=[];paired_checks=[]
    for stage in ['search','formal']:
        for job in reg[stage+'_jobs']:
            run=out/stage/job['run_id'];config=json.loads((run/'config.json').read_text());status=json.loads((run/'status.json').read_text());selection=json.loads((run/'selection.json').read_text());assert status['state']=='completed'
            common={k:job[k] for k in ['dataset','backbone','fold','method','seed','run_id']};common['stage']=stage
            ledger=next(r for r in execution['jobs'] if r['run_id']==job['run_id'])
            costs.append(dict(**common,parameters=config['parameters'],epochs=job['epochs'],steps=status['steps'],
                process_seconds=status['seconds'],reserved_seconds=ledger['reserved_seconds'],fitting_seconds=status['fitting_seconds'],
                inference_seconds_per_packet=status['inference_seconds_per_packet'],peak_allocated_bytes=status['peak_allocated_bytes'],
                lr=config['lr'],source_train_accuracy=selection['frozen_train_scores']['accuracy'],
                source_val_accuracy=selection['source_validation']['accuracy'],source_val_CE=selection['source_validation']['CE'],
                selected_epoch=selection.get('epoch'),swad_converged=selection.get('converged'),swad_segments=selection.get('n_segments')))
            if stage=='search':search.append(dict(**common,lr=config['lr'],**selection['source_validation']))
            df=pd.read_csv(run/'history.csv')
            for k,v in common.items():df[k]=v
            curves.append(df)
            diagnostics=pd.read_csv(run/'update_diagnostics.csv')
            for k,v in common.items():diagnostics[k]=v
            diagnostics['lr']=config['lr']
            selected_end=selection.get('end_step',selection.get('epoch',job['epochs'])*(status['steps']//job['epochs']))
            diagnostics['selected_end_step']=selected_end
            diagnostics['within_selected_checkpoint_prefix']=diagnostics.step<=selected_end
            risk.append(diagnostics)
            paired_checks.append(dict(**common,initial_sha256=config['initial_sha256'],schedule_sha256=config['schedule_sha256'],train_data_sha256=config['fitting_data_sha256']['x']))
    pd.DataFrame(costs).to_csv(public/'training_cost.csv',index=False)
    formal_cost=pd.DataFrame(costs);formal_cost=formal_cost[formal_cost.stage=='formal']
    pd.read_csv(public/'per_seed.csv').merge(formal_cost[['run_id','source_train_accuracy','source_val_accuracy','source_val_CE','selected_epoch','swad_converged']],on='run_id',validate='many_to_one').to_csv(public/'source_validation_vs_development.csv',index=False)
    pd.DataFrame(search).to_csv(public/'source_search_metrics.csv',index=False)
    pd.concat(curves).to_csv(public/'learning_curves.csv',index=False)
    pd.DataFrame(paired_checks).to_csv(public/'paired_initialization_sampling.csv',index=False)
    diagnostics=pd.concat(risk);diagnostics.to_csv(public/'update_diagnostics.csv',index=False)
    risk_keys=['dataset','backbone','fold','method','seed','stage','run_id','lr']
    drx=diagnostics.groupby(risk_keys+['epoch','step','rx','date'],as_index=False).agg(
        inner_delta=('inner_delta','mean'),outer_delta=('outer_delta','mean'),gradient_cosine=('gradient_cosine','first'),
        source_inner_delta=('meta_train_inner_delta','first'),positive_TX_fraction=('outer_delta',lambda a:float((a>0).mean())),
        maximum_TX_loss_increase=('outer_delta','max'))
    boundaries=diagnostics.groupby('run_id').selected_end_step.first().to_dict()
    drx['selected_end_step']=drx.run_id.map(boundaries)
    drx['within_selected_checkpoint_prefix']=drx.step<=drx.selected_end_step
    drx['source_beneficial_held_harmful']=(drx['source_inner_delta']<0)&(drx['inner_delta']>0)
    drx.to_csv(public/'update_RX_risks.csv',index=False)
    rs=drx.groupby(risk_keys,as_index=False).agg(
        probes=('inner_delta','count'),gradient_conflict_fraction=('gradient_cosine',lambda a:float((a<0).mean())),
        inner_RX_harm_fraction=('inner_delta',lambda a:float((a>0).mean())),outer_RX_harm_fraction=('outer_delta',lambda a:float((a>0).mean())),
        source_beneficial_held_harmful_fraction=('source_beneficial_held_harmful','mean'),
        mean_positive_TX_fraction=('positive_TX_fraction','mean'),mean_max_TX_loss_increase=('maximum_TX_loss_increase','mean'))
    rs.to_csv(public/'update_risk_per_seed.csv',index=False)
    prefix=drx[drx.within_selected_checkpoint_prefix]
    prefix_summary=prefix.groupby(risk_keys,as_index=False).agg(
        probes=('inner_delta','count'),gradient_conflict_fraction=('gradient_cosine',lambda a:float((a<0).mean())),
        inner_RX_harm_fraction=('inner_delta',lambda a:float((a>0).mean())),outer_RX_harm_fraction=('outer_delta',lambda a:float((a>0).mean())),
        source_beneficial_held_harmful_fraction=('source_beneficial_held_harmful','mean'),mean_positive_TX_fraction=('positive_TX_fraction','mean'))
    prefix_summary.to_csv(public/'update_risk_selected_prefix.csv',index=False)
    perrx=pd.read_csv(public/'per_RX_date.csv');pertx=pd.read_csv(public/'per_RX_TX_date.csv')
    # Merge Shen folds at RX level, not means of fold quantiles.
    columns=['dataset','backbone','method','seed','role'];combined=[];combined_rx=[]
    for key,g in perrx.groupby(columns,sort=True):
        rx=g.groupby('rx')[['accuracy','macro_f1','worst_tx_recall','balanced_accuracy']].mean()
        matching=pertx
        for col,value in zip(columns,key):matching=matching[matching[col]==value]
        tail_by_rx=matching.groupby(['rx','tx']).recall.mean().groupby('rx').min()
        acc=rx.accuracy.to_numpy();tail=tail_by_rx.loc[rx.index].to_numpy()
        for receiver,values in rx.iterrows():
            combined_rx.append(dict(zip(columns,key),rx=int(receiver),accuracy=float(values.accuracy),macro_f1=float(values.macro_f1),
                worst_TX_recall_after_equal_date_mean=float(tail_by_rx.loc[receiver]),mean_date_worst_TX_recall=float(values.worst_tx_recall)))
        combined.append(dict(zip(columns,key),RX_count=len(rx),accuracy=float(acc.mean()),macro_f1=float(rx.macro_f1.mean()),
            RX_accuracy_q10=float(np.quantile(acc,.1)),RX_accuracy_std=float(acc.std(ddof=1)),
            mean_RX_worst_TX_recall=float(tail.mean()),mean_RX_date_worst_TX_recall=float(rx.worst_tx_recall.mean()),RX_worst_TX_recall_q10=float(np.quantile(tail,.1)),
            min_RX_accuracy=float(acc.min()),balanced_accuracy=float(rx.balanced_accuracy.mean())))
    combined=pd.DataFrame(combined);combined.to_csv(public/'combined_per_seed.csv',index=False)
    pd.DataFrame(combined_rx).to_csv(public/'combined_per_RX.csv',index=False)
    summary=combined.groupby(['dataset','backbone','method','role'],as_index=False).agg(
        seeds=('seed','nunique'),accuracy=('accuracy','mean'),seed_accuracy_std=('accuracy','std'),
        accuracy_min_seed=('accuracy','min'),accuracy_max_seed=('accuracy','max'),macro_f1=('macro_f1','mean'),
        RX_accuracy_q10=('RX_accuracy_q10','mean'),RX_accuracy_std=('RX_accuracy_std','mean'),
        mean_RX_worst_TX_recall=('mean_RX_worst_TX_recall','mean'),RX_worst_TX_recall_q10=('RX_worst_TX_recall_q10','mean'))
    assert (summary.seeds==3).all();summary.to_csv(public/'method_summaries.csv',index=False)
    base=combined[combined.method=='ce'];paired=combined.merge(base,on=['dataset','backbone','seed','role'],suffixes=('','_CE'))
    for metric in ['accuracy','macro_f1','RX_accuracy_q10','mean_RX_worst_TX_recall']:
        paired[metric+'_delta_pp']=(paired[metric]-paired[metric+'_CE'])*100
    paired.to_csv(public/'paired_seed_changes.csv',index=False)
    psummary=paired.groupby(['dataset','backbone','method','role'],as_index=False).agg(
        seeds=('seed','nunique'),accuracy_delta_pp=('accuracy_delta_pp','mean'),accuracy_min_delta_pp=('accuracy_delta_pp','min'),
        accuracy_max_delta_pp=('accuracy_delta_pp','max'),positive_seeds=('accuracy_delta_pp',lambda a:int((a>0).sum())),
        q10_delta_pp=('RX_accuracy_q10_delta_pp','mean'),worst_TX_recall_delta_pp=('mean_RX_worst_TX_recall_delta_pp','mean'))
    psummary.to_csv(public/'paired_method_changes.csv',index=False)
    cel=pertx[pertx.method=='ce'];join=['dataset','backbone','fold','seed','role','rx','date','tx']
    changes=pertx.merge(cel,on=join,suffixes=('','_CE'));changes['recall_delta_pp']=(changes.recall-changes.recall_CE)*100
    changes.to_csv(public/'paired_RX_TX_changes.csv',index=False)
    harms=changes.groupby(['dataset','backbone','method','seed','role'],as_index=False).agg(
        supported_cells=('recall_delta_pp','count'),recall_drop_ge5pp_cells=('recall_delta_pp',lambda a:int((a<=-5).sum())),
        worst_recall_delta_pp=('recall_delta_pp','min'),mean_recall_delta_pp=('recall_delta_pp','mean'))
    harms.to_csv(public/'class_degradation_per_seed.csv',index=False)
    evaluation=json.loads((public/'evaluation_cost.json').read_text());pre=json.loads((public/'numerical_preflight.json').read_text())
    # Ten seconds per standalone check/evaluation for interpreter/setup overhead: conservative reservation allowance.
    overhead=(pre['seconds']+evaluation['seconds']+20)/3600
    cap=json.loads((ROOT/'configs/receiver_r01/budget_amendment.json').read_text())['new_cap_reserved_V100_GPUh']
    total=execution['reserved_gpu_hours']+overhead;assert total<cap
    cost=dict(fitting_reserved_GPUh=execution['reserved_gpu_hours'],numerical_and_evaluation_with_setup_allowance_GPUh=overhead,
        total_conservative_reserved_GPUh=total,cap_GPUh=cap,setup_allowance_seconds=20,fitting_jobs=85,search_jobs=34,formal_jobs=51,
        failures=0,retries=0,seed_replacements=0,r2_fitting_jobs=0,final_testing_jobs=0)
    dump(public/'total_cost.json',cost)
    swad=pd.DataFrame(costs);swad=swad[(swad.method=='swad')&(swad.stage=='formal')]
    selection_quality=dict(formal_SWAD_jobs=len(swad),converged_formal_SWAD_jobs=int((swad.swad_converged==True).sum()),
        fallback_formal_SWAD_jobs=int((swad.swad_converged==False).sum()))
    # Result-level repeated improvement is descriptive, with no packet-level p-values.
    wifi=summary[(summary.dataset=='wisig')&(summary.backbone=='official')&(summary.role=='internal_unseen_date')]
    sh=summary[(summary.dataset=='shen')&(summary.role=='source_pool_outer_RX')]
    best_wifi=wifi.sort_values('accuracy',ascending=False).iloc[0].to_dict();best_shen=sh.sort_values('accuracy',ascending=False).iloc[0].to_dict()
    wce=wifi[wifi.method=='ce'].iloc[0];wmeta=wifi[wifi.method=='mldg_rx_date'].iloc[0]
    hce=sh[sh.method=='ce'].iloc[0];hswa=sh[sh.method=='swad'].iloc[0]
    wp=psummary[(psummary.dataset=='wisig')&(psummary.backbone=='official')&(psummary.role=='internal_unseen_date')&(psummary.method=='mldg_rx_date')].iloc[0]
    hp=psummary[(psummary.dataset=='shen')&(psummary.role=='source_pool_outer_RX')&(psummary.method=='swad')].iloc[0]
    cap_control=summary[(summary.backbone=='resnet')&(summary.role=='internal_unseen_date')].iloc[0]
    decision=dict(status='R0_R1_COMPLETE_STOP_BEFORE_R2_AND_FINAL_TEST',
        approved_matrix_completed=True,best_WiSig_internal_unseen_date=best_wifi,best_Shen_historical_source_CV=best_shen,
        selection_quality=selection_quality,cost=cost,
        no_new_algorithm_claim=True,causal_update_reliability_hypothesis_proven=False,
        R1_result_interpretation='No uniformly stable cross-dataset remedy in this finite recipe; WiSig date-MLDG mean gain is 2/3-seed and increases seed variability; Shen SWAD has small 3/3-seed mean/tail gains; check recipe/convergence/probe repeatability before novel-mechanism claims',
        R2_design_only=True,R2_training_authorized=False,final_RX_date_equalized_sealed=True,
        no_single_seed_automatic_rejection=True,remaining_budget_reuse_authorized=False)
    dump(public/'decision.json',decision)
    lines=['# Receiver-held-out R0–R1 完整报告','',
        '2026-10-13，北京时间。已完成冻结34源内搜索+51正式拟合，零失败/重试/seed替换；三个配对seed，两数据集，全部checkpoint只用源域选择。停止在R1，R2只有设计，最终确认测试未运行。','',
        f"累计reserved拟合成本{cost['fitting_reserved_GPUh']:.6f} V100 GPUh；含数值预检、冻结评价及20秒进程启动余量的保守总额{total:.6f} GPUh，低于用户后续批准的7小时上限。原6小时不足保守完整矩阵估计约6.22，12个首作业后用户明确提高到7并保持原矩阵，批准记录公开。",'',
        '## 主要发现','',
        f"WiSig官方骨干、内部RX且未见开发日期：CE为{wce.accuracy*100:.2f}%±{wce.seed_accuracy_std*100:.2f}pp；RX+日期MLDG为{wmeta.accuracy*100:.2f}%±{wmeta.seed_accuracy_std*100:.2f}pp，平均提高{wp.accuracy_delta_pp:.2f}pp，但仅{wp.positive_seeds}/3个seed改善，配对变化范围[{wp.accuracy_min_delta_pp:.2f},{wp.accuracy_max_delta_pp:.2f}]pp。RX下10%分位平均提高{wp.q10_delta_pp:.2f}pp，平均RX最差TX Recall仍仅{wmeta.mean_RX_worst_TX_recall*100:.2f}%。这个配方有均值收益但没有解决seed稳定性，不能只引用平均增益。",
        f"Shen历史源池两折CV：CE为{hce.accuracy*100:.2f}%，SWAD为{hswa.accuracy*100:.2f}%，平均+{hp.accuracy_delta_pp:.2f}pp且{hp.positive_seeds}/3个seed同向；RX下10%分位+{hp.q10_delta_pp:.2f}pp、平均RX最差TX Recall+{hp.worst_TX_recall_delta_pp:.2f}pp。SWAD是本轮Shen最强已实现方法，收益应归于已有训练配方。其余方法在这个有限固定配方下没有平均优势，不能推断它们普遍无效。",
        f"固定残差容量对照在源域验证约95.6%，但内部RX未见日期只有{cap_control.accuracy*100:.2f}%，相对官方CE的未知RX平均变化{(cap_control.accuracy-wce.accuracy)*100:.2f}pp。该架构有4.87倍参数且显著更高成本，其源内改善没有转化为此场景泛化改善；仅一个固定容量对照不足定位原因。",
        '这些结果保留了跨RX稳定性的研发问题，但没有确证两个数据集共享同一因果机制。正式官方CNN有11个普通方法checkpoint处于99/100epoch；有限horizon与固定meta-alpha/beta、probe步长和train/eval模式差异需要先核查，不能把尚未充分收敛或配方受限归结为必须发明新模块。本轮按登记horizon结束，未追加续训。','',
        '## 三个seed的RX等权结果','',
        '| 数据/骨干 | 评价场景 | 方法 | Accuracy均值±seed SD (%) | RX下10%分位 (%) | Macro-F1 (%) | 平均RX最差TX Recall (%) |',
        '|---|---|---|---:|---:|---:|---:|']
    for row in summary.to_dict('records'):
        lines.append(f"| {row['dataset']}/{row['backbone']} | {row['role']} | {row['method']} | {row['accuracy']*100:.2f}±{row['seed_accuracy_std']*100:.2f} | {row['RX_accuracy_q10']*100:.2f} | {row['macro_f1']*100:.2f} | {row['mean_RX_worst_TX_recall']*100:.2f} |")
    lines+=['','## CE配对变化','',
        '| 数据/骨干 | 场景 | 方法 | 平均ΔAccuracy (pp) | 三seed Δ范围 (pp) | 正改善seed | 下尾Δ (pp) | 最差TX均值Δ (pp) |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in psummary.to_dict('records'):
        if r['method']=='ce':continue
        lines.append(f"| {r['dataset']}/{r['backbone']} | {r['role']} | {r['method']} | {r['accuracy_delta_pp']:.2f} | [{r['accuracy_min_delta_pp']:.2f},{r['accuracy_max_delta_pp']:.2f}] | {r['positive_seeds']}/3 | {r['q10_delta_pp']:.2f} | {r['worst_TX_recall_delta_pp']:.2f} |")
    lines+=['','## 解释范围与强基线','',
        '不从某一seed的结果直接结束研究问题。本次全部矩阵完成后比较。上表最优方法来自固定完整矩阵的描述，不用于挑选checkpoint；不能把内部开发成绩当最终盲测。WiSig剩余八RX、第四日期、等化版持续封存。Shen两折源池已大量观察，仅交叉验证/辅助确认。',
        'WiSig改为十类闭集，所以旧八类W0精度不可直接对比。Shen本轮每类160拟合/40源内验证，标准化只用160，旧B1每类200拟合不可直接同协议对比。源内完整学习曲线、搜索及所有候选保留。有限两LR而固定alpha/beta/eta的配方可能限制已有方法表现；不得把它们在该配方下的欠佳当作机制普遍失效。',
        'source_validation_vs_development.csv将每个冻结模型的源训练/验证分数、checkpoint与各部署场景分数逐seed/折配对列出，便于直接评估源域分数是否代表开发泛化；不对混合不同方法、RX与三个seed的点给出伪显著性或因果相关结论。',
        f"官方SWAD正式作业{selection_quality['formal_SWAD_jobs']}个，检测到收敛{selection_quality['converged_formal_SWAD_jobs']}个，未检测收敛而按作者末段回退{selection_quality['fallback_formal_SWAD_jobs']}个；回退不能称成功找到平坦谷。固定100/20epoch的收敛质量请结合训练/验证曲线解释，没有训练预算触发续训。",
        'CE与SWAD同LR同seed的原始训练目标和源验证轨迹必须相同，平均操作不影响CE更新。二阶MLDG相对CE成本显著更高；按相同数据/horizon比较，并同时报告成本，不称计算等价。残差CNN的CE容量对照独立列出；容量收益不归因于元学习。PCGrad只采用两个随机RX分组任务，未跑全部逐RX任务的额外变体。',
        '资源调度校正：初始控制器把慢的WiSig残差速度暂代尚未计时Shen，产生超预算替代估算，行政暂停记录完整保留。先对原矩阵每种Shen机制计时，再按各自速度及正式RX数量重估，才继续未启动的原作业；没有拟合失败或重试。完整解释见RESERVATION_ACCOUNTING.md及三份调度记录。',
        '统计单位为RX/日期/seed。无packet级p值、无三seed假设性显著性结论。源RX和开发日期均有限，数据包按历史位置切分并非独立采集session。未等化输入同时包含信道/日期等因素，不把RX差异解释为纯硬件因果。',
        '所有Macro-F1针对真值支持类，缺TX的Recall为NA；预测到缺TX仍作为错误。RX均值先等权日期再等权接收机，Shen合并两折十RX后再算分位，不平均两折的分位数。主指标平均RX最差TX Recall先对每RX×TX在有样本的日期等权平均Recall，再取该RX支持TX中的最小值，最后平均RX；逐日期最差Recall的平均另存，二者不混用。每RX/日期完整覆盖计数、混淆矩阵、每TX支持数和Recall均公开。',
        'class_index/tx0–9的身份映射见identity_labels.csv（Shen对应TX31–40），开发日期映射见date_labels.csv；Shen date=-1明确表示未知日期。复现需重建既有输入缓存或具备相同SHA输入，干净checkout本身不含数据和权重。',
        '', '## 训练过程风险','',
        '固定频率的首个平衡批次探针记录eval-mode的inner SGD候选与实际outer update，训练RNG保持不变。每RX×TX只有两包，探针为随机源RX模拟，不能当作未来未见RX的预测器，更不能证明因果机制。实际outer包含留RX元梯度，故其局部损失不是完全未参与外层优化的测试损失。RX+日期版本的探针是共同RX协议，未另做日期风险探针。',
        'update_risk_per_seed.csv给出负梯度夹角、源训练有益且留RX有害的比例，inner/outer的RX损失正增幅与类别尾部风险分别保存。所有结果只能证明该小批次确定性局部诊断现象，独立包重测、后续轨迹预测、相同训练与推理模式的检验尚未确证。',
        '完整训练风险与所选checkpoint之前轨迹分别列于update_risk_per_seed.csv和update_risk_selected_prefix.csv；若所选checkpoint早于第一次探针，该模型不具备prefix探针证据。SWAD采用平均窗口终点作为prefix边界，它不是对平均权重本身的风险测量。不能把未进入所选模型的后续更新直接当作部署风险原因。',
        '', '## 研发决定','',
        '完整R0/R1基线和源模拟风险已交付。优先保留实测最强已有方法；本轮没有新算法效果或创新性结论。R2只提交数学机制候选和区分MLDG/PCGrad/Fishr/Group DRO的必要实验；模块C须在A+B确实改善总体而仍有可重复身份尾部退化之后才考虑。R2训练、额外系数/架构搜索或最终测试需要后续用户指令。',
        '先行论文及准确的实现区别见docs/receiver_r01/PRIOR_WORK_MATRIX.md；条件R2见docs/receiver_r01/R2_DESIGN.md。历史G1/G2/VIRP/EISR/P0/C0/D0/B1/M0/receiver transform/W0/W1停止状态和文件保持不变。',
        '', '## 可复核证据','',
        '公开source_10class_coverage.csv、data/numerical_preflight.json、selection.frozen.json、paired_initialization_sampling.csv、85作业成本/完整学习曲线/源内搜索、逐RX/TX/date/seed统计、混淆counts、配对变化、局部更新风险和derived-only evidence.zip。原IQ、输入feature、预测logit缓存和权重不上传。输入SHA与公开文件SHA目录说明完整保留。']
    (public/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Curated archive includes raw process logs and all run definitions/results, never arrays/weights.
    archive=public/'evidence.zip';members=[]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_LZMA) as z:
        for stage in ['search','formal']:
            for job in reg[stage+'_jobs']:
                run=out/stage/job['run_id']
                for name in ['config.json','status.json','selection.json','history.csv','update_diagnostics.csv']:
                    p=run/name;member=f"{stage}/{job['run_id']}/{name}";z.write(p,member);members.append(dict(path=member,sha256=sha(p),bytes=p.stat().st_size))
                p=out/'logs'/(job['run_id']+'.log');member='logs/'+p.name;z.write(p,member);members.append(dict(path=member,sha256=sha(p),bytes=p.stat().st_size))
        for name in ['confusion_counts.csv','per_RX_TX_date.csv','paired_RX_TX_changes.csv','update_diagnostics.csv','update_RX_risks.csv']:
            p=public/name;member='tables/'+name;z.write(p,member);members.append(dict(path=member,sha256=sha(p),bytes=p.stat().st_size))
    dump(public/'evidence_members.json',dict(archive_sha256=sha(archive),members=members,raw_IQ=False,weights=False,feature_arrays=False))
    # Standard standalone scientific plot, exportable as PNG and PDF.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,(dataset,role) in zip(axes,[('wisig','internal_seen_date'),('wisig','internal_unseen_date'),('shen','source_pool_outer_RX')]):
        s=summary[(summary.dataset==dataset)&(summary.role==role)&(summary.backbone!='resnet')]
        ax.errorbar(np.arange(len(s)),s.accuracy*100,yerr=s.seed_accuracy_std*100,fmt='o',capsize=4,label='mean ± seed SD')
        ax.scatter(np.arange(len(s)),s.RX_accuracy_q10*100,marker='x',label='RX 10% quantile')
        ax.set_xticks(np.arange(len(s)));ax.set_xticklabels(s.method,rotation=45,ha='right');ax.set_ylabel('RX equal-weight accuracy (%)');ax.set_title(dataset+' / '+role);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(public/'baseline_comparison.png',dpi=180);fig.savefig(public/'baseline_comparison.pdf');plt.close(fig)
    print(json.dumps(dict(cost=cost,decision=decision['status'],evidence_members=len(members))))

if __name__=='__main__':main()
