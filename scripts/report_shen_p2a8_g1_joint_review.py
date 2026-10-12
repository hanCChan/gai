#!/usr/bin/env python3
"""Joint evidence and next-stage gate from unchanged G1 predictions, not retuned models."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from shen_p2a.cache import write_json
from shen_p2a8.g1_data import cache,sha
from report_shen_p2a import read_csv,write_csv

def joint(shen_review,osu_review,osu_pipeline,out):
    out.mkdir(parents=True,exist_ok=False)
    models={d:read_csv(path/'model_source_learning_and_collapse.csv') for d,path in [('shen',shen_review),('osu',osu_review)]}
    pairs={d:read_csv(path/'CIS_source_success_conditioned_pairs.csv') for d,path in [('shen',shen_review),('osu',osu_review)]}
    targets={d:read_csv(path/'per_seed_target_metrics.csv') for d,path in [('shen',shen_review),('osu',osu_review)]}
    summary=[]
    for dataset in ('shen','osu'):
        for direction,domain in sorted({(r['direction'],r['domain']) for r in pairs[dataset]}):
            group=[r for r in pairs[dataset] if r['direction']==direction and r['domain']==domain];assert len(group)==3
            full=[r for r in targets[dataset] if r['direction']==direction and r['domain']==domain and r['method']=='full_cis']
            strict=[r for r in targets[dataset] if r['direction']==direction and r['domain']==domain and r['method']=='strict_cis']
            valid=[r for r in group if r['both_source_tasks_learned']=='True']
            summary.append(dict(dataset=dataset,direction=direction,target=domain,full_cis_accuracy_mean=float(np.mean([float(r['accuracy']) for r in full])),strict_cis_accuracy_mean=float(np.mean([float(r['accuracy']) for r in strict])),
                full_cis_f1_mean=float(np.mean([float(r['macro_f1']) for r in full])),strict_cis_f1_mean=float(np.mean([float(r['macro_f1']) for r in strict])),
                paired_accuracy_delta_pp_mean=float(np.mean([float(r['accuracy_delta_pp']) for r in group])),paired_f1_delta_mean=float(np.mean([float(r['macro_f1_delta']) for r in group])),
                accuracy_and_f1_positive_pairs=sum(float(r['accuracy_delta_pp'])>0 and float(r['macro_f1_delta'])>0 for r in group),all_pairs=3,both_source_tasks_learned_pairs=len(valid),
                source_success_conditioned_f1_delta_mean=float(np.mean([float(r['macro_f1_delta']) for r in valid])) if valid else None))
    osu_rows=[r for r in summary if r['dataset']=='osu'];assert len(osu_rows)==2
    all_cis_learned=all(r['source_task_learned']=='True' for r in models['osu'] if r['method'] in ('full_cis','strict_cis'))
    means=[r['paired_f1_delta_mean'] for r in osu_rows]
    positive=all_cis_learned and all(r['paired_f1_delta_mean']>0 and r['accuracy_and_f1_positive_pairs']>=2 and r['both_source_tasks_learned_pairs']==3 for r in osu_rows)
    if not all_cis_learned:decision='source_optimization_or_learning_failure_prevents_clean_frequency_inference'
    elif positive:decision='positive_external_replication_allows_exploratory_G2_0'
    elif all(v<0 for v in means):decision='both_directions_negative_stop_universal_fixed_filter_claim'
    elif min(means)<0<max(means):decision='direction_dependent_effect_prioritize_conditional_mechanism'
    else:decision='insufficient_replicated_gain_G2_0_not_automatically_started'
    severe=[]
    for dataset,path in [('shen',shen_review),('osu',osu_review)]:
        rows=read_csv(path/'paired_per_TX_recall_margin_errors.csv')
        severe.extend(dict(dataset=dataset,**r) for r in rows if r['comparison']=='strict_cis')
    severe=sorted(severe,key=lambda r:float(r['recall_delta_pp']));write_csv(out/'worst_TX_regressions.csv',severe[:30])
    counts=[]
    for rx in ('RX1','RX2'):
        info=json.loads((cache(ROOT,'osu_'+rx)/'manifest.json').read_text())
        for source in info['source']:counts.append(dict(receiver=rx,raw_label=source['raw_label'],eligible_periodic_windows=source['eligible_windows'],selected_windows=source['selected_windows'],source_sha256=source['sha256'],source_path=source['path']))
    write_csv(out/'OSU_actual_per_TX_windows.csv',counts);write_csv(out/'Full_Strict_CIS_direction_and_source_success.csv',summary)
    write_csv(out/'all_model_source_learning.csv',[{**r,'dataset':d} for d in models for r in models[d]])
    write_csv(out/'all_CIS_seed_pairs.csv',[dict(dataset=d,**r) for d in pairs for r in pairs[d]])
    write_json(out/'decision.json',dict(status='completed',G1_external_replication=decision,G2_0_allowed=positive,G2_1_allowed=False,G3_started=False,old_P2_B_P3_started=False,
        universal_physical_frequency_mechanism_proven=False,OSU_CIS_source_tasks_learned=all_cis_learned,source_task_screen_is_not_full_optimization_proof=True,
        frozen_G1_registration_sha256=sha(ROOT/'configs/shen_p2a8/g1_registration.json'),OSU_source_variance_available=False,failed_seeds_preserved=True))
    table=['|Dataset /方向|Full CIS Acc|Strict CIS Acc|配对ΔF1|正向seed|源任务均学会|','|---|---:|---:|---:|---:|---:|']+[f'|{r["dataset"]} {r["direction"]}→{r["target"]}|{r["full_cis_accuracy_mean"]*100:.2f}%|{r["strict_cis_accuracy_mean"]*100:.2f}%|{r["paired_f1_delta_mean"]:+.4f}|{r["accuracy_and_f1_positive_pairs"]}/3|{r["both_source_tasks_learned_pairs"]}/3|' for r in summary]
    worst=severe[0]
    (out/'REPORT.md').write_text('\n'.join(['# G1 联合复核与执行决策','','\n'.join(table),'', '操作关卡：'+decision+'。G2-0允许='+str(positive)+'；G2-1/G3未启动。关卡是后续资源/研究决策，不是显著性检验或普适物理机制证明。', '',
        '最严重CIS类别回退：'+str({k:worst[k] for k in ('dataset','domain','seed','raw_label','baseline_recall','comparison_recall','new_errors','both_source_tasks_learned')})+'。完整误判去向见各数据集new_error_destinations.csv，不能只看平均Accuracy或TX34。', '',
        '所有失败seed作为主结果保留；source_success_conditioned分组只是解释，不替换主结果。Shen十类随机10%，OSU二十五类随机4%。源任务筛查与全零head证据不同，若无法拟合源任务不能将低分单独归因为频谱过滤。Shen Full/Strict CIS源CV分别选16/19epoch且scaler不同，比较的是训练流程而非纯OOB因果；OSU固定20epoch，scaler仍按源输入各自拟合。','',
        'OSU窗口数详见OSU_actual_per_TX_windows.csv，周期窗不是严格同步packet。两个方向共享同一对RX，不是两个独立硬件环境样本；每方向只有一个源RX，V_RX缺失。Shen root实际日期/session未知；OSU来自独立采集团队，但研究史留出独立性不能夸大。','',
        '训练曲线/失败日志/权重保留在'+str(osu_pipeline)+'/runs和logs。固定频带的效益可因方向、TX和优化不同，尚不能命名为新算法。若允许G2-0，必须先冻结九源拟合、一留出评价，以及源分类准确率负对照；不能使用全十源训练的CNN制造留一泄漏。源域风险是否可预测尚未由G1证明。'])+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--shen-review',type=Path,required=True);p.add_argument('--osu-review',type=Path,required=True);p.add_argument('--osu-pipeline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();joint(a.shen_review,a.osu_review,a.osu_pipeline,a.output)
