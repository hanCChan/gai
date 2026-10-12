#!/usr/bin/env python3
"""Source LORO diagnostic with inner-source receiver CV as the accuracy-only control."""
import argparse,datetime,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from shen_p2a8.g1_data import cache,sha
from shen_p2a8.reliability import fixed_cis_moments,band_columns,fit_scaler,scale,fit_ridge,ridge_logits,source_statistics,rank_correlation
from shen_p2a.cache import write_json
from report_shen_p2a import write_csv

def features(key,representation):
    folder=cache(ROOT,key);m=json.loads((folder/'manifest.json').read_text());assert m['status']=='completed'
    x=fixed_cis_moments(np.load(folder/(representation+'.npy'),mmap_mode='r'))
    return x,np.load(folder/'labels.npy'),np.load(folder/'rows.npy')

def metric(y,logits):
    pred=logits.argmax(1)
    return dict(accuracy=float((pred==y).mean()),macro_f1=float(f1_score(y,pred,labels=list(range(10)),average='macro',zero_division=0))),pred

def run(out):
    gate=ROOT/'outputs/shen_p2a8/g1_joint_review_v2/decision.json';decision=json.loads(gate.read_text());assert decision['G2_0_allowed'] is True
    out.mkdir(parents=True,exist_ok=False);reg=json.loads((ROOT/'configs/shen_p2a8/g1_registration.json').read_text())
    registry=dict(stage='G2-0 exploratory statistical LORO',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        G1_decision_sha256=sha(gate),receivers=reg['train_receivers'],representations=['full_cis','strict_cis'],
        feature='fixed per-STFT-row temporal mean+population SD, 104D; no learned ten-RX CNN features',
        bands=4,band_rows=[list(range(b*13,(b+1)*13)) for b in range(4)],classifier='source standardized ridge one-hot regression, alpha1, intercept unpenalized',
        reliability='D_between/(V_RX_given_TX+1e-8)',primary_negative_control='nine-source internal two receiver-group-fold mean per-RX classification ACCURACY',
        additional_negative_control='same source-group CV Macro-F1; retained stronger performance-only comparator',
        outer_unit='one of ten physical named source RX; fit all statistics and classifiers on the other nine',
        target_selection_forbidden=True,developer_receivers_accessed=False,locked_accessed=False,deep_training_runs=0,
        positive_engineering_gate=dict(mean_rank_correlation_advantage_min=.05,top_choice_mean_macro_f1_gain_min=.03,minimum_positive_RX=6,both_accuracy_and_F1_controls_must_be_exceeded=True),
        gate_not_a_significance_test=True,G3_started=False)
    write_json(out/'registration.json',registry)
    state=lambda status,**kw:write_json(out/'status.json',dict(status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
    state('running');results=[];choices=[];corrs=[];txrows=[];full_models=[];candidate_sources=[]
    for heldout in registry['receivers']:
        source_rx=[r for r in registry['receivers'] if r!=heldout];source_keys=['shen_'+r+'_train' for r in source_rx];fold=out/heldout;fold.mkdir();modeldir=fold/'source_models';modeldir.mkdir()
        candidates=[];whole=[]
        for representation in registry['representations']:
            groups=[features(key,representation) for key in source_keys];x=np.concatenate([g[0] for g in groups]);y=np.concatenate([g[1] for g in groups]);domains=np.concatenate([np.full(len(g[1]),i) for i,g in enumerate(groups)])
            assert len(x)==18000 and set(y)==set(range(10))
            for band in range(4):
                columns=band_columns(band);raw=x[:,columns];outer_scale=fit_scaler(raw);z=scale(raw,outer_scale);weights=fit_ridge(z,y)
                stats=source_statistics(z,y,domains);scores=[];accuracy_scores=[]
                # Same nine source RX only; every inner fold fits its own scalar/vector statistics.
                for inner in (np.arange(9)[::2],np.arange(9)[1::2]):
                    validation=np.isin(domains,inner);training=~validation
                    inner_scale=fit_scaler(raw[training]);inner_weights=fit_ridge(scale(raw[training],inner_scale),y[training])
                    for d in inner:
                        ids=domains==d;value,_=metric(y[ids],ridge_logits(scale(raw[ids],inner_scale),inner_weights));scores.append(value['macro_f1']);accuracy_scores.append(value['accuracy'])
                train_value,_=metric(y,ridge_logits(z,weights))
                identifier=f'{representation}_band{band}'
                np.savez(modeldir/(identifier+'.npz'),mean=outer_scale['mean'],std=outer_scale['std'],weights=weights,columns=columns)
                candidate=dict(heldout_receiver=heldout,source_receivers=source_rx,candidate=identifier,representation=representation,band=band,
                    inner_source_CV_macro_f1=float(np.mean(scores)),inner_source_CV_accuracy=float(np.mean(accuracy_scores)),source_resubstitution_macro_f1=train_value['macro_f1'],source_resubstitution_accuracy=train_value['accuracy'],**stats)
                candidates.append(candidate);candidate_sources.append({k:v for k,v in candidate.items() if k!='source_receivers'})
            whole_scale=fit_scaler(x);whole_weights=fit_ridge(scale(x,whole_scale),y)
            np.savez(modeldir/(representation+'_whole.npz'),mean=whole_scale['mean'],std=whole_scale['std'],weights=whole_weights)
            whole.append(representation)
        score=lambda r:float(r['reliability_score']) if r['reliability_score'] is not None else -np.inf
        selected_reliability=max(candidates,key=lambda r:(score(r),-registry['representations'].index(r['representation']),-r['band']))
        selected_accuracy=max(candidates,key=lambda r:(r['inner_source_CV_accuracy'],-registry['representations'].index(r['representation']),-r['band']))
        selected_f1=max(candidates,key=lambda r:(r['inner_source_CV_macro_f1'],-registry['representations'].index(r['representation']),-r['band']))
        frozen=dict(heldout_receiver=heldout,source_receivers=source_rx,source_keys=source_keys,heldout_key='shen_'+heldout+'_train',
            selected_reliability=selected_reliability['candidate'],selected_accuracy_only=selected_accuracy['candidate'],selected_source_F1_only=selected_f1['candidate'],candidates=candidates,
            frozen_before_heldout_evaluation_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_manifest_sha256={key:sha(cache(ROOT,key)/'manifest.json') for key in source_keys})
        assert heldout not in source_rx
        write_json(fold/'source_selection.frozen.json',frozen)
        # Heldout signal/labels enter only after source scalers, coefficients and choices are saved.
        target_results={};target_labels=None;target_rows=None
        for representation in registry['representations']:
            xx,yy,ids=features('shen_'+heldout+'_train',representation)
            if target_labels is None:target_labels=yy;target_rows=ids
            else:np.testing.assert_array_equal(yy,target_labels);np.testing.assert_array_equal(ids,target_rows)
            for band in range(4):
                identifier=f'{representation}_band{band}';saved=np.load(modeldir/(identifier+'.npz'));columns=saved['columns']
                logits=ridge_logits((xx[:,columns]-saved['mean'])/saved['std'],saved['weights']);value,pred=metric(yy,logits)
                candidate=next(r for r in candidates if r['candidate']==identifier)
                row=dict(heldout_receiver=heldout,candidate=identifier,representation=representation,band=band,packets=len(yy),**value,
                    reliability_score=candidate['reliability_score'],source_CV_macro_f1=candidate['inner_source_CV_macro_f1'],source_CV_accuracy=candidate['inner_source_CV_accuracy'])
                results.append(row);target_results[identifier]=row
                np.savetxt(fold/(identifier+'_confusion.csv'),confusion_matrix(yy,pred,labels=list(range(10))),fmt='%d',delimiter=',')
                write_csv(fold/(identifier+'_predictions.csv'),[dict(sample_id=f'shen_{heldout}_train:row{int(ids[i])}',label=int(yy[i]),prediction=int(pred[i]),true_class_margin=float(logits[i,yy[i]]-np.max(np.delete(logits[i],yy[i])))) for i in range(len(yy))])
                for c in range(10):txrows.append(dict(heldout_receiver=heldout,candidate=identifier,raw_label=c+31,packets=int((yy==c).sum()),recall=float((pred[yy==c]==c).mean())))
            saved=np.load(modeldir/(representation+'_whole.npz'));value,_=metric(yy,ridge_logits((xx-saved['mean'])/saved['std'],saved['weights']));full_models.append(dict(heldout_receiver=heldout,representation=representation,features=104,**value))
        names=[r['candidate'] for r in candidates];true=[target_results[n]['macro_f1'] for n in names]
        corr_r=rank_correlation([score(r) for r in candidates],true);corr_a=rank_correlation([r['inner_source_CV_accuracy'] for r in candidates],true);corr_f=rank_correlation([r['inner_source_CV_macro_f1'] for r in candidates],true)
        corrs.append(dict(heldout_receiver=heldout,reliability_vs_actual_F1_spearman=corr_r,source_CV_accuracy_vs_actual_F1_spearman=corr_a,source_CV_F1_vs_actual_F1_spearman=corr_f,
            rank_correlation_advantage=corr_r-corr_a if corr_r is not None and corr_a is not None else None,
            rank_correlation_advantage_vs_source_F1=corr_r-corr_f if corr_r is not None and corr_f is not None else None))
        a=target_results[selected_reliability['candidate']];b=target_results[selected_accuracy['candidate']];c=target_results[selected_f1['candidate']]
        choices.append(dict(heldout_receiver=heldout,reliability_choice=a['candidate'],source_accuracy_choice=b['candidate'],source_F1_choice=c['candidate'],reliability_choice_macro_f1=a['macro_f1'],source_accuracy_choice_macro_f1=b['macro_f1'],source_F1_choice_macro_f1=c['macro_f1'],macro_f1_delta=a['macro_f1']-b['macro_f1'],macro_f1_delta_vs_source_F1=a['macro_f1']-c['macro_f1'],same_choice=a['candidate']==b['candidate']))
        state('running',outer_RX_completed=len(choices),total_RX=10);print(heldout,'completed',choices[-1],flush=True)
    write_csv(out/'source_only_candidate_scores.csv',candidate_sources);write_csv(out/'heldout_candidate_metrics.csv',results);write_csv(out/'source_predictor_rank_correlations.csv',corrs);write_csv(out/'source_only_selection_vs_actual_performance.csv',choices);write_csv(out/'heldout_per_TX_recall.csv',txrows);write_csv(out/'Full_Strict_CIS_whole_moment_baselines.csv',full_models)
    usable=[r for r in corrs if r['rank_correlation_advantage'] is not None];mean_adv=float(np.mean([r['rank_correlation_advantage'] for r in usable])) if usable else None
    gain=float(np.mean([r['macro_f1_delta'] for r in choices]));positive=sum(r['macro_f1_delta']>0 for r in choices)
    usable_f=[r for r in corrs if r['rank_correlation_advantage_vs_source_F1'] is not None];mean_adv_f=float(np.mean([r['rank_correlation_advantage_vs_source_F1'] for r in usable_f])) if usable_f else None
    gain_f=float(np.mean([r['macro_f1_delta_vs_source_F1'] for r in choices]));positive_f=sum(r['macro_f1_delta_vs_source_F1']>0 for r in choices)
    allowed=mean_adv is not None and mean_adv_f is not None and len(usable)==len(usable_f)==10 and mean_adv>=.05 and mean_adv_f>=.05 and gain>=.03 and gain_f>=.03 and min(positive,positive_f)>=6
    decision=dict(status='completed',outer_RX=10,source_RX_per_fold=9,usable_rank_correlations=len(usable),mean_rank_advantage_vs_source_accuracy_control=mean_adv,
        mean_top_selection_F1_gain_vs_source_accuracy_control=gain,positive_RX=positive,mean_rank_advantage_vs_source_F1_control=mean_adv_f,mean_top_selection_F1_gain_vs_source_F1_control=gain_f,positive_RX_vs_source_F1=positive_f,G2_0_positive_engineering_gate=allowed,G2_1_allowed=allowed,
        G3_started=False,old_P2_B_P3_started=False,developer_receivers_accessed=False,locked_accessed=False,deep_training_runs=0,
        result_scope='predictability for frozen linear classifiers of fixed CIS moments, not CNN failure prediction or proof of universal mechanism')
    write_json(out/'decision.json',decision)
    fig,axes=plt.subplots(1,2,figsize=(12,4));x=np.arange(10)
    axes[0].bar(x-.18,[r['reliability_vs_actual_F1_spearman'] if r['reliability_vs_actual_F1_spearman'] is not None else np.nan for r in corrs],.36,label='Reliability score')
    axes[0].bar(x+.18,[r['source_CV_accuracy_vs_actual_F1_spearman'] if r['source_CV_accuracy_vs_actual_F1_spearman'] is not None else np.nan for r in corrs],.36,label='Source-only CV Accuracy');axes[0].set_ylim(-1,1);axes[0].set_ylabel('Spearman rank correlation')
    axes[1].bar(x,[r['macro_f1_delta'] for r in choices]);axes[1].axhline(0,color='k',linewidth=.7);axes[1].set_ylabel('Top-choice F1: reliability minus source CV control')
    for ax in axes:ax.set_xticks(x,registry['receivers'],rotation=30,ha='right')
    axes[0].legend();fig.tight_layout();fig.savefig(out/'source_only_predictive_validity.png',dpi=180);plt.close(fig)
    table=['|留出RX|S与真实F1秩相关|源CV Accuracy与真实F1秩相关|源CV F1与真实F1秩相关|选择ΔF1（相对Accuracy）|','|---|---:|---:|---:|---:|']+[f'|{r["heldout_receiver"]}|{r["reliability_vs_actual_F1_spearman"]}|{r["source_CV_accuracy_vs_actual_F1_spearman"]}|{r["source_CV_F1_vs_actual_F1_spearman"]}|{choices[i]["macro_f1_delta"]:+.4f}|' for i,r in enumerate(corrs)]
    (out/'REPORT.md').write_text('\n'.join(['# G2-0 九源拟合、一RX留出预测性诊断','','使用固定CIS每行时间mean/SD与固定ridge统计分类器，没有借用十源CNN权重。每折九源拟合scaler/D/V/分类器与源内分组CV负对照，所有选择写入source_selection.frozen.json后才读取留出RX评价。四个等宽13行STFT频带×Full/Strict共八候选，目标标签不选频带。','','\n'.join(table),'',f'平均秩相关优势={mean_adv}；可靠性选带相对源CV选带平均ΔF1={gain:+.4f}；正收益RX={positive}/10。操作关卡通过={allowed}。','',
        '若未通过，当前D/(V+epsilon)指标没有显示超越源分类表现负对照的充分预测价值，不为它开发新损失。此失败不等于任何稳定性概念都不成立，也不证明学到的CNN表征不可预测。若通过，也只能进入统一优化强基线检验，不能直接宣称G3创新。','',
        '这是首轮探索性统计代理，不是端到端CNN的新训练或CNN失败预测验证。频带坐标来自生成STFT（Strict中的带外行可含局部窗泄漏，不含原始被排除FFT信息）。十RX共享训练源且同型号不独立，不能把八十候选当独立环境；不输出packet显著性概率。源CV负对照比重代入准确率更严格。源样本预算仍每TX/RX200，研究史使用披露不变。补充Source-CV Macro-F1对照平均秩相关优势='+str(mean_adv_f)+'、选择ΔF1='+str(gain_f)+'。','', 'v1为源CV F1负对照版本，保留；v2按用户原文加入Accuracy字面负对照，并保留F1为附加更严格对照。这是对照完整性修正，不修改频带、classifier、alpha或数据，也不丢弃v1结果。'])+'\n')
    state('completed',decision=decision)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/shen_p2a8/g2_0_v2');a=p.parse_args();run(a.output.resolve())
