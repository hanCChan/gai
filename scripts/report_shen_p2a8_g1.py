#!/usr/bin/env python3
"""Recompute predictions, paired errors and domain-level summaries before decisions."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from report_shen_p2a import read_csv,write_csv
from shen_p2a.cache import write_json

def read_prediction(path,classes):
    rows=sorted(read_csv(path/'predictions.csv'),key=lambda r:r['sample_id']);ids=[r['sample_id'] for r in rows]
    assert len(ids)==len(set(ids));y=np.array([int(r['label']) for r in rows]);p=np.array([int(r['prediction']) for r in rows])
    m=json.loads((path/'metrics.json').read_text());assert abs(m['accuracy']-(y==p).mean())<1e-9
    assert abs(m['macro_f1']-f1_score(y,p,labels=list(range(classes)),average='macro',zero_division=0))<1e-9
    np.testing.assert_array_equal(confusion_matrix(y,p,labels=list(range(classes))),np.loadtxt(path/'confusion_matrix.csv',delimiter=',',dtype=int))
    return rows,y,p,m

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pipeline',type=Path,required=True);parser.add_argument('--dataset',choices=['shen','osu'],required=True);a=parser.parse_args();out=a.pipeline/'report';out.mkdir(exist_ok=False)
    reg=json.loads((ROOT/'configs/shen_p2a8/g1_registration.json').read_text());classes=10 if a.dataset=='shen' else 25
    rows=[];runs={};values={};pairs=[];devices=[];cost=[]
    for folder in sorted((a.pipeline/'runs').iterdir()):
        cfg=json.loads((folder/'config.json').read_text());status=json.loads((folder/'status.json').read_text());assert status['status']=='completed'
        if cfg['phase']=='cv':continue
        direction=cfg['train_keys'][0] if a.dataset=='osu' else '10_training_RX'
        cost.append(dict(method=cfg['method'],seed=cfg['seed'],direction=direction,parameters=cfg['parameters'],epochs=cfg['epochs'],seconds=status['seconds'],initial_model_sha=cfg['initial_model_payload_sha256']))
        runs[direction,cfg['method'],cfg['seed']]=cfg
        for path in (folder/'evaluation').iterdir():
            v=read_prediction(path,classes);values[direction,cfg['method'],cfg['seed'],path.name]=v
            rows.append(dict(dataset=a.dataset,direction=direction,method=cfg['method'],seed=cfg['seed'],domain=path.name,epochs=cfg['epochs'],parameters=cfg['parameters'],**v[3]))
            cm=confusion_matrix(v[1],v[2],labels=list(range(classes)))
            for tx in range(classes):
                mask=v[1]==tx;tp=cm[tx,tx];n=cm[tx].sum();npred=cm[:,tx].sum()
                margin=np.array([float(r['true_class_logit_margin']) for r in v[0]])
                devices.append(dict(direction=direction,domain=path.name,method=cfg['method'],seed=cfg['seed'],raw_label=tx+(31 if a.dataset=='shen' else 1),packets=int(n),recall=float(tp/max(n,1)),f1=float(2*tp/max(n+npred,1)),margin_median=float(np.median(margin[mask]))))
    domains=sorted({r['domain'] for r in rows});directions=sorted({r['direction'] for r in rows});summary=[]
    for direction in directions:
        for domain in domains:
            for method in reg['methods']:
                subset=[r for r in rows if r['direction']==direction and r['domain']==domain and r['method']==method]
                if not subset:continue
                assert len(subset)==3 and {r['seed'] for r in subset}==set(reg['confirmation_seeds'])
                summary.append(dict(direction=direction,domain=domain,method=method,seeds=3,accuracy_mean=float(np.mean([r['accuracy'] for r in subset])),accuracy_sample_std=float(np.std([r['accuracy'] for r in subset],ddof=1)),macro_f1_mean=float(np.mean([r['macro_f1'] for r in subset])),macro_f1_sample_std=float(np.std([r['macro_f1'] for r in subset],ddof=1))))
            for baseline,comparison in (('full_iq','strict_iq'),('full_cis','strict_cis')):
                for seed in reg['confirmation_seeds']:
                    if (direction,baseline,seed,domain) not in values:continue
                    v,w=[values[direction,m,seed,domain] for m in (baseline,comparison)]
                    assert [r['sample_id'] for r in v[0]]==[r['sample_id'] for r in w[0]];np.testing.assert_array_equal(v[1],w[1])
                    assert runs[direction,baseline,seed]['initial_model_payload_sha256']==runs[direction,comparison,seed]['initial_model_payload_sha256'],'Initialization mismatch'
                    ca=v[2]==v[1];cb=w[2]==w[1]
                    for tx in [-1,*range(classes)]:
                        ix=np.ones(len(ca),dtype=bool) if tx==-1 else v[1]==tx
                        pairs.append(dict(direction=direction,domain=domain,baseline=baseline,comparison=comparison,seed=seed,raw_label='all' if tx==-1 else tx+(31 if a.dataset=='shen' else 1),packets=int(ix.sum()),corrected=int((~ca&cb&ix).sum()),new_errors=int((ca&~cb&ix).sum()),delta_accuracy_pp=float((cb[ix].mean()-ca[ix].mean())*100),macro_f1_delta=w[3]['macro_f1']-v[3]['macro_f1'] if tx==-1 else ''))
    write_csv(out/'all_domain_metrics.csv',rows);write_csv(out/'three_seed_domain_summary.csv',summary);write_csv(out/'all_TX_metrics_and_margins.csv',devices);write_csv(out/'paired_corrections_and_regressions.csv',pairs);write_csv(out/'cost_and_budget.csv',cost)
    aggregate=[]
    if a.dataset=='shen':
        for method in reg['methods']:
            for seed in reg['confirmation_seeds']:
                subset=[r for r in rows if r['method']==method and r['seed']==seed and r['domain'] in ['shen_'+rx+'_test' for rx in reg['development_receivers']]]
                assert len(subset)==4
                aggregate.append(dict(method=method,seed=seed,development_mean_accuracy=float(np.mean([r['accuracy'] for r in subset])),development_worst_accuracy=float(min(r['accuracy'] for r in subset)),development_accuracy_sd_across_RX=float(np.std([r['accuracy'] for r in subset],ddof=1)),development_mean_macro_f1=float(np.mean([r['macro_f1'] for r in subset]))))
        write_csv(out/'development_aggregate.csv',aggregate)
    table=['|Source|Evaluation domain|Baseline|Accuracy mean ± seed SD|Macro-F1|','|---|---|---|---:|---:|']
    for r in summary:
        table.append(f'|{r["direction"]}|{r["domain"]}|{r["method"]}|{r["accuracy_mean"]*100:.2f} ± {r["accuracy_sample_std"]*100:.2f}%|{r["macro_f1_mean"]:.4f}|')
    fig,axes=plt.subplots(1,len(directions),figsize=(max(12,8*len(directions)),5),squeeze=False)
    for ax,direction in zip(axes[0],directions):
        shown=[d for d in domains if a.dataset=='osu' or d in ['shen_'+rx+'_test' for rx in reg['development_receivers']]];x=np.arange(len(shown))
        for i,method in enumerate(reg['methods']):
            group=[next(r for r in summary if r['direction']==direction and r['domain']==d and r['method']==method) for d in shown]
            ax.bar(x+(i-1.5)*.18,[r['accuracy_mean']*100 for r in group],.18,yerr=[r['accuracy_sample_std']*100 for r in group],label=method,capsize=2)
        ax.set_xticks(x,shown,rotation=20,ha='right');ax.set_ylim(0,105);ax.set_ylabel('Accuracy %, seed sample SD');ax.set_title(direction);ax.legend()
    fig.tight_layout();fig.savefig(out/'cross_RX_baselines.png',dpi=180);plt.close(fig)
    notes=[f'# G1 {a.dataset} 首批跨RX强基线','', '固定四种强基线与三seed。全部预测、完整混淆、逐类别margin与纠错均独立重算。Shen用源RX两折选epoch后冻结，OSU用20固定epoch；没有用目标RX标签选择模型或scaler。', '', '\n'.join(table), '', 'root Shen采集日期/session未知，不能称同日cross-RX；Day1/2 drift另列。B200/N210为训练未含同型号的外推，RTL4/8是同型号实例转移，不能只看四RX均值。OSU是独立采集团队；该subset是否曾参与旧方法选择仍需执行历史核实，当前只作开发复现，不称研究史全新盲测。连续周期窗口不等于严格同步前导，所有结果均为开发机制证据。3seed只描述训练随机性，两OSU接收机不等于接收机总体。','', 'G1需要联合Shen与OSU趋势判定。本单数据集报告不宣布通用频带规律；G2源域预测关系、G3新方法/GRL/对比/FAN比较、G4跨体制外部验证尚未完成。未启动旧P2-B/P3。']
    (out/'REPORT.md').write_text('\n'.join(notes)+'\n')
    write_json(out/'verification.json',dict(passed=True,models_checked=len(runs),domain_prediction_sets_checked=len(values),source_developer_roles_enforced=True,paired_ids_labels_exact=True,full_strict_initial_weights_exact=True,locked_accessed=False,G2_G3_started=False))
    if not (ROOT/f'outputs/shen_p2a8/g1_{a.dataset}_bands_v1').exists():
        import subprocess
        subprocess.run([sys.executable,'scripts/diagnose_shen_p2a8_g1_bands.py','--dataset',a.dataset],cwd=ROOT,check=True)
    import subprocess
    subprocess.run([sys.executable,'scripts/diagnose_shen_p2a8_g1_optimization.py','--pipeline',str(a.pipeline)],cwd=ROOT,check=True)
