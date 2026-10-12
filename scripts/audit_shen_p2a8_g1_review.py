#!/usr/bin/env python3
"""G1-S/G1-O reporting-only audit; preserve frozen optimization and all failed seeds."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from shen_p2a8.g1_data import G1Dataset,fit_scaler,sha
from train_shen_p2a8_g1 import make,prepare
from report_shen_p2a8_g1 import read_prediction
from report_shen_p2a import read_csv,write_csv
from shen_p2a.cache import write_json

def audit(pipeline,dataset,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    reg=json.loads((ROOT/'configs/shen_p2a8/g1_registration.json').read_text());classes=10 if dataset=='shen' else 25
    domains=['shen_'+r+'_test' for r in reg['development_receivers']] if dataset=='shen' else ['osu_RX1','osu_RX2']
    model_rows=[];prediction_rows=[];paired=[];errors=[];source_audits={}
    for folder in sorted((pipeline/'runs').iterdir()):
        cfg=json.loads((folder/'config.json').read_text());status=json.loads((folder/'status.json').read_text());assert status['status']=='completed'
        if cfg['phase']=='cv':continue
        history=read_csv(folder/'history.csv');last=history[-1];cp=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
        model=make(cfg['method'],classes).eval();model.load_state_dict(cp['model']);zs=[];preds=[];ys=[]
        for key in cfg['train_keys']:
            ds=G1Dataset(ROOT,[key],cfg['method'],cfg['scaler']);ix=[int(np.flatnonzero(ds.labels_all==c)[0]) for c in range(classes)]
            batch=[ds[i] for i in ix];x=prepare(torch.from_numpy(np.stack([b[0] for b in batch])),cfg['method'],'cpu')
            capture=[];layer=model.head[-1] if cfg['method'].endswith('iq') else model.classifier
            h=layer.register_forward_pre_hook(lambda module,inputs:capture.append(inputs[0].detach().numpy()))
            with torch.no_grad():logits=model(x)
            h.remove();zs.append(capture[0]);preds.extend(logits.argmax(1).tolist());ys.extend([b[1] for b in batch])
        z=np.concatenate(zs);train_acc=float(last['train_accuracy']);loss=float(last['train_loss']);chance=1/classes
        confirmed_collapse=bool((z==0).mean()>.999 and np.var(z,axis=0).mean()<1e-12 and len(set(preds))==1 and train_acc<=chance+.05)
        source_learned=bool(train_acc>chance+.05 and not confirmed_collapse)
        source_key=cfg['train_keys'][0] if dataset=='osu' else '10_training_RX'
        source_eval_path=folder/'evaluation'/source_key
        source_eval_accuracy=json.loads((source_eval_path/'metrics.json').read_text())['accuracy'] if source_eval_path.exists() else None
        row=dict(run=folder.name,dataset=dataset,direction=source_key,method=cfg['method'],seed=cfg['seed'],epochs=cfg['epochs'],classes=classes,chance_accuracy=chance,
            final_train_accuracy=train_acc,final_train_loss=loss,near_chance_epoch_count=sum(float(r['train_accuracy'])<=chance+.05 for r in history),
            source_probe_accuracy=float((np.array(preds)==ys).mean()),source_probe_samples=len(ys),source_embedding_zero_fraction=float((z==0).mean()),
            source_embedding_variance=float(np.var(z,axis=0).mean()),source_probe_distinct_predictions=len(set(preds)),confirmed_source_collapse=confirmed_collapse,
            source_task_learned=source_learned,source_evaluation_accuracy=source_eval_accuracy,checkpoint_sha256=sha(folder/'final.pt'),initial_model_sha256=cfg['initial_model_payload_sha256'])
        model_rows.append(row);source_audits[source_key,cfg['method'],cfg['seed']]=row
        for domain in domains:
            if dataset=='osu' and domain==source_key:continue
            p=folder/'evaluation'/domain
            if not p.exists():continue
            records,y,pred,m=read_prediction(p,classes)
            prediction_rows.append(dict(direction=source_key,domain=domain,method=cfg['method'],seed=cfg['seed'],windows=m['packets'],accuracy=m['accuracy'],macro_f1=m['macro_f1'],source_task_learned=source_learned,confirmed_source_collapse=confirmed_collapse))
    groups={}
    for r in prediction_rows:groups[r['direction'],r['domain'],r['method'],r['seed']]=r
    for direction,domain in sorted({(r['direction'],r['domain']) for r in prediction_rows}):
        for seed in reg['confirmation_seeds']:
            for base,comparison in [('full_cis','strict_cis'),('full_iq','strict_iq')]:
                rb,rc=[groups[direction,domain,m,seed] for m in (base,comparison)]
                fb=next(f for f in (pipeline/'runs').iterdir() if f.name==source_audits[direction,base,seed]['run'])
                fc=next(f for f in (pipeline/'runs').iterdir() if f.name==source_audits[direction,comparison,seed]['run'])
                a,b=[read_prediction(f/'evaluation'/domain,classes) for f in (fb,fc)]
                assert [r['sample_id'] for r in a[0]]==[r['sample_id'] for r in b[0]];np.testing.assert_array_equal(a[1],b[1])
                assert source_audits[direction,base,seed]['initial_model_sha256']==source_audits[direction,comparison,seed]['initial_model_sha256']
                for c in range(classes):
                    ix=a[1]==c;ca=a[2]==a[1];cb=b[2]==b[1];ma=np.array([float(r['true_class_logit_margin']) for r in a[0]]);mb=np.array([float(r['true_class_logit_margin']) for r in b[0]])
                    paired.append(dict(direction=direction,domain=domain,seed=seed,baseline=base,comparison=comparison,raw_label=c+(31 if dataset=='shen' else 1),windows=int(ix.sum()),
                        baseline_recall=float(ca[ix].mean()),comparison_recall=float(cb[ix].mean()),recall_delta_pp=float((cb[ix].mean()-ca[ix].mean())*100),corrected=int((~ca&cb&ix).sum()),new_errors=int((ca&~cb&ix).sum()),
                        baseline_margin_median=float(np.median(ma[ix])),comparison_margin_median=float(np.median(mb[ix])),both_source_tasks_learned=rb['source_task_learned'] and rc['source_task_learned']))
                    for destination in range(classes):
                        selected=ix&ca&~cb&(b[2]==destination)
                        if selected.any():errors.append(dict(direction=direction,domain=domain,seed=seed,baseline=base,comparison=comparison,raw_label=c+(31 if dataset=='shen' else 1),new_prediction_raw_label=destination+(31 if dataset=='shen' else 1),new_error_count=int(selected.sum())))
    write_csv(out/'model_source_learning_and_collapse.csv',model_rows);write_csv(out/'per_seed_target_metrics.csv',prediction_rows);write_csv(out/'paired_per_TX_recall_margin_errors.csv',paired);write_csv(out/'new_error_destinations.csv',errors)
    adverse=sorted([r for r in paired if r['comparison']=='strict_cis'],key=lambda r:r['recall_delta_pp'])[:20];write_csv(out/'worst_CIS_TX_regressions.csv',adverse)
    cis_pairs=[]
    for direction,domain in sorted({(r['direction'],r['domain']) for r in prediction_rows}):
        for seed in reg['confirmation_seeds']:
            a,b=[groups[direction,domain,m,seed] for m in ('full_cis','strict_cis')]
            cis_pairs.append(dict(direction=direction,domain=domain,seed=seed,accuracy_delta_pp=100*(b['accuracy']-a['accuracy']),macro_f1_delta=b['macro_f1']-a['macro_f1'],both_source_tasks_learned=a['source_task_learned'] and b['source_task_learned'],full_source_collapse=a['confirmed_source_collapse'],strict_source_collapse=b['confirmed_source_collapse']))
    write_csv(out/'CIS_source_success_conditioned_pairs.csv',cis_pairs)
    write_json(out/'status.json',dict(status='completed',dataset=dataset,models_checked=len(model_rows),prediction_sets_checked=len(prediction_rows),training_changed=False,failed_seeds_removed=False,classes=classes,chance_accuracy=chance,CIS_positive_pairs=sum(r['accuracy_delta_pp']>0 for r in cis_pairs),CIS_pairs=len(cis_pairs)))
    table=['|方向/开发RX|Seed|ΔAccuracy pp|ΔMacro-F1|两个源任务均学会|','|---|---:|---:|---:|---|']+[f'|{r["direction"]}→{r["domain"]}|{r["seed"]}|{r["accuracy_delta_pp"]:+.2f}|{r["macro_f1_delta"]:+.4f}|{r["both_source_tasks_learned"]}|' for r in cis_pairs]
    (out/'REPORT.md').write_text('\n'.join(['# G1 逐RX、逐seed、逐TX与源任务有效性审计','','只分析已保存checkpoint/预测，不修改训练。全部失败保留，条件分组单列。source_task_learned仅为工程筛查（最后epoch源训练准确率高于随机+0.05且无确认全零塌缩），不能证明模型优化完全充分或诊断早期根因。类数为'+str(classes)+'，随机水平'+str(chance)+'。','','\n'.join(table),'','最严重类别回退及误判去向见worst_CIS_TX_regressions.csv/new_error_destinations.csv。两个源任务学会时的配对仍可能受scaler、epoch、BN和优化轨迹影响，不能当作纯删除OOB因果。11/12之类计数不是独立环境显著性检验。源训练probe仅检查每类首个样本，不能替代完整训练曲线或全源评估。'])+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pipeline',type=Path,required=True);p.add_argument('--dataset',choices=['shen','osu'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.pipeline.resolve(),a.dataset,a.output.resolve())
