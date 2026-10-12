#!/usr/bin/env python3
"""Sealed CPU-only E0/E1/E2; no expert training or external RX access."""
import argparse
import ast
import concurrent.futures
import csv
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax, logsumexp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from eisr.diagnostics import FEATURES, convex_oracle, metric, observables, receiver_splits


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def sealed(all_inputs=False):
    path=ROOT/'configs/eisr/v1_registration.json'
    reg=json.loads(path.read_text());seal=json.loads(path.with_suffix('.seal.json').read_text())
    assert sha(path)==seal['registration_sha256'],'Changed registration'
    for p,digest in reg['immutable_code_sha256'].items():assert sha(ROOT/p)==digest,'Changed code: '+p
    if all_inputs:
        for collection in ('asset_sha256','cache_sha256'):
            for p,digest in reg[collection].items():assert sha(ROOT/p)==digest,'Changed input: '+p
    return reg,seal


def provenance(reg,out):
    import torch
    from shen_p2a8.g1_data import fit_scaler
    prior_path=ROOT/'configs/shen_p2a8/g1_registration.json'
    assert sha(prior_path)==reg['prior_registration_sha256']
    prior=json.loads(prior_path.read_text())
    script_path=ROOT/'scripts/train_shen_p2a8_g1.py'
    current=script_path.read_text()
    run_node=next(n for n in ast.parse(current).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    loops=[n for n in ast.walk(run_node) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch']
    assert len(loops)==1
    assert ast.unparse(loops[0].iter)=="range(1, cfg['epochs'] + 1)"
    assert not any(isinstance(n,(ast.Break,ast.Return)) for n in ast.walk(loops[0]))
    assert 'lr=reg[\'lr\'],weight_decay=reg[\'weight_decay\']' in current
    assert 'scheduler' not in current and 'optimizer.param_groups' not in current
    assert "torch.save(dict(model=model.state_dict(),config=cfg,epoch=cfg['epochs']),out/'final.pt')" in current
    historical={}
    for revision in ('5661a32','8b81fbb','HEAD'):
        text=subprocess.check_output(['git','show',revision+':scripts/train_shen_p2a8_g1.py'],cwd=ROOT,text=True)
        assert 'scheduler' not in text and "for epoch in range(1,cfg['epochs']+1):" in text
        assert "epoch=cfg['epochs']),out/'final.pt'" in text
        historical[revision]=hashlib.sha256(text.encode()).hexdigest()
    checks=[]
    for run in reg['runs']:
        folder=ROOT/reg['pipeline']/'runs'/run
        cfg=json.loads((folder/'config.json').read_text())
        status=json.loads((folder/'status.json').read_text())
        cp_path=folder/'final.pt'
        if not cp_path.exists():raise ValueError('No actual final checkpoint: '+run)
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        assert cp['epoch']==20 and cfg['epochs']==20 and status['final_epoch']==20 and status['status']=='completed'
        assert cp['config']==cfg and cfg['phase']=='cv' and cfg['classes']==10 and cfg['seed']==20261012
        assert cfg['registration_sha256']==reg['prior_registration_sha256']
        heldout=reg['source_receiver_folds'][int(run[2])]
        assert cfg['validation_keys']==['shen_'+r+'_train' for r in heldout]
        assert set(cfg['train_keys'])=={'shen_'+r+'_train' for r in reg['source_receivers'] if r not in heldout}
        assert not set(cfg['train_keys'])&set(cfg['validation_keys'])
        assert cfg['scaler']['fit_source_keys']==cfg['train_keys']
        pooled=fit_scaler(ROOT,cfg['train_keys'],cfg['method'])
        assert pooled==cfg['scaler'],'Archived scaler differs from original training statistics'
        for p,digest in cfg['source_sha256'].items():assert sha(ROOT/p)==digest,'Original source code differs: '+p
        history=read_csv(folder/'history.csv')
        logs=[json.loads(line) for line in (ROOT/reg['pipeline']/'logs'/(run+'.log')).read_text().splitlines() if line.startswith('{')]
        assert [int(r['epoch']) for r in history]==list(range(1,21))
        assert [r['epoch'] for r in logs]==list(range(1,21))
        for row,line in zip(history,logs):
            assert line['run']==run
            for k in row:assert abs(float(row[k])-float(line[k]))<1e-10
        assert all(torch.isfinite(value).all().item() for value in cp['model'].values())
        payload=hashlib.sha256(b''.join(v.numpy().tobytes() for v in cp['model'].values())).hexdigest()
        checks.append(dict(run=run,epoch=cp['epoch'],checkpoint_sha256=sha(cp_path),state_payload_sha256=payload,
            best_epoch_not_used=status['best_source_epoch'],train_keys=cfg['train_keys'],validation_keys=cfg['validation_keys'],
            original_scaler=cfg['scaler'],history_epochs=20,matching_log_epochs=20,
            optimizer='Adam',lr=prior['lr'],weight_decay=prior['weight_decay'],
            fixed_training_control_flow_verified=True,archived_source_hashes_verified=True,passed=True))
    result=dict(passed=True,checks=checks,training_entrypoint_versions=historical,
        evidence_limit=reg['E0']['provenance_limit'],no_new_CNN_training=True)
    write_json(out/'E0_checkpoint_provenance.json',result)
    print('E0 provenance passed for all four actual epoch20 checkpoints',flush=True)


def worker(run,out):
    reg,seal=sealed()
    assert run in reg['runs']
    assert json.loads((out/'E0_checkpoint_provenance.json').read_text())['passed']
    import torch
    from shen_p2a.models import FactorizedAuthorCNN
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    folder=ROOT/reg['pipeline']/'runs'/run
    cp_path=folder/'final.pt'
    assert sha(cp_path)==reg['asset_sha256'][str(cp_path.relative_to(ROOT))]
    cp=torch.load(cp_path,map_location='cpu',weights_only=False);cfg=cp['config']
    assert cp['epoch']==20
    model=FactorizedAuthorCNN(16);model.load_state_dict(cp['model']);model.eval()
    assert next(model.parameters()).device.type=='cpu'
    dst=out/'E0_predictions'/run;dst.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();all_rows=[];sources=[]
    for key in cfg['validation_keys']:
        assert key in ['shen_'+r+'_train' for r in reg['source_receivers']]
        rx=key.removeprefix('shen_').removesuffix('_train')
        cache=ROOT/'outputs/shen_p2a8/g1_cache_v1'/key
        source=json.loads((cache/'manifest.json').read_text())
        assert source['status']=='completed' and source['source']['role']=='train'
        x=np.load(cache/(cfg['method']+'.npy'),mmap_mode='r');y=np.load(cache/'labels.npy');rows=np.load(cache/'rows.npy')
        assert x.shape==(2000,1,52,126) and np.all(np.bincount(y,minlength=10)==200)
        logits=[]
        with torch.inference_mode():
            for start in range(0,len(x),64):
                values=np.array(x[start:start+64],copy=True)
                values=(values-cfg['scaler']['mean'])/cfg['scaler']['std']
                got=model(torch.from_numpy(values.astype('float32'))).numpy()
                assert np.isfinite(got).all();logits.append(got)
        logits=np.concatenate(logits).astype('float64');prob=softmax(logits,axis=1)
        np.savez(dst/(rx+'.npz'),logits=logits,probabilities=prob,labels=y,rows=rows)
        for i in range(len(y)):
            all_rows.append(dict(sample_id=f'{key}:TX{int(y[i])}:row{int(rows[i])}',receiver=rx,TX=int(y[i])+31,label=int(y[i]),raw_row=int(rows[i]),
                **{f'p_{j}':float(prob[i,j]) for j in range(10)},**{f'logit_{j}':float(logits[i,j]) for j in range(10)}))
        sources.append(dict(receiver=rx,cache=str(cache.relative_to(ROOT)),role='source_train_only',packets=len(y)))
        print(run+' '+rx+' CPU inference complete',flush=True)
    write_csv(dst/'predictions.csv',all_rows)
    write_json(dst/'manifest.json',dict(run=run,epoch=20,checkpoint_sha256=sha(cp_path),registration_sha256=seal['registration_sha256'],
        sources=sources,scaler=cfg['scaler'],packets=len(all_rows),device='cpu',training=False,seconds=time.perf_counter()-started))


def pairs(out,fold,rx):
    names=['strict_cis','full_cis']
    arrays=[]
    for name in names:
        with np.load(out/'E0_predictions'/f'CV{fold}_{name}'/(rx+'.npz')) as a:arrays.append({k:a[k].copy() for k in a.files})
    s,f=arrays
    np.testing.assert_array_equal(s['rows'],f['rows']);np.testing.assert_array_equal(s['labels'],f['labels'])
    assert len(np.unique(s['rows']))==len(s['rows'])==2000
    for a in arrays:np.testing.assert_allclose(a['probabilities'].sum(1),1,atol=1e-14,rtol=0)
    return dict(ps=s['probabilities'],pf=f['probabilities'],ls=s['logits'],lf=f['logits'],y=s['labels'],rows=s['rows'])


def E1(reg,out):
    dst=out/'E1';dst.mkdir()
    per_rx=[];per_tx=[];states=[]
    for fold,receivers in enumerate(reg['source_receiver_folds']):
        for rx in receivers:
            a=pairs(out,fold,rx);s=a['ps'].argmax(1);f=a['pf'].argmax(1);y=a['y']
            sc,fc=s==y,f==y;hard=sc|fc
            convex=convex_oracle(a['ps'],a['pf'],y)
            assert (convex['feasible']|~hard).all(),'Soft oracle cannot be below hard endpoints'
            sm,fm=metric(y,s),metric(y,f)
            row=dict(fold=fold,receiver=rx,packets=len(y),Full_accuracy=fm['accuracy'],Strict_accuracy=sm['accuracy'],
                Full_macro_f1=fm['macro_f1'],Strict_macro_f1=sm['macro_f1'],Hard_oracle_accuracy=float(hard.mean()),
                Convex_soft_oracle_accuracy=float(convex['feasible'].mean()),
                Hard_gain_over_RX_best=float(hard.mean()-max(sc.mean(),fc.mean())),
                Full_rescue=int((fc&~sc).sum()),Full_damage=int((sc&~fc).sum()),Both_correct=int((sc&fc).sum()),Both_wrong=int((~sc&~fc).sum()),
                Both_wrong_soft_rescuable=int((convex['feasible']&~hard).sum()))
            per_rx.append(row)
            for cls in range(10):
                m=y==cls
                per_tx.append(dict(fold=fold,receiver=rx,TX=cls+31,packets=int(m.sum()),
                    Full_recall=float(fc[m].mean()),Strict_recall=float(sc[m].mean()),
                    Hard_oracle_recall=float(hard[m].mean()),Convex_soft_oracle_recall=float(convex['feasible'][m].mean()),
                    Full_rescue=int((fc&~sc&m).sum()),Full_damage=int((sc&~fc&m).sum()),Both_correct=int((sc&fc&m).sum()),Both_wrong=int((~sc&~fc&m).sum())))
            for i in range(len(y)):
                states.append(dict(fold=fold,receiver=rx,raw_row=int(a['rows'][i]),TX=int(y[i])+31,
                    Strict_prediction=int(s[i])+31,Full_prediction=int(f[i])+31,
                    Strict_correct=bool(sc[i]),Full_correct=bool(fc[i]),hard_oracle_correct=bool(hard[i]),soft_oracle_feasible=bool(convex['feasible'][i]),
                    feasible_lambda_lower=float(convex['lower'][i]),feasible_lambda_upper=float(convex['upper'][i]),
                    lambda_lower_open=bool(convex['lower_open'][i]),lambda_upper_open=bool(convex['upper_open'][i])))
    write_csv(dst/'per_RX.csv',per_rx);write_csv(dst/'per_RX_TX.csv',per_tx);write_csv(dst/'packet_correctness_and_oracle.csv',states)
    gain=float(np.mean([r['Hard_gain_over_RX_best'] for r in per_rx]));passed=gain>=reg['E1']['hard_gain_min']
    result=dict(passed=passed,mean_equal_RX_hard_gain=gain,threshold=.03,
        mean_equal_RX_Full_accuracy=float(np.mean([r['Full_accuracy'] for r in per_rx])),
        mean_equal_RX_Strict_accuracy=float(np.mean([r['Strict_accuracy'] for r in per_rx])),
        mean_equal_RX_Hard_oracle_accuracy=float(np.mean([r['Hard_oracle_accuracy'] for r in per_rx])),
        mean_equal_RX_Convex_soft_oracle_accuracy=float(np.mean([r['Convex_soft_oracle_accuracy'] for r in per_rx])),
        source_RX=10,unique_source_packets=20000,expert_pairs=2,seed=20261012,
        state_totals={k:sum(r[k] for r in per_rx) for k in ('Full_rescue','Full_damage','Both_correct','Both_wrong','Both_wrong_soft_rescuable')},
        soft_oracle_used_for_gate=False,oracle_macro_f1_claimed=False)
    write_json(dst/'decision.json',result);print('E1 '+json.dumps(result),flush=True)
    return result


def mean_rx_f1(y,pred,rx):
    return float(np.mean([metric(y[rx==r],pred[rx==r])['macro_f1'] for r in np.unique(rx)]))


def fit_temperature(logits,y,rx):
    groups=np.unique(rx)
    def objective(logt):
        z=logits/np.exp(logt);loss=logsumexp(z,axis=1)-z[np.arange(len(y)),y]
        return float(np.mean([loss[rx==r].mean() for r in groups]))
    fit=minimize_scalar(objective,bounds=(np.log(.25),np.log(4)),method='bounded',options=dict(xatol=1e-6,maxiter=100))
    if not fit.success:raise ValueError('Temperature fit failed; no alternative algorithm')
    return float(np.exp(fit.x))


def baseline_predict(name,a,params):
    if name=='probability_average_50_50':return ((a['ps']+a['pf'])/2).argmax(1)
    if name=='temperature_calibrated_average':return ((softmax(a['ls']/params['T_S'],axis=1)+softmax(a['lf']/params['T_F'],axis=1))/2).argmax(1)
    if name=='source_weight_selected_fusion':return ((1-params['Full_weight'])*a['ps']+params['Full_weight']*a['pf']).argmax(1)
    if name=='maximum_confidence_selection':return np.where(a['pf'].max(1)>a['ps'].max(1),a['pf'].argmax(1),a['ps'].argmax(1))
    raise ValueError(name)


def E2(reg,out):
    dst=out/'E2';dst.mkdir()
    per_rx=[];all_tx=[];all_methods=[];all_predictions=[]
    for fold,receivers in enumerate(reg['source_receiver_folds']):
        for test_rx,train_rx in receiver_splits(receivers):
            folder=dst/test_rx;folder.mkdir()
            train_parts=[pairs(out,fold,r) for r in train_rx]
            train={k:np.concatenate([a[k] for a in train_parts]) for k in train_parts[0]}
            group=np.concatenate([np.repeat(r,len(a['y'])) for r,a in zip(train_rx,train_parts)])
            y=train['y'];ps,pf=train['ps'],train['pf'];s,f=ps.argmax(1),pf.argmax(1)
            eligible=(s!=f)&((s==y)^(f==y));target=(f[eligible]==y[eligible]).astype('int64')
            feats=observables(ps,pf);scaler=StandardScaler();model=None;fallback=None
            if len(target) and len(np.unique(target))==2:
                z=scaler.fit_transform(feats[eligible])
                gr=group[eligible];weights=np.zeros(len(gr))
                for r in np.unique(gr):weights[gr==r]=len(gr)/(len(np.unique(gr))*np.sum(gr==r))
                model=LogisticRegression(C=1.,penalty='l2',solver='lbfgs',max_iter=1000,tol=1e-8,random_state=20261012)
                model.fit(z,target,sample_weight=weights)
                assert model.n_iter_[0]<1000,'Linear model failed to converge'
            else:fallback=int(target[0]) if len(target) else 0
            temp_s=fit_temperature(train['ls'],y,group);temp_f=fit_temperature(train['lf'],y,group)
            weighted=[]
            for weight in reg['E2']['fixed_weights']:
                pred=((1-weight)*ps+weight*pf).argmax(1)
                weighted.append(dict(weight=weight,mean_RX_macro_f1=mean_rx_f1(y,pred,group)))
            chosen_weight=sorted(weighted,key=lambda r:(-r['mean_RX_macro_f1'],abs(r['weight']-.5),r['weight']))[0]['weight']
            params=dict(T_S=temp_s,T_F=temp_f,Full_weight=chosen_weight)
            baseline_scores={name:mean_rx_f1(y,baseline_predict(name,train,params),group) for name in reg['E2']['baselines']}
            best=max(reg['E2']['baselines'],key=lambda name:baseline_scores[name])
            freeze=dict(expert_fold=fold,selector_train_RX=train_rx,test_RX=test_rx,
                fit_feature_names=FEATURES,informative_packets=int(eligible.sum()),Full_correct_training_cases=int(target.sum()),Strict_correct_training_cases=int(len(target)-target.sum()),
                baseline_params=params,baseline_training_scores=baseline_scores,selected_baseline=best,weight_search=weighted,
                linear_model=None if model is None else dict(intercept=model.intercept_.tolist(),coef=model.coef_.tolist(),scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),iterations=int(model.n_iter_[0])),
                fallback=fallback,choice_threshold=.5,
                selection_frozen_before_this_RX_E2_evaluation=True,
                E1_previously_used_source_labels=True,frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_json(folder/'selection.frozen.json',freeze)
            # Evaluation starts only after source-only calibration, baseline choice and selector are frozen.
            test=pairs(out,fold,test_rx);ts,tf=test['ps'].argmax(1),test['pf'].argmax(1)
            gate=np.full(len(ts),float(fallback)) if model is None else model.predict_proba(scaler.transform(observables(test['ps'],test['pf'])))[:,1]
            take_full=(ts!=tf)&(gate>=.5);pred=np.where(take_full,tf,ts)
            candidates={name:baseline_predict(name,test,params) for name in reg['E2']['baselines']}
            candidates.update(Strict=ts,Full=tf,Hard_selector=pred)
            candidate_metrics={name:metric(test['y'],p) for name,p in candidates.items()}
            baseline=candidate_metrics[best];result=candidate_metrics['Hard_selector']
            per_rx.append(dict(fold=fold,receiver=test_rx,selected_baseline=best,packets=len(ts),
                baseline_macro_f1=baseline['macro_f1'],selector_macro_f1=result['macro_f1'],delta_macro_f1=result['macro_f1']-baseline['macro_f1'],
                baseline_accuracy=baseline['accuracy'],selector_accuracy=result['accuracy'],
                baseline_worst_TX_recall=baseline['worst_TX_recall'],selector_worst_TX_recall=result['worst_TX_recall'],
                rescued_vs_baseline=int(((pred==test['y'])&(candidates[best]!=test['y'])).sum()),
                new_errors_vs_baseline=int(((pred!=test['y'])&(candidates[best]==test['y'])).sum()),Full_selection_fraction=float(take_full.mean()),
                disagreement_Full_selection_fraction=float(take_full.sum()/max(np.sum(ts!=tf),1))))
            for name,m in candidate_metrics.items():
                all_methods.append(dict(receiver=test_rx,fold=fold,method=name,selected_for_gate=name==best,accuracy=m['accuracy'],macro_f1=m['macro_f1'],worst_TX_recall=m['worst_TX_recall']))
                np.savetxt(folder/(name+'_confusion.csv'),m['confusion'],fmt='%d',delimiter=',')
            for cls in range(10):
                loss=baseline['recall'][cls]-result['recall'][cls]
                all_tx.append(dict(receiver=test_rx,fold=fold,TX=cls+31,support=int(result['support'][cls]),baseline=best,
                    baseline_recall=float(baseline['recall'][cls]),selector_recall=float(result['recall'][cls]),delta_recall=float(-loss),
                    severe=bool(result['support'][cls]>=50 and loss>.10)))
            for i in range(len(ts)):
                all_predictions.append(dict(receiver=test_rx,fold=fold,raw_row=int(test['rows'][i]),label=int(test['y'][i]),
                    Strict=int(ts[i]),Full=int(tf[i]),selector=int(pred[i]),Full_risk_probability=float(gate[i]),
                    selected_baseline=best,baseline_prediction=int(candidates[best][i]),
                    **{name:int(p[i]) for name,p in candidates.items() if name not in ('Strict','Full','Hard_selector')}))
            print('E2 '+test_rx+' '+json.dumps(per_rx[-1]),flush=True)
    write_csv(dst/'per_RX.csv',per_rx);write_csv(dst/'per_RX_TX.csv',all_tx);write_csv(dst/'all_baseline_metrics.csv',all_methods);write_csv(dst/'predictions.csv',all_predictions)
    baseline_recall=[np.mean([r['baseline_recall'] for r in all_tx if r['TX']==cls]) for cls in range(31,41)]
    selector_recall=[np.mean([r['selector_recall'] for r in all_tx if r['TX']==cls]) for cls in range(31,41)]
    delta=float(np.mean([r['delta_macro_f1'] for r in per_rx]));worst_gain=float(min(selector_recall)-min(baseline_recall))
    g=reg['E2']['gate'];severe=[r for r in all_tx if r['severe']]
    gates=dict(mean_macro_f1=delta>=g['mean_macro_f1_gain_min'],all_ten_RX_nonnegative=all(r['delta_macro_f1']>=0 for r in per_rx),
               worst_TX_recall=worst_gain>=g['worst_TX_recall_gain_min'],no_severe_RX_TX_degradation=not severe)
    result=dict(passed=all(gates.values()),gates=gates,mean_equal_RX_delta_macro_f1=delta,
        minimum_RX_delta_macro_f1=float(min(r['delta_macro_f1'] for r in per_rx)),
        baseline_worst_TX_recall=float(min(baseline_recall)),selector_worst_TX_recall=float(min(selector_recall)),worst_TX_recall_gain=worst_gain,
        baseline_equal_RX_recall_by_TX=baseline_recall,selector_equal_RX_recall_by_TX=selector_recall,
        severe_RX_TX=severe,severe_fraction=len(severe)/len(all_tx),linear_fits=10,expert_training_runs=0,
        baseline_selection_used_test_labels=False,next_stage_not_authorized=True)
    write_json(dst/'decision.json',result);return result


def report(reg,seal,out,e1,e2):
    state='STOP_E1' if not e1['passed'] else ('STOP_E2' if not e2['passed'] else 'E2_PASSED_NO_E3_AUTHORIZATION')
    decision=dict(experiment='EISR-v1',state=state,E0_passed=True,E1_passed=e1['passed'],E2_executed=e2 is not None,
        E2_passed=None if e2 is None else e2['passed'],new_CNN_training_runs=0,device='CPU',
        developer_RX_accessed=False,historical_locked_RX_accessed=False,OSU_accessed=False,
        registration_sha256=seal['registration_sha256'],E3_started=False,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write_json(out/'decision.json',decision)
    lines=['# EISR-v1 · E0/E1/E2 CPU 筛选','',f'**决策：{state}。**','',
        '这是历史源数据上的两级接收机分组内部筛选，不是 CNN 十折留一或独立外部盲测。VIRP-v1 负结果保持封存。','',
        f"注册 SHA256：`{seal['registration_sha256']}`。先冻结后推理，无新增 CNN 训练。四个真实 epoch20 `final.pt` 已经通过 metadata、训练历史、逐 epoch 日志、源 scaler 和原代码哈希核对；未使用 best checkpoint 预测。",'',
        '两组专家分别在五源 RX 训练，并仅对另外五源 RX 重新进行 CPU 推理。Full/Strict 各 20,000 次预测逐原始行 ID 配对；两个一级折复用专家，因此十 RX 结果相关，不按包计算显著性。','',
        '## E1：互补性与 Accuracy oracle','',
        f"十 RX 等权 Full Accuracy={e1['mean_equal_RX_Full_accuracy']:.4%}；Strict={e1['mean_equal_RX_Strict_accuracy']:.4%}；Hard oracle={e1['mean_equal_RX_Hard_oracle_accuracy']:.4%}；Convex soft oracle={e1['mean_equal_RX_Convex_soft_oracle_accuracy']:.4%}。",'',
        f"Hard oracle 相对每 RX 最强单专家的平均空间为 **{e1['mean_equal_RX_hard_gain']*100:.4f} pp**；预登记要求 ≥3 pp；{'通过' if e1['passed'] else '未通过'}。Soft oracle 不参与关口，也不称为 Macro-F1 上界。",'',
        '|RX|Full Acc|Strict Acc|Hard oracle|Soft oracle|相对 RX 最强单专家空间|Full 可挽救包|Full 引入错误包|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in read_csv(out/'E1/per_RX.csv'):
        lines.append(f"|{r['receiver']}|{float(r['Full_accuracy']):.2%}|{float(r['Strict_accuracy']):.2%}|{float(r['Hard_oracle_accuracy']):.2%}|{float(r['Convex_soft_oracle_accuracy']):.2%}|{float(r['Hard_gain_over_RX_best'])*100:.2f} pp|{r['Full_rescue']}|{r['Full_damage']}|")
    if e2 is None:
        lines+=['','E1 未通过，按规则不拟合选择器，不改用更宽松的软融合上界继续。']
    else:
        lines+=['','## E2：四 RX 拟合、一 RX 评价','',
            '每次仅使用同一对固定专家未见的四台 RX 训练一个线性选择器及校准简单基线；第五台 RX 不用于温度、融合权重、基线名称或选择器参数的选择。基线选择记录在每 RX 的 `selection.frozen.json` 中。','',
            '|RX|训练四 RX 选定基线|基线 F1|选择器 F1|ΔF1|基线最差 TX Recall|选择器最差 TX Recall|',
            '|---|---|---:|---:|---:|---:|---:|']
        for r in read_csv(out/'E2/per_RX.csv'):
            lines.append(f"|{r['receiver']}|{r['selected_baseline']}|{float(r['baseline_macro_f1']):.4f}|{float(r['selector_macro_f1']):.4f}|{float(r['delta_macro_f1']):+.4f}|{float(r['baseline_worst_TX_recall']):.2%}|{float(r['selector_worst_TX_recall']):.2%}|")
        lines+=['',f"平均 ΔMacro-F1={e2['mean_equal_RX_delta_macro_f1']:+.6f}（要求≥.01）；最小 RX Δ={e2['minimum_RX_delta_macro_f1']:+.6f}（要求≥0）；等权 RX 后最差 TX Recall 增益={e2['worst_TX_recall_gain']:+.6f}（要求≥.05）；严重 RX×TX 退化 {len(e2['severe_RX_TX'])}/100（要求0）。",'',
            'E2 所有预注册基线都报告，但关口只对四训练 RX 选定者比较，不在第五 RX 挑最好的基线。Full/Strict 单专家同时报告。', '',
            '选择器只利用预测置信度、Top1−Top2 margin、熵、分歧及概率分布差异，不输入真实 TX、RX 身份、真实类别 margin、原型或人工 TX36 规则。学习目标只来自四训练 RX 的一对一正确性分歧。']
    lines+=['','## 证据边界与复现','',
        '第20轮权重排除了本轮最佳 checkpoint 选择回流；此前源数据上的模型与超参数开发偏差仍存在。E1 用了十 RX 标签决定是否执行 E2，所以 E2 属于同批数据的阶段性机制研究，不能称完全独立确认。原 checkpoint 不包含 optimizer state/入口脚本哈希；现有代码版本、匹配日志、checkpoint 元数据与 source SHA 相互佐证，不声称具有密码学执行证明。','',
        '文献/官方代码核对见 `docs/eisr/E0_LITERATURE_AUDIT.md`。EISR 尚无已证明的新机制贡献；普通置信度选择、模型堆叠、概率融合和 MoE 均不能改名当创新。', '',
        '冻结协议：`configs/eisr/v1_registration.json`、`.seal.json`；E0：`E0_checkpoint_provenance.json`、`E0_predictions/*`；E1：逐 RX/TX、逐包四种正确性与解析 soft 可行区间；E2：每 RX 的冻结选择记录、线性模型参数、全基线指标、逐包预测与混淆矩阵。','',
        '复现需要原始已核验的 G1 权重/缓存及新的空输出目录：`CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 /home/keyan/venvs/shen-stage1/bin/python scripts/run_eisr_v1.py --output outputs/eisr/v1_reproduction`。','',
        '当前注册不启动 E3/E4、新 CNN、HSTU/CrossAttn、Shen 四开发或六历史锁定 RX、OSU 新评价。任何后续方法变化需要新协议，并保留当前结果。','']
    (out/'REPORT.md').write_text('\n'.join(lines));return decision


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output');parser.add_argument('--worker');args=parser.parse_args()
    reg,seal=sealed(all_inputs=not bool(args.worker));out=ROOT/(args.output or reg['output'])
    if args.worker:worker(args.worker,out);return
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    write_json(out/'registration.json',reg);write_json(out/'seal.json',seal)
    write_json(out/'status.json',dict(status='running',stage='E0_provenance'))
    try:
        provenance(reg,out)
        logs=out/'inference_logs';logs.mkdir()
        def launch(run):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
            with (logs/(run+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/run_eisr_v1.py'),'--worker',run,'--output',str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(run+' fixed epoch20 CPU prediction saved',flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(launch,reg['runs']))
        write_json(out/'status.json',dict(status='running',stage='E1'))
        e1=E1(reg,out);e2=None
        if e1['passed']:
            write_json(out/'status.json',dict(status='running',stage='E2'));e2=E2(reg,out)
        decision=report(reg,seal,out,e1,e2)
        write_json(out/'status.json',dict(status='completed',decision=decision['state'],seconds=time.perf_counter()-started))
        print(json.dumps(decision),flush=True)
    except Exception as e:
        write_json(out/'status.json',dict(status='failed',error=repr(e),seconds=time.perf_counter()-started))
        write_json(out/'decision.json',dict(state='STOP_E0_OR_EXECUTION_ERROR',new_CNN_training_runs=0,E3_started=False,error=repr(e)))
        raise


if __name__=='__main__':main()
