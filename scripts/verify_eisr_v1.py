#!/usr/bin/env python3
"""Independent recomputation of EISR result gates and selected algorithms."""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy.optimize import linprog
from scipy.special import expit,softmax
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csv_rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def read_pair(out,fold,rx):
    values=[]
    for method in ('strict_cis','full_cis'):
        with np.load(out/'E0_predictions'/f'CV{fold}_{method}'/(rx+'.npz')) as a:values.append({k:a[k].copy() for k in a.files})
    s,f=values
    np.testing.assert_array_equal(s['labels'],f['labels']);np.testing.assert_array_equal(s['rows'],f['rows'])
    return s,f


def independent_observables(s,f):
    sx,fx=np.sort(s,axis=1),np.sort(f,axis=1)
    cs,cf=sx[:,-1],fx[:,-1];ms,mf=cs-sx[:,-2],cf-fx[:,-2]
    es=-(s*np.log(np.maximum(s,1e-300))).sum(1)/np.log(10)
    ef=-(f*np.log(np.maximum(f,1e-300))).sum(1)/np.log(10)
    p=(s+f)/2
    js=((s*np.log(np.maximum(s,1e-300)/np.maximum(p,1e-300))).sum(1)+(f*np.log(np.maximum(f,1e-300)/np.maximum(p,1e-300))).sum(1))/2
    return np.column_stack((cs,cf,ms,mf,es,ef,cf-cs,mf-ms,ef-es,s.argmax(1)!=f.argmax(1),np.abs(s-f).sum(1)/2,js))


def LP_soft(s,f,y):
    rows=[];bounds=[]
    for j in range(10):
        if j==y:continue
        a=float(s[y]-s[j]);b=float(f[y]-f[j]-a)
        rows.append([-b,1. if j<y else 0.]);bounds.append(a)
    r=linprog([0,-1],A_ub=rows,b_ub=bounds,bounds=[(0,1),(0,1)],method='highs')
    if not r.success:return False
    return y==0 or r.x[1]>0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/eisr/v1_20261012');args=parser.parse_args()
    out=ROOT/args.output;reg=json.loads((out/'registration.json').read_text());seal=json.loads((out/'seal.json').read_text())
    assert sha(ROOT/'configs/eisr/v1_registration.json')==seal['registration_sha256']
    for p,digest in reg['immutable_code_sha256'].items():assert sha(ROOT/p)==digest
    assert json.loads((out/'status.json').read_text())['status']=='completed'
    e1=json.loads((out/'E1/decision.json').read_text());per_rx={r['receiver']:r for r in csv_rows(out/'E1/per_RX.csv')}
    oracle_rows={(r['receiver'],int(r['raw_row'])):r for r in csv_rows(out/'E1/packet_correctness_and_oracle.csv')}
    gains=[];lp_checks=0;witness_checks=0;prob_checks=0
    for fold,receivers in enumerate(reg['source_receiver_folds']):
        for rx in receivers:
            s,f=read_pair(out,fold,rx);y=s['labels'];sp,fp=s['probabilities'],f['probabilities']
            for a in (s,f):np.testing.assert_allclose(softmax(a['logits'],axis=1),a['probabilities'],atol=1e-14,rtol=0)
            prob_checks+=len(y)*2
            correct_s,correct_f=sp.argmax(1)==y,fp.argmax(1)==y
            gain=float(np.mean(correct_s|correct_f)-max(correct_s.mean(),correct_f.mean()));gains.append(gain)
            assert abs(gain-float(per_rx[rx]['Hard_gain_over_RX_best']))<1e-14
            for i in range(len(y)):
                row=oracle_rows[rx,int(s['rows'][i])]
                feasible=row['soft_oracle_feasible']=='True'
                if feasible:
                    lower=float(row['feasible_lambda_lower']);upper=float(row['feasible_lambda_upper']);g=(lower+upper)/2
                    assert ((1-g)*sp[i]+g*fp[i]).argmax()==y[i]
                    witness_checks+=1
            # Fixed first two rows per class, 200 independent LP checks, no outcome sampling.
            for cls in range(10):
                for i in np.flatnonzero(y==cls)[:2]:
                    row=oracle_rows[rx,int(s['rows'][i])]
                    assert LP_soft(sp[i],fp[i],int(y[i]))==(row['soft_oracle_feasible']=='True')
                    lp_checks+=1
    assert abs(np.mean(gains)-e1['mean_equal_RX_hard_gain'])<1e-14
    assert (np.mean(gains)>=.03)==e1['passed']
    overall=json.loads((out/'decision.json').read_text())
    e2_verification=None
    if overall['E2_executed']:
        per_e2={r['receiver']:r for r in csv_rows(out/'E2/per_RX.csv')}
        output_predictions=csv_rows(out/'E2/predictions.csv')
        rec_s=[];rec_b=[];f1_delta=[];severe=[];selector_checks=0
        for fold,receivers in enumerate(reg['source_receiver_folds']):
            for rx in receivers:
                chosen=json.loads((out/'E2'/rx/'selection.frozen.json').read_text())
                assert chosen['test_RX']==rx and set(chosen['selector_train_RX'])==set(receivers)-{rx}
                assert chosen['expert_fold']==fold
                baseline_names=reg['E2']['baselines']
                source_scores={name:[] for name in baseline_names}
                weight_scores={w:[] for w in reg['E2']['fixed_weights']}
                params=chosen['baseline_params']
                def baselines(s,f):
                    sp,fp=s['probabilities'],f['probabilities']
                    return {'probability_average_50_50':((sp+fp)/2).argmax(1),
                        'temperature_calibrated_average':((softmax(s['logits']/params['T_S'],axis=1)+softmax(f['logits']/params['T_F'],axis=1))/2).argmax(1),
                        'source_weight_selected_fusion':((1-params['Full_weight'])*sp+params['Full_weight']*fp).argmax(1),
                        'maximum_confidence_selection':np.where(fp.max(1)>sp.max(1),fp.argmax(1),sp.argmax(1))}
                for source in chosen['selector_train_RX']:
                    s,f=read_pair(out,fold,source)
                    for name,pred in baselines(s,f).items():source_scores[name].append(f1_score(s['labels'],pred,labels=np.arange(10),average='macro',zero_division=0))
                    for w in weight_scores:
                        pred=((1-w)*s['probabilities']+w*f['probabilities']).argmax(1)
                        weight_scores[w].append(f1_score(s['labels'],pred,labels=np.arange(10),average='macro',zero_division=0))
                best_weight=sorted(weight_scores,key=lambda w:(-np.mean(weight_scores[w]),abs(w-.5),w))[0]
                assert best_weight==params['Full_weight']
                best=max(baseline_names,key=lambda n:np.mean(source_scores[n]));assert best==chosen['selected_baseline']
                s,f=read_pair(out,fold,rx);sp,fp=s['probabilities'],f['probabilities'];y=s['labels']
                model=chosen['linear_model']
                if model is None:gate=np.full(len(y),chosen['fallback'],dtype=float)
                else:
                    features=independent_observables(sp,fp)
                    z=(features-np.array(model['scaler_mean']))/np.array(model['scaler_scale'])
                    gate=expit(z@np.array(model['coef'])[0]+model['intercept'][0])
                pred=np.where((sp.argmax(1)!=fp.argmax(1))&(gate>=.5),fp.argmax(1),sp.argmax(1))
                base=baselines(s,f)[best]
                indexed={int(r['raw_row']):r for r in output_predictions if r['receiver']==rx}
                for i,row in enumerate(s['rows']):
                    exported=indexed[int(row)]
                    assert pred[i]==int(exported['selector']) and base[i]==int(exported['baseline_prediction'])
                    assert abs(gate[i]-float(exported['Full_risk_probability']))<1e-12
                    selector_checks+=1
                delta=f1_score(y,pred,labels=np.arange(10),average='macro',zero_division=0)-f1_score(y,base,labels=np.arange(10),average='macro',zero_division=0)
                assert abs(delta-float(per_e2[rx]['delta_macro_f1']))<1e-14;f1_delta.append(delta)
                rs=[float(np.mean(pred[y==c]==c)) for c in range(10)];rb=[float(np.mean(base[y==c]==c)) for c in range(10)]
                rec_s.append(rs);rec_b.append(rb)
                for cls in range(10):
                    if np.sum(y==cls)>=50 and rb[cls]-rs[cls]>.1:severe.append((rx,cls+31))
        e2=json.loads((out/'E2/decision.json').read_text())
        gain=float(np.min(np.mean(rec_s,axis=0))-np.min(np.mean(rec_b,axis=0)))
        gates={'mean_macro_f1':np.mean(f1_delta)>=.01,'all_ten_RX_nonnegative':min(f1_delta)>=0,'worst_TX_recall':gain>=.05,'no_severe_RX_TX_degradation':not severe}
        gates={k:bool(v) for k,v in gates.items()}
        assert gates==e2['gates'] and all(gates.values())==e2['passed']
        e2_verification=dict(selector_predictions_recomputed=selector_checks,source_baseline_choices_recomputed=10,
            equal_RX_macro_f1_gain=float(np.mean(f1_delta)),worst_TX_recall_gain=gain,serious_degradation_groups=severe,gates=gates)
    # Verify model inference against the original source maps, separately from the worker implementation.
    import torch
    from shen_p2a.models import FactorizedAuthorCNN
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    max_logit_error=0.;inference_count=0
    for run in reg['runs']:
        cp=torch.load(ROOT/reg['pipeline']/'runs'/run/'final.pt',map_location='cpu',weights_only=False)
        assert cp['epoch']==20
        model=FactorizedAuthorCNN(16);model.load_state_dict(cp['model']);model.eval();cfg=cp['config']
        for key in cfg['validation_keys']:
            rx=key.removeprefix('shen_').removesuffix('_train')
            cache=ROOT/'outputs/shen_p2a8/g1_cache_v1'/key;y=np.load(cache/'labels.npy')
            ids=np.array([np.flatnonzero(y==cls)[0] for cls in range(10)])
            a=np.load(cache/(cfg['method']+'.npy'),mmap_mode='r')[ids].copy()
            a=(a-cfg['scaler']['mean'])/cfg['scaler']['std']
            with torch.inference_mode():got=model(torch.from_numpy(a.astype('float32'))).numpy().astype('float64')
            with np.load(out/'E0_predictions'/run/(rx+'.npz')) as z:expected=z['logits'][ids]
            error=float(np.max(np.abs(got-expected)));max_logit_error=max(max_logit_error,error)
            assert error<2e-5;np.testing.assert_array_equal(got.argmax(1),expected.argmax(1));inference_count+=len(ids)
    result=dict(status='passed',registration_seal_verified=True,E0_independent_CPU_model_predictions=inference_count,
        max_CPU_logit_difference=max_logit_error,expert_probability_records_recomputed=prob_checks,
        E1_source_packets_recomputed=20000,soft_feasible_witnesses_verified=witness_checks,independent_soft_oracle_LP_checks=lp_checks,
        equal_RX_Hard_gain=float(np.mean(gains)),E2=e2_verification,no_new_CNN_training=True)
    payload=json.dumps(result,indent=2,allow_nan=False)+'\n'
    with (out/'independent_verification.json').open('x') as fp:fp.write(payload)
    print(json.dumps(result))


if __name__=='__main__':main()
