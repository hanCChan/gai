#!/usr/bin/env python3
"""Independent reconstruction; deliberately does not import D0 metric code."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'reports/d0_class_pair/2026-10-13_D0'

def main():
    r=json.loads((ROOT/'configs/d0_class_pair/registration.json').read_text())
    pairs=pd.read_csv(P/'directed_pairs.csv');evaluation=pd.read_csv(P/'risk_prediction_evaluation.csv')
    vectors=pd.read_csv(P/'risk_prediction_vectors.csv');conf=pd.read_csv(P/'confusion_counts.csv')
    matrix={};metric_checks=0;profile_checks=0
    for a in r['prediction_assets']:
        rx,method=a['receiver'],a['method']
        with np.load(a['path'],allow_pickle=False) as f: z=f['logits'];y=f['labels']
        prediction=z.argmax(1)
        c=np.array([[np.sum((y==i)&(prediction==j)) for j in range(10)] for i in range(10)])
        m=c/200;matrix[rx,method]=m
        observed=conf[(conf.receiver==rx)&(conf.method==method)].sort_values(['tx_i','tx_j'])['count'].to_numpy().reshape(10,10)
        assert np.array_equal(c,observed)
        scale=np.linalg.norm(z-z.mean(1)[:,None],axis=1)/np.sqrt(10)
        scale[scale==0]=1
        for _,row in pairs[(pairs.receiver==rx)&(pairs.method==method)].iterrows():
            i,j=int(row.i),int(row.j);d=(z[y==i,i]-z[y==i,j]);n=d/scale[y==i]
            wanted=[m[i,j],np.mean(d<=0),np.mean(d),np.median(d),np.quantile(d,.05),np.mean(n),np.median(n),np.quantile(n,.05)]
            keys=['confusion_risk','pair_order_violation','raw_margin_mean','raw_margin_median','raw_margin_q05','normalized_margin_mean','normalized_margin_median','normalized_margin_q05']
            assert np.allclose(wanted,[row[k] for k in keys],rtol=1e-11,atol=1e-11);metric_checks+=len(keys)
    mask=~np.eye(10,dtype=bool)
    for _,e in evaluation.iterrows():
        refs=e.reference_receivers.split(';');probe=e.probe_receiver
        assert probe not in refs and len(refs)==e.reference_count
        fold=int(e.fold);same=r['source_receiver_folds'][fold]
        expected=r['source_receiver_folds'][1-fold] if e.control=='cross_cnn_transfer' else [rx for rx in same if rx!=probe]
        if e.control=='different_family_reference':expected=[rx for rx in expected if r['hardware_families'][rx]!=r['hardware_families'][probe]]
        assert refs==expected
        avg=np.mean([matrix[rx,e.method] for rx in refs],axis=0);np.fill_diagonal(avg,0)
        if e.predictor=='pairwise_mean':scores=avg[mask]
        elif e.predictor=='uniform':scores=np.ones(90)
        elif e.predictor=='source_class_only':scores=np.repeat(avg.sum(1)/9,9)
        else:
            scores=[]
            dest=avg.sum(0)
            for i in range(10):
                weights=dest.copy();weights[i]=0
                if weights.sum()==0: weights=np.ones(10);weights[i]=0
                scores.extend((avg[i].sum()*weights/weights.sum())[np.arange(10)!=i])
            scores=np.array(scores)
        vv=vectors[(vectors.method==e.method)&(vectors.probe_receiver==probe)&(vectors.control==e.control)&(vectors.predictor==e.predictor)].sort_values(['tx_i','tx_j'])
        observed=matrix[probe,e.method][mask]
        assert len(vv)==90 and np.allclose(scores,vv.predicted_risk) and np.allclose(observed,vv.observed_risk)
        rho=np.corrcoef(rankdata(scores),rankdata(observed))[0,1] if np.ptp(scores)>0 and np.ptp(observed)>0 else np.nan
        boundary=sorted(scores,reverse=True)[9];above=scores>boundary;tie=scores==boundary
        selected=above.astype(float);selected[tie]=(10-sum(above))/sum(tie)
        capture=np.sum(selected*observed)/sum(observed) if sum(observed)>0 else np.nan
        assert np.isclose(sum(selected),10)
        assert np.allclose([rho,capture],[e.spearman,e.top10_error_capture],equal_nan=True)
        profile_checks+=1
    decision=json.loads((P/'decision.json').read_text())
    accuracy={m:np.mean([np.trace(matrix[rx,m])/10 for rx in r['source_receivers']]) for m in r['methods']}
    assert all(np.isclose(accuracy[m],decision['mean_accuracy'][m]) for m in accuracy)
    changes=pd.read_csv(P/'strict_full_pair_changes.csv')
    harmful=changes[changes.delta_confusion_risk>0].groupby(['tx_i','tx_j']).fold.nunique()
    assert sum(harmful==2)==decision['harmful_edges_both_folds']
    assert len(pairs)==1800 and len(conf)==2000 and len(changes)==900
    seal=json.loads((ROOT/'configs/d0_class_pair/pre_execution_seal.json').read_text())
    for path,wanted in seal['files_sha256'].items(): assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==wanted
    result={'passed':True,'independent_scalar_metric_checks':metric_checks,'independent_profile_checks':profile_checks,'all_ranking_reference_sets_exclude_probe':True,'raw_confusion_reconstruction':True,'frozen_source_hashes_verified':True,'no_metric_module_import':True,'no_new_model_or_raw_signal_access':True}
    (P/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
