#!/usr/bin/env python3
"""Rebuild every profile without importing D0/D0.5 metric functions."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'reports/d05_b0/2026-10-13/d05'
def factorized(m):
    m=m.copy();np.fill_diagonal(m,0);result=np.zeros_like(m);dest=m.sum(0)
    for i in range(10):
        w=dest.copy();w[i]=0
        if w.sum()==0:w=np.ones(10);w[i]=0
        result[i]=m[i].sum()*w/w.sum()
    return result
def main():
    reg=json.loads((ROOT/'configs/d05_b0/registration.json').read_text());old=json.loads((ROOT/'configs/d0_class_pair/registration.json').read_text())
    matrices={};mask=~np.eye(10,dtype=bool)
    for a in old['prediction_assets']:
        with np.load(a['path'],allow_pickle=False) as archive:y=archive['labels'];pred=archive['logits'].argmax(1)
        matrices[a['receiver'],a['method']]=np.array([[np.sum((y==i)&(pred==j))/200 for j in range(10)] for i in range(10)])
    evaluation=pd.read_csv(P/'profile_evaluation.csv');vectors=pd.read_csv(P/'prediction_vectors.csv');checks=0
    for _,e in evaluation.iterrows():
        refs=e.reference_receivers.split(';');probe=e.probe_receiver;fold=int(e.fold);assert probe not in refs
        expected=reg['folds'][1-fold] if e.control=='cross_paired_cnn_transfer' else [r for r in reg['folds'][fold] if r!=probe]
        if e.control=='different_family_reference':expected=[r for r in expected if reg['hardware_families'][r]!=reg['hardware_families'][probe]]
        assert refs==expected
        f=np.mean([matrices[r,'full_cis'] for r in refs],axis=0);s=np.mean([matrices[r,'strict_cis'] for r in refs],axis=0)
        fd=f.copy();sd=s.copy();np.fill_diagonal(fd,0);np.fill_diagonal(sd,0)
        positive=np.mean([np.maximum(matrices[r,'strict_cis']-matrices[r,'full_cis'],0) for r in refs],axis=0);np.fill_diagonal(positive,0)
        scores={'signed_delta_mean':(s-f)[mask],'recall_drop_only':np.repeat((sd-fd).sum(1)/9,9),'class_difficulty_full':np.repeat(fd.sum(1)/9,9),'absolute_full_confusion':f[mask],'absolute_strict_confusion':s[mask],'factorized_delta':(factorized(s)-factorized(f))[mask],'positive_mass_factorized':factorized(positive)[mask],'uniform':np.ones(90)}[e.predictor]
        target=(matrices[probe,'strict_cis']-matrices[probe,'full_cis'])[mask]
        vv=vectors[(vectors.probe_receiver==probe)&(vectors.control==e.control)&(vectors.predictor==e.predictor)].sort_values(['tx_i','tx_j'])
        # Exact arithmetic ordering for zero-boundary ties must follow the
        # registered per-reference delta mean, not subtract rounded means.
        if e.predictor=='signed_delta_mean':scores=np.mean([matrices[r,'strict_cis']-matrices[r,'full_cis'] for r in refs],axis=0)[mask]
        if e.predictor=='recall_drop_only':
            delta=np.mean([matrices[r,'strict_cis']-matrices[r,'full_cis'] for r in refs],axis=0);np.fill_diagonal(delta,0);scores=np.repeat(delta.sum(1)/9,9)
        assert np.allclose(scores,vv.predicted_score,atol=1e-13) and np.allclose(target,vv.observed_signed_delta)
        rho=np.corrcoef(rankdata(scores),rankdata(target))[0,1] if np.ptp(scores)>0 and np.ptp(target)>0 else np.nan
        cut=sorted(scores,reverse=True)[9];weights=(scores>cut).astype(float);tie=scores==cut;weights[tie]=(10-weights.sum())/tie.sum()
        pos=np.maximum(target,0);capture=weights@pos/pos.sum() if pos.sum()>0 else np.nan
        assert np.allclose([rho,capture],[e.signed_spearman,e.positive_mass_capture],atol=1e-11,equal_nan=True),(e.probe_receiver,e.predictor,e.control,rho,e.signed_spearman)
        assert np.allclose([target.sum(),pos.sum(),-np.minimum(target,0).sum()],[e.net_delta_mass,e.positive_delta_mass,e.negative_delta_mass])
        checks+=1
    proof=dict(passed=True,profiles_independently_rebuilt=checks,ranking_vectors_checked=len(vectors),all_references_exclude_probe=True,signed_correlation_and_positive_mass_checked=True,metrics_modules_imported=False)
    (P/'independent_verification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()
