#!/usr/bin/env python3
"""Independently audit heldout predictions and saved source-only fits before a stop/go decision."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from shen_p2a8.g1_data import cache
from shen_p2a.cache import write_json
from report_shen_p2a import read_csv

if __name__=='__main__':
    out=ROOT/'outputs/shen_p2a8/g2_0_v2';reg=json.loads((out/'registration.json').read_text());dec=json.loads((out/'decision.json').read_text());assert dec['status']=='completed'
    actual=read_csv(out/'heldout_candidate_metrics.csv');scores=read_csv(out/'source_only_candidate_scores.csv');correlations=read_csv(out/'source_predictor_rank_correlations.csv');choices=read_csv(out/'source_only_selection_vs_actual_performance.csv')
    assert len(actual)==80 and len(scores)==80 and len(correlations)==len(choices)==10
    checked=[]
    for rx in reg['receivers']:
        folder=out/rx;frozen=json.loads((folder/'source_selection.frozen.json').read_text());source=frozen['source_receivers']
        assert len(source)==9 and rx not in source and set(source)==set(reg['receivers'])-{rx}
        assert frozen['source_keys']==['shen_'+r+'_train' for r in source]
        candidates=frozen['candidates'];assert len(candidates)==8
        selected_r=max(candidates,key=lambda r:(r['reliability_score'] if r['reliability_score'] is not None else -np.inf,-reg['representations'].index(r['representation']),-r['band']))
        selected_a=max(candidates,key=lambda r:(r['inner_source_CV_accuracy'],-reg['representations'].index(r['representation']),-r['band']))
        selected_f=max(candidates,key=lambda r:(r['inner_source_CV_macro_f1'],-reg['representations'].index(r['representation']),-r['band']))
        assert frozen['selected_reliability']==selected_r['candidate'] and frozen['selected_accuracy_only']==selected_a['candidate'] and frozen['selected_source_F1_only']==selected_f['candidate']
        ix={r['candidate']:r for r in actual if r['heldout_receiver']==rx};truth=np.load(cache(ROOT,'shen_'+rx+'_train')/'labels.npy');ids=np.load(cache(ROOT,'shen_'+rx+'_train')/'rows.npy')
        for c in candidates:
            name=c['candidate'];rows=read_csv(folder/(name+'_predictions.csv'));y=np.array([int(r['label']) for r in rows]);p=np.array([int(r['prediction']) for r in rows])
            np.testing.assert_array_equal(y,truth);assert [r['sample_id'] for r in rows]==[f'shen_{rx}_train:row{int(r)}' for r in ids]
            assert len(set(r['sample_id'] for r in rows))==2000
            np.testing.assert_array_equal(confusion_matrix(y,p,labels=range(10)),np.loadtxt(folder/(name+'_confusion.csv'),delimiter=',',dtype=int))
            assert abs(f1_score(y,p,labels=range(10),average='macro',zero_division=0)-float(ix[name]['macro_f1']))<1e-12
            assert abs((y==p).mean()-float(ix[name]['accuracy']))<1e-12
            model=np.load(folder/'source_models'/(name+'.npz'));source_x=[];source_y=[]
            for key in frozen['source_keys']:
                maps=np.load(cache(ROOT,key)/(c['representation']+'.npy'),mmap_mode='r')[:,0]
                # Fresh independent moment calculation, not the fitting helper.
                mu=np.mean(maps,axis=-1,dtype='float64');sd=np.sqrt(np.mean(np.square(maps.astype('float64')-mu[:,:,None]),axis=-1))
                features=np.concatenate((mu,sd),axis=1)[:,model['columns']];source_x.append(features);source_y.append(np.load(cache(ROOT,key)/'labels.npy'))
            x=np.concatenate(source_x);sy=np.concatenate(source_y);mu=x.mean(0);sd=x.std(0);sd=np.where(sd<1e-12,1.,sd)
            np.testing.assert_allclose(model['mean'],mu,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(model['std'],sd,rtol=1e-12,atol=1e-12)
            z=(x-mu)/sd;design=np.column_stack((z,np.ones(len(z))));penalty=np.eye(design.shape[1]);penalty[-1,-1]=0
            residual=(design.T@design+penalty)@model['weights']-design.T@np.eye(10)[sy]
            assert np.max(np.abs(residual))<1e-7
            # Reporting amendment added literal Accuracy control, without changing any classifier/prediction.
            old=ROOT/'outputs/shen_p2a8/g2_0_v1'/rx/(name+'_predictions.csv')
            assert old.read_bytes()==(folder/(name+'_predictions.csv')).read_bytes()
            checked.append(dict(rx=rx,candidate=name,source_rx=9,predictions=2000,ids_labels_cm_f1_exact=True,source_only_scaler_verified=True,ridge_normal_equation_verified=True,v1_v2_predictions_exact=True))
        target=[float(ix[c['candidate']]['macro_f1']) for c in candidates];r=next(r for r in correlations if r['heldout_receiver']==rx)
        for key,source_score in [('reliability_vs_actual_F1_spearman',[c['reliability_score'] for c in candidates]),('source_CV_accuracy_vs_actual_F1_spearman',[c['inner_source_CV_accuracy'] for c in candidates]),('source_CV_F1_vs_actual_F1_spearman',[c['inner_source_CV_macro_f1'] for c in candidates])]:
            assert abs(float(spearmanr(source_score,target).statistic)-float(r[key]))<1e-12
        choice=next(r for r in choices if r['heldout_receiver']==rx)
        assert abs(float(choice['macro_f1_delta'])-(float(ix[selected_r['candidate']]['macro_f1'])-float(ix[selected_a['candidate']]['macro_f1'])))<1e-12
    write_json(out/'independent_verification.json',dict(passed=True,outer_RX=10,source_RX_per_fold=9,prediction_sets=80,heldout_predictions=160000,
        developer_receivers_accessed=False,locked_accessed=False,source_only_fit_normal_equations_verified=True,v1_v2_predictions_bit_identical=True,checks=checked))
    print('80 models and 160000 heldout predictions independently verified',flush=True)
