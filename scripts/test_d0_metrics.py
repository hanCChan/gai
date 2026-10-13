#!/usr/bin/env python3
"""Synthetic correctness checks, without reading historical predictions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from d0_class_pair.metrics import pair_metrics, topk_weights, evaluate_profile, predictors

def main():
    z = np.array([[1.,2.,3.],[5.,1.,0.],[0.,1.,3.],[1.,2.,0.],[1.,0.,3.],[3.,2.,1.]])
    y = np.array([0,0,1,1,2,2])
    counts, risk, rows = pair_metrics(z,y)
    assert counts.tolist() == [[1,0,1],[0,1,1],[1,0,1]]
    # j can outrank the true class even if a third class wins.
    row = next(r for r in rows if r['i']==0 and r['j']==1)
    assert row['confusion_risk']==0 and row['pair_order_violation']==.5
    _,_,transformed = pair_metrics(z * 7 + np.arange(6)[:,None],y)
    for a,b in zip(rows, transformed):
        for key in ('normalized_margin_mean','normalized_margin_median','normalized_margin_q05'):
            assert np.isclose(a[key],b[key])
    weights = topk_weights(np.ones(90))
    assert np.isclose(weights.sum(),10) and np.allclose(weights,10/90)
    _, capture = evaluate_profile(np.ones(90),np.arange(90))
    assert np.isclose(capture,10/90)
    assert evaluate_profile(np.ones(90),np.zeros(90)) == (None,None)
    assert np.isclose(evaluate_profile(np.arange(90),np.arange(90))[0],1)
    p = predictors([risk,risk])
    assert np.allclose(p['pairwise_mean'],risk[~np.eye(3,dtype=bool)])
    factor = p['source_class_and_destination_factorized'].reshape(3,2)
    assert np.allclose(factor.sum(axis=1),1-np.diag(risk))
    # A split tie must retain exact budget, independent of array order.
    s=np.r_[np.ones(4)*2,np.ones(12),np.zeros(74)]
    assert np.isclose(topk_weights(s).sum(),10)
    assert np.allclose(topk_weights(s)[4:16],.5)
    print('PASS: margins, confusion distinction, invariance, tied budget, degenerate risk, baseline mass')

if __name__ == '__main__': main()
