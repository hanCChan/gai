#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from d05_b0.damage import damage_predictors,evaluate_damage
from d0_class_pair.metrics import topk_weights
def main():
    actual=np.r_[.2,-.8,np.zeros(88)]
    score=np.arange(90,dtype=float);score[0]=100
    r=evaluate_damage(score,actual)
    assert np.isclose(r['positive_mass_capture'],1) and np.isclose(r['net_delta_mass'],-.6)
    assert np.isclose(r['positive_delta_mass'],.2) and np.isclose(r['negative_delta_mass'],.8)
    assert np.isclose(evaluate_damage(np.ones(90),actual)['positive_mass_capture'],10/90)
    assert evaluate_damage(score,-np.ones(90))['positive_mass_capture'] is None
    f=np.array([[[.8,.1,.1],[.2,.7,.1],[.1,.1,.8]]]*4)
    s=f.copy();s[:,0,0]-=.1;s[:,0,1]+=.1
    p=damage_predictors(f,s)
    assert np.allclose(p['signed_delta_mean'],[.1,0,0,0,0,0])
    assert np.allclose(p['recall_drop_only'],[.05,.05,0,0,0,0])
    # Average signed change differs from averaging clipped reference damage.
    s[0,0,0]+=.4;s[0,0,1]-=.4
    p=damage_predictors(f,s);assert abs(p['signed_delta_mean'][0])<1e-15
    assert np.isclose(topk_weights(np.ones(90)).sum(),10)
    print('PASS: signed vs gross positive mass, undefined zero harm, tie budget, signed-before-clipping, class baseline')
if __name__=='__main__':main()
