"""Frozen extra-error targets; all rankings use reference RX only."""
import numpy as np
from d0_class_pair.metrics import offdiag, predictors, topk_weights, evaluate_profile

def damage_predictors(full, strict):
    full=np.asarray(full);strict=np.asarray(strict)
    assert full.shape==strict.shape and full.ndim==3
    delta=strict-full
    mean=np.mean(delta,axis=0).copy();np.fill_diagonal(mean,0)
    f=predictors(full);s=predictors(strict)
    positive=np.maximum(delta,0).copy()
    for mat in positive:np.fill_diagonal(mat,0)
    p=predictors(positive)
    c=len(mean)
    return {'signed_delta_mean':offdiag(mean),
            'recall_drop_only':np.repeat(mean.sum(1)/(c-1),c-1),
            'class_difficulty_full':f['source_class_only'],
            'absolute_full_confusion':f['pairwise_mean'],
            'absolute_strict_confusion':s['pairwise_mean'],
            'factorized_delta':s['source_class_and_destination_factorized']-f['source_class_and_destination_factorized'],
            'positive_mass_factorized':p['source_class_and_destination_factorized'],
            'uniform':np.ones(c*(c-1))}

def evaluate_damage(score,signed_target,k=10):
    rho,_=evaluate_profile(score,signed_target,k)
    positive=np.maximum(signed_target,0)
    capture=None if positive.sum()==0 else float(topk_weights(score,k)@positive/positive.sum())
    return dict(signed_spearman=rho,positive_mass_capture=capture,
                net_delta_mass=float(signed_target.sum()),positive_delta_mass=float(positive.sum()),
                negative_delta_mass=float(-np.minimum(signed_target,0).sum()))
