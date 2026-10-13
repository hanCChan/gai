#!/usr/bin/env python3
"""Post-audit floating-tie sensitivity; never changes frozen D0.5 metrics."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'reports/d05_b0/2026-10-13/d05'
def main():
    path=P/'prediction_vectors.csv';v=pd.read_csv(path);results=[]
    keys=['probe_receiver','fold','control','predictor']
    for values,g in v.groupby(keys):
        g=g.sort_values(['tx_i','tx_j']);s=np.round(g.predicted_score.to_numpy(),12);y=np.round(g.observed_signed_delta.to_numpy(),12)
        positive=np.maximum(y,0);cut=np.sort(s)[-10];w=(s>cut).astype(float);ties=s==cut;w[ties]=(10-w.sum())/ties.sum()
        rho=float(spearmanr(s,y)[0]) if np.ptp(s)>0 and np.ptp(y)>0 else None
        capture=float(w@positive/positive.sum()) if positive.sum()>0 else None
        results.append(dict(zip(keys,values),rounded_spearman=rho,rounded_capture=capture))
    frame=pd.DataFrame(results);original=pd.read_csv(P/'profile_evaluation.csv');joined=frame.merge(original,on=keys)
    primary=[]
    for fold in (0,1):
        selected=frame[(frame.fold==fold)&(frame.control=='same_paired_cnn_loro')]
        pair=selected[selected.predictor=='signed_delta_mean']
        pivot=selected.pivot(index='probe_receiver',columns='predictor',values='rounded_capture')
        primary.append(dict(fold=fold,median_signed_spearman=float(pair.rounded_spearman.median()),median_capture=float(pair.rounded_capture.median()),median_gain_over={b:float((pivot.signed_delta_mean-pivot[b]).median()) for b in ('recall_drop_only','class_difficulty_full','factorized_delta','absolute_full_confusion','absolute_strict_confusion','uniform')}))
    frozen=json.loads((P/'decision.json').read_text())['primary_summaries']
    stable=all(np.isclose(new['median_capture'],old['median_positive_mass_capture']) and all(np.sign(gain)==np.sign(old['median_capture_gain_over_'+base]) for base,gain in new['median_gain_over'].items()) for new,old in zip(primary,frozen))
    value=dict(scope='post-audit numeric sensitivity only, not a new primary definition or gate',round_decimals=12,reason='probability subtraction and averaging may create floating micro-differences among mathematically tied count differences',profiles_checked=len(results),input_vector_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),max_individual_capture_change=float(abs(joined.rounded_capture-joined.positive_mass_capture).max()),max_individual_spearman_change=float(abs(joined.rounded_spearman-joined.signed_spearman).max()),primary=primary,primary_capture_medians_and_paired_gain_directions_unchanged=stable,frozen_outputs_modified=False,new_training_or_inference=0)
    (P/'numeric_tie_sensitivity.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
if __name__=='__main__':main()
