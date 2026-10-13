#!/usr/bin/env python3
"""Post-fit public tables, cross-date descriptions and historical Shen readback."""
import csv,json
from pathlib import Path
import numpy as np
from wisig_w01_common import dump
ROOT=Path(__file__).resolve().parents[1]
def table(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
 report=ROOT/'reports/wisig_w01/2026-10-13';out=ROOT/'outputs/wisig_w01';reg=json.loads((ROOT/'configs/wisig_w01/registration.json').read_text());exe=json.loads((out/'execution.json').read_text())
 summary=[];evaluation=[];history=[];repeat=[]
 for seed in reg['pilot_seeds']:
  run=out/'runs'/f'pilot_seed{seed}';status=json.loads((run/'status.json').read_text());metrics=json.loads((run/'metrics.json').read_text())
  summary.append(dict(seed=seed,epochs=status['epochs'],best_epoch=status['best_epoch'],optimizer_steps=status['optimizer_steps'],train_eval_accuracy=metrics['train_eval']['accuracy'],source_validation_accuracy=metrics['source_validation']['accuracy'],quality_passed=metrics['encoder_quality_passed'],checkpoint_sha256=status['checkpoint_sha256'],reserved_seconds=exe['jobs'][reg['pilot_seeds'].index(seed)]['reserved_seconds']))
  evaluation.extend(dict(seed=seed,**row) for row in metrics['all_evaluation'])
  history.extend(dict(seed=seed,**row) for row in csv.DictReader((run/'history.csv').open()))
  cells={}
  for ri in reg['roles']['source']:
   for di in [0,1,2]:
    with np.load(run/'features'/f'rx{ri:02d}_day{di}.npz',allow_pickle=False) as f:
     for ti in range(10):cells[ti,ri,di]=f['features'][f['y']==ti].mean(0).astype('float64')
  for label,tx in [('encoder_TX',list(range(8))),('strict_held_TX',[8,9]),('all10_descriptive',list(range(10)))]:
   before=np.stack([[np.mean([cells[t,ri,d] for d in [0,1]],axis=0) for ri in reg['roles']['source']] for t in tx]);after=np.stack([[cells[t,ri,2] for ri in reg['roles']['source']] for t in tx])
   def interaction(a):return a-a.mean(1,keepdims=True)-a.mean(0,keepdims=True)+a.mean((0,1),keepdims=True)
   a,b=interaction(before),interaction(after)
   repeat.append(dict(seed=seed,TX_group=label,fit_dates='2021_03_01/2021_03_08',held_date='2021_03_15',interaction_cross_date_cosine=float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))),fit_interaction_mean_square=float(np.mean(a*a)),held_interaction_mean_square=float(np.mean(b*b)),descriptive_not_out_of_TX_prediction=True))
 table(report/'w0_pilot_summary.csv',summary);table(report/'w0_receiver_date_metrics.csv',evaluation);table(report/'w0_learning_curves.csv',history);table(report/'interaction_cross_date_repeatability.csv',repeat)
 coverage=[dict(rx_index=row['rx_index'],date_index=row['date_index'],tx_index=row['tx_index'],valid_packets=row['valid_packets'],missing_class=int(row['valid_packets']=='0')) for row in csv.DictReader((report/'data_audit.csv').open()) if row['role']=='internal_rx' and int(row['tx_index'])<8]
 table(report/'internal_RX_class_coverage.csv',coverage);dump(report/'w0_execution.json',exe)
 units={}
 for row in csv.DictReader((ROOT/'reports/receiver_transform/2026-10-13/held_TX_RX_pair_distances.csv').open()):
  key=tuple(row[k] for k in ['fold','seed','partition','source_rx','target_rx','tx']);units.setdefault(key,{})[row['method']]=row
 count={k:0 for k in ['mean_lt_coral','mean_lt_translation','w2_lt_coral','both_lt_all_controls']}
 for u in units.values():
  get=lambda method,field:float(u[method][field])
  count['mean_lt_coral']+=get('lowrank','mean_squared_distance')<get('coral','mean_squared_distance')
  count['mean_lt_translation']+=get('lowrank','mean_squared_distance')<get('translation','mean_squared_distance')
  count['w2_lt_coral']+=get('lowrank','gaussian_w2_squared')<get('coral','gaussian_w2_squared')
  count['both_lt_all_controls']+=all(get('lowrank',field)<min(get(k,field) for k in ['identity','translation','coral']) for field in ['mean_squared_distance','gaussian_w2_squared'])
 dump(report/'shen_failure_review.json',dict(units=len(units),posthoc_counts=count,centered_prototype_rank_upper_bound=7,feature_dimensions=32,estimation_underdetermined=True,encoder_train_accuracy_min=.494375,rank2_covariance_not_fitted=True,coral_pooled_covariance_fitted=True,causal_failure_contribution_unidentified=True,no_shen_new_training=True))
 print('Public W0 summaries/cross-date descriptions/Shen posthoc counts exported')
if __name__=='__main__':main()
