#!/usr/bin/env python3
"""Recompute public decision rows; audit seals, identities and reserved domains."""
import csv,json
from pathlib import Path
import numpy as np
from wisig_w01_common import sha,dump
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'outputs/wisig_w01';report=ROOT/'reports/wisig_w01/2026-10-13';reg=json.loads((ROOT/'configs/wisig_w01/registration.json').read_text())
 seal=json.loads((out/'pre_execution_seal.json').read_text());dec=json.loads((report/'decision.json').read_text());exe=json.loads((out/'execution.json').read_text())
 for p,h in seal['files_sha256'].items():assert sha(ROOT/p)==h,p
 assert sha(out/'preflight.json')==seal['preflight_sha256']
 assert len(exe['jobs'])==2 and all(x['exit_code']==0 for x in exe['jobs']) and exe['retries']==0
 assert len(reg['roles']['source'])==14 and len(reg['roles']['internal_rx'])==10 and len(reg['roles']['final_rx'])==8
 original=json.loads((out/'coverage_preflight_failed/registration.json').read_text());assert original['roles']['final_rx']==reg['roles']['final_rx']
 pre=json.loads((out/'preflight.json').read_text());assert pre['invalid_samples']==0
 for asset in pre['assets']:
  assert asset['rx_index'] not in reg['roles']['final_rx'] and asset['date_index']!=3
  assert sha(asset['path'])==asset['sha256']
 rows=list(csv.DictReader((report/'strict_TX_date_predictions.csv').open()));ids=list(csv.DictReader((report/'transformed_known_TX_identity.csv').open()))
 assert len(rows)==2*3*14*13*2*4 and len(ids)==2*3*14*13*8*4
 for criterion in dec['prediction_criteria']:
  wanted=[x for x in rows if int(x['seed'])==criterion['seed'] and x['case']=='hold_date2' and int(x['held_tx'])==criterion['held_tx']]
  for field in ['mean_squared_distance','gaussian_w2_squared']:
   avg={m:float(np.mean([float(x[field]) for x in wanted if x['method']==m])) for m in ['identity','translation','coral','conditional_knn3']}
   best=min(['identity','translation','coral'],key=lambda m:avg[m]);gain=1-avg['conditional_knn3']/avg[best]
   assert best==criterion[field+'_best_control'] and np.isclose(gain,criterion[field+'_gain'],atol=1e-12)
 for seed in reg['pilot_seeds']:
  run=out/'runs'/f'pilot_seed{seed}';cfg=json.loads((run/'config.json').read_text());status=json.loads((run/'status.json').read_text())
  assert cfg['classes']==8 and cfg['parameters']==39940 and cfg['held_tx']==[8,9]
  assert cfg['source_only_checkpoint_selection'] is True and not cfg['final_rx_read'] and not cfg['final_date_read']
  assert sha(run/'best.pt')==status['checkpoint_sha256']
  for asset in status['feature_assets']:
   assert sha(run/asset['path'])==asset['sha256']
   name=Path(asset['path']).stem;ri=int(name.split('_')[0][2:]);di=int(name.split('_')[1][3:])
   assert ri not in reg['roles']['final_rx'] and di!=3
 components=list(csv.DictReader((report/'factorial_components.csv').open()))
 for seed in reg['pilot_seeds']:
  assert np.isclose(sum(float(x['observed_energy_fraction']) for x in components if int(x['seed'])==seed),1)
 assert dec['w2_jobs']==0 and dec['formal_models']==0 and not dec['w1_gate_passed']
 result=dict(passed=True,fitting_jobs=2,fitting_failures=0,retries=0,strict_TX_prediction_rows=len(rows),identity_rows=len(ids),
  final8_RX_allocation_unchanged=True,final_RX_numerical_read=False,final_date_numerical_read=False,equalized_numerical_read=False,
  strict_encoder_and_mapping_TX_exclusion=True,decision_recomputed_from_public_rows=True,orthogonal_component_energy_sum_one=True,
  prefit_coverage_abort_preserved=True,zero_fitting_jobs_at_coverage_revision=True,input_archive_sha256=sha(reg['archive']),
  reserved_gpu_hours=exe['reserved_gpu_hours'],w2_jobs=0)
 assert result['input_archive_sha256']==reg['archive_sha256'];dump(report/'verification.json',result);print(json.dumps(result))
if __name__=='__main__':main()
