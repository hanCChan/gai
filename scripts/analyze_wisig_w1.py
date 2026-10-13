#!/usr/bin/env python3
"""Balanced factorial descriptions and strict-TX/date prediction; no fitting."""
import csv,json
from pathlib import Path
import numpy as np
import torch
from wisig_w01_common import dump,sha
from receiver_transform_common import power,covariance,distances
ROOT=Path(__file__).resolve().parents[1]
def table(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def factorial(m):
 mu=m.mean((0,1,2),keepdims=True)
 tx=m.mean((1,2),keepdims=True)-mu;rx=m.mean((0,2),keepdims=True)-mu;day=m.mean((0,1),keepdims=True)-mu
 tr=m.mean(2,keepdims=True)-mu-tx-rx;td=m.mean(1,keepdims=True)-mu-tx-day;rd=m.mean(0,keepdims=True)-mu-rx-day
 triple=m-mu-tx-rx-day-tr-td-rd
 return dict(TX=tx,RX=rx,date=day,TX_RX=tr,TX_date=td,RX_date=rd,TX_RX_date=triple)
def fit(source,target):
 mx=source.mean((0,1));my=target.mean((0,1));d=source.shape[-1]
 a=power(covariance(source.reshape(-1,d),.1),-.5)@power(covariance(target.reshape(-1,d),.1),.5)
 b=my-mx@a
 descriptors=source.mean(1);residual=target.mean(1)-(descriptors@a+b)
 return mx,my,a,b,descriptors,residual
def neighbor_correction(query,descriptors,residual):
 nearest=np.argsort(np.sum((descriptors-query)**2,axis=1),kind='stable')[:3]
 return residual[nearest].mean(0)
def main():
 out=ROOT/'outputs/wisig_w01';report=ROOT/'reports/wisig_w01/2026-10-13'
 reg=json.loads((ROOT/'configs/wisig_w01/registration.json').read_text());exe=json.loads((out/'execution.json').read_text())
 assert exe['state']=='completed' and len(exe['jobs'])==2
 rx=reg['roles']['source'];means_rows=[];effect_rows=[];prediction=[];identity=[];quality=[];repeat_rows=[]
 for seed in reg['pilot_seeds']:
  run=out/'runs'/f'pilot_seed{seed}';status=json.loads((run/'status.json').read_text());metrics=json.loads((run/'metrics.json').read_text())
  assert status['state']=='completed';quality.append(bool(metrics['encoder_quality_passed']))
  for asset in status['feature_assets']:assert sha(run/asset['path'])==asset['sha256']
  cells={}
  for ri in rx:
   for di in [0,1,2]:
    with np.load(run/'features'/f'rx{ri:02d}_day{di}.npz',allow_pickle=False) as f:
     for ti in range(10):
      selected=f['features'][f['y']==ti].astype('float64');assert len(selected)>=40
      cells[ti,ri,di]=selected
  reference=np.concatenate([cells[t,r,d] for t in range(8) for r in rx for d in [0,1]])
  center=reference.mean(0);scale=float(np.sqrt(np.mean((reference-center)**2)));assert scale>1e-10
  cells={k:(v-center)/scale for k,v in cells.items()}
  m=np.stack([[[cells[t,r,d].mean(0) for d in [0,1,2]] for r in rx] for t in range(10)])
  a=np.stack([[[cells[t,r,d][::2].mean(0) for d in [0,1,2]] for r in rx] for t in range(10)])
  b=np.stack([[[cells[t,r,d][1::2].mean(0) for d in [0,1,2]] for r in rx] for t in range(10)])
  total=float(np.mean((m-m.mean((0,1,2),keepdims=True))**2));effects=factorial(m);ea=factorial(a);eb=factorial(b);ed=factorial(a-b)
  energy_sum=0.
  for name,value in effects.items():
   energy=float(np.mean(value**2));noise=.25*float(np.mean(ed[name]**2));energy_sum+=energy
   corr=float(np.sum(ea[name]*eb[name])/max(np.linalg.norm(ea[name])*np.linalg.norm(eb[name]),1e-30))
   effect_rows.append(dict(seed=seed,effect=name,mean_square=energy,observed_energy_fraction=energy/total,
    split_half_noise_estimate=noise,noise_corrected_energy=max(0,energy-noise),split_half_cosine=corr))
  assert np.isclose(energy_sum,total,rtol=1e-8,atol=1e-12)
  noise=.25*float(np.mean((a-b)**2))
  repeat_rows.append(dict(seed=seed,total_cell_mean_variation=total,cell_mean_sampling_uncertainty=noise,
   sampling_uncertainty_fraction=noise/total,packet_split='deterministic even/odd within each cell;not physical noise'))
  for t in range(10):
   for rpos,ri in enumerate(rx):
    for d in [0,1,2]:
     value=cells[t,ri,d]
     means_rows.append(dict(seed=seed,tx_index=t,rx_index=ri,date_index=d,packets=len(value),mean_norm=float(np.linalg.norm(m[t,rpos,d])),mean_cov_trace=float(np.trace(covariance(value))),split_half_mean_squared=float(np.sum((a[t,rpos,d]-b[t,rpos,d])**2))))
  state=torch.load(run/'best.pt',map_location='cpu',weights_only=True);weight=state['head.weight'].numpy().astype('float64');bias=state['head.bias'].numpy().astype('float64')
  for case in ['same_date0','hold_date1','hold_date2']:
   for src in rx:
    for dst in rx:
     if src==dst:continue
     if case=='same_date0':
      sf=np.stack([cells[t,src,0][:20] for t in range(8)]);tf=np.stack([cells[t,dst,0][:20] for t in range(8)])
      query={t:cells[t,src,0][20:] for t in range(10)};target={t:cells[t,dst,0][20:] for t in range(10)}
     elif case=='hold_date1':
      sf=np.stack([cells[t,src,0][:40] for t in range(8)]);tf=np.stack([cells[t,dst,0][:40] for t in range(8)])
      query={t:cells[t,src,1] for t in range(10)};target={t:cells[t,dst,1] for t in range(10)}
     else:
      sf=np.stack([np.concatenate([cells[t,src,d][:40] for d in [0,1]]) for t in range(8)])
      tf=np.stack([np.concatenate([cells[t,dst,d][:40] for d in [0,1]]) for t in range(8)])
      query={t:cells[t,src,2] for t in range(10)};target={t:cells[t,dst,2] for t in range(10)}
     mx,my,ca,cb,descriptors,residual=fit(sf,tf);target_proto=np.stack([target[t].mean(0) for t in range(10)])
     for method in ['identity','translation','coral','conditional_knn3']:
      for t in range(10):
       q=query[t]
       if method=='identity':transformed=q
       elif method=='translation':transformed=q+my-mx
       else:
        transformed=q@ca+cb
        if method=='conditional_knn3':transformed=transformed+neighbor_correction(q.mean(0),descriptors,residual)
       if t in [8,9]:
        mean,w2=distances(transformed,target[t]);retrieval=int(np.argmin(np.sum((target_proto-transformed.mean(0))**2,axis=1)))
        sampling_floor=float(np.trace(covariance(q))/len(q)+np.trace(covariance(target[t]))/len(target[t]))
        prediction.append(dict(seed=seed,case=case,source_rx=src,target_rx=dst,held_tx=t,method=method,
         mean_squared_distance=mean,gaussian_w2_squared=w2,diagnostic_identity_retrieval=int(retrieval==t),sampling_mean_distance_floor=sampling_floor))
       else:
        pred=((transformed*scale+center)@weight.T+bias).argmax(1)
        identity.append(dict(seed=seed,case=case,source_rx=src,target_rx=dst,tx=t,method=method,packets=len(q),correct=int((pred==t).sum()),accuracy=float((pred==t).mean())))
   print('Prediction complete:',seed,case,flush=True)
 table(report/'factorial_components.csv',effect_rows);table(report/'sampling_uncertainty.csv',repeat_rows);table(report/'cell_feature_summaries.csv',means_rows)
 table(report/'strict_TX_date_predictions.csv',prediction);table(report/'transformed_known_TX_identity.csv',identity)
 summaries=[];criteria=[]
 for seed in reg['pilot_seeds']:
  for case in ['same_date0','hold_date1','hold_date2']:
   for tx in [8,9]:
    grouped={}
    for method in ['identity','translation','coral','conditional_knn3']:
     rows=[x for x in prediction if x['seed']==seed and x['case']==case and x['held_tx']==tx and x['method']==method]
     avg={field:float(np.mean([x[field] for x in rows])) for field in ['mean_squared_distance','gaussian_w2_squared','diagnostic_identity_retrieval','sampling_mean_distance_floor']}
     grouped[method]=avg;summaries.append(dict(seed=seed,case=case,held_tx=tx,method=method,**avg))
    if case=='hold_date2':
     entry=dict(seed=seed,held_tx=tx)
     passed=True
     for field in ['mean_squared_distance','gaussian_w2_squared']:
      best=min(['identity','translation','coral'],key=lambda k:grouped[k][field]);gain=1-grouped['conditional_knn3'][field]/max(grouped[best][field],1e-30)
      entry[field+'_gain']=gain;entry[field+'_best_control']=best;passed=passed and gain>=.05
     absolute=min(grouped[k]['mean_squared_distance'] for k in ['identity','translation','coral'])-grouped['conditional_knn3']['mean_squared_distance']
     entry['improvement_exceeds_sampling_floor']=absolute>grouped['conditional_knn3']['sampling_mean_distance_floor'];passed=passed and entry['improvement_exceeds_sampling_floor']
     same=[x for x in identity if x['seed']==seed and x['case']==case]
     accuracy={k:sum(x['correct'] for x in same if x['method']==k)/sum(x['packets'] for x in same if x['method']==k) for k in ['identity','translation','coral','conditional_knn3']}
     entry['original_classifier_accuracy']=accuracy['identity'];entry['conditional_classifier_accuracy']=accuracy['conditional_knn3'];entry['identity_preserved']=accuracy['conditional_knn3']>=accuracy['identity']-.02
     passed=passed and entry['identity_preserved'];entry['passed']=bool(passed);criteria.append(entry)
 table(report/'prediction_summary.csv',summaries)
 supported=all(quality) and all(x['passed'] for x in criteria)
 dump(report/'decision.json',dict(status='W1_SUPPORTED_W2_PLAN_ONLY_AWAIT_APPROVAL' if supported else 'HOLD_W2_ENCODER_INSUFFICIENT' if not all(quality) else 'STOP_TRANSFORMATION_GENERATION_W1_UNSUPPORTED',
  encoder_quality_passed=quality,w1_gate_passed=bool(supported),prediction_criteria=criteria,
  fitting_jobs=2,w2_jobs=0,formal_models=0,reserved_gpu_hours=exe['reserved_gpu_hours'],
  final_rx_numeric_read=False,final_date_numeric_read=False,strict_TX_exclusion=True,
  identity_retrieval_is_labeled_diagnostic=True,conditional_mapping_uses_source_class_distribution=True,
  shen_rank2_stop_preserved=True,causal_RX_hardware_identified=False,official_equivalence_proven=False,
  proposed_12_model_gpu_hours=exe['estimated_12_model_gpu_hours'],
  limitation='Only two strict held identities; two seeds with one fixed RX split. Final8 RX/final date and equalized numerical data remain unopened. W2 is not automatically authorized.'))
 print('W1 complete;W2 not launched',flush=True)
if __name__=='__main__':main()
