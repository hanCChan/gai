#!/usr/bin/env python3
"""Freeze metadata-only splits, then audit/cache authorized unequ­alized cells."""
import csv,json,datetime
from pathlib import Path
import numpy as np
from wisig_w01_common import PlainReader,load_archive,sha,dump
ROOT=Path(__file__).resolve().parents[1]
def table(path,rows):
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
 out=ROOT/'outputs/wisig_w01';out.mkdir(parents=True,exist_ok=False);(out/'cache').mkdir()
 public=ROOT/'reports/wisig_w01'/datetime.date.today().isoformat();public.mkdir(parents=True,exist_ok=False)
 archive=Path('/workspace/rl1/l0_local/data/wisig-manyrx-kaggle/ManyRx.kaggle.v1.zip')
 assert sha(archive)=='b0eca705816b03a47290286fd23ca1a10e02c6f9bd29b5461cfd92c13a86fbdd'
 hardware=Path('/workspace/rl1/l0_local/wisig-evidence/orbit_hardware.pkl')
 with hardware.open('rb') as f:txhw,rxhw=PlainReader(f).load()
 data=load_archive(archive);rxs=data['rx_list'];txs=data['tx_list'];dates=data['capture_date_list']
 assert len(rxs)==32 and len(txs)==10 and len(dates)==4 and data['equalized_list']==[0,1]
 coverage=[]
 for ri,rx in enumerate(rxs):
  counts=[data['data'][ti][ri][di][0].shape[0] for ti in range(10) for di in range(3)]
  coverage.append(dict(rx_index=ri,rx=rx,hardware=rxhw.get(rx,'UNKNOWN'),min_development_cell=min(counts),development_packets=sum(counts)))
 assert all(x['hardware']!='UNKNOWN' for x in coverage)
 ordered=sorted(coverage,key=lambda x:(x['hardware'],-x['min_development_cell'],-x['development_packets'],x['rx']))
 pattern=['source','source','internal_rx','final_rx']
 for i,row in enumerate(ordered):row['role']=pattern[i%4]
 # Preserve the metadata-only final allocation; incomplete development RX become internal.
 for row in ordered:
  if row['role']!='final_rx':row['role']='source' if row['min_development_cell']>=160 else 'internal_rx'
 roles={role:[x['rx_index'] for x in ordered if x['role']==role] for role in pattern}
 assert len(roles['source'])==14 and len(roles['internal_rx'])==10 and len(roles['final_rx'])==8
 reg=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),date_shanghai='2026-10-13',
  archive=str(archive),archive_sha256=sha(archive),hardware_sha256=sha(hardware),
  official_model_source='https://github.com/WiSig-dataset/wisig-examples/blob/master/py/d003_ManyRx_nrx.py',
  official_model_source_sha256=sha('/workspace/rl1/l0_local/wisig-evidence/py_d003_ManyRx_nrx.py'),
  rx_names=rxs,tx_names=txs,tx_hardware=[txhw.get(t,'UNKNOWN') for t in txs],dates=dates,roles=roles,hardware_and_coverage=ordered,
  split_rule='hardware/metadata coverage then repeating source/source/internal/final freezes final8; remaining24 with minimum cell>=160 become source14, remaining10 internal',
  train_dates=[0,1],w1_hold_date=2,final_date=3,encoder_tx=list(range(8)),strict_held_tx=[8,9],
  prior_strict_tx_exclusion_preserved=True,encoder_positions=[0,120],validation_positions=[120,160],same_rx_test_positions=[160,200],
  w1_positions=[120,200],unequalized_only=True,numeric_final_rx_or_date_access=False,
  pilot_seeds=[20261013,20261014],maximum_fitting_jobs=2,lr=.0005,max_epochs=100,patience=5,batch_groups='source_RX x train_date x encoder_TX;two packets/group;each packet once/epoch',
  l2_dense=.0001,dropout=.5,adam_epsilon=1e-7,pilot_timeout_seconds=1800,
  encoder_quality_gate=dict(frozen_training_accuracy=.90,source_validation_accuracy=.85),
  w1_prediction='Identity/Translation/CORAL;fixed3-neighbor class-conditioned CORAL mean residual, diagnostic only, no rank model',
  w1_gate=dict(mean_distance_gain=.05,w2_distance_gain=.05,identity_noninferiority=.02,requires_both_seeds_and_both_held_TX_and_date_holdout=True),
  w2_authorized_now=False,w2_plan_only=True,final_rx_and_date_reserved=True)
 (ROOT/'configs/wisig_w01').mkdir(parents=True,exist_ok=True)
 dump(ROOT/'configs/wisig_w01/registration.json',reg);table(public/'rx_hardware_and_split.csv',ordered)
 audit=[];invalid=[];assets=[]
 for ri in roles['source']+roles['internal_rx']:
  for di in [0,1,2]:
   blocks=[];labels=[];positions=[]
   for ti in range(10):
    a=data['data'][ti][ri][di][0];assert a.shape[1:]==(256,2)
    value=a.numeric();assert value.shape[0]<=200
    finite=np.isfinite(value).all(axis=(1,2));energy=np.mean(np.sum(value**2,axis=2),axis=1)
    good=finite&np.isfinite(energy)&(energy>0)
    for position in np.flatnonzero(~good):invalid.append(dict(rx_index=ri,date_index=di,tx_index=ti,position=int(position),reason='nonfinite_or_zero_power'))
    normalized=value[good]/np.sqrt(energy[good])[:,None,None]
    blocks.append(normalized[:,None].astype('float32'));labels.extend([ti]*int(good.sum()));positions.extend(np.flatnonzero(good).tolist())
    audit.append(dict(rx_index=ri,rx=rxs[ri],role=next(x['role'] for x in ordered if x['rx_index']==ri),date_index=di,date=dates[di],tx_index=ti,tx=txs[ti],packets=a.shape[0],valid_packets=int(good.sum()),invalid_packets=int((~good).sum()),dtype=a.dtype))
   p=out/'cache'/f'rx{ri:02d}_day{di}.npz';np.savez(p,x=np.concatenate(blocks),y=np.array(labels,dtype=np.int64),position=np.array(positions,dtype=np.int64))
   assets.append(dict(path=str(p),sha256=sha(p),rx_index=ri,date_index=di))
  print('Audited development RX:',ri,rxs[ri],rxhw[rxs[ri]],flush=True)
 table(public/'data_audit.csv',audit)
 if invalid:table(public/'invalid_samples.csv',invalid)
 else:(public/'invalid_samples.csv').write_text('rx_index,date_index,tx_index,position,reason\n')
 for row in audit:
  if row['role']=='source':assert row['valid_packets']>=160,'Source coverage insufficient for frozen split'
 dump(out/'preflight.json',dict(assets=assets,invalid_samples=len(invalid),audited_packets=sum(x['packets'] for x in audit),
  final_rx_numeric_read=False,final_date_numeric_read=False,equalized_numeric_read=False,
  all_pickle_payload_bytes_read_opaquely=True,no_untrusted_global_execution=True))
 sources=['scripts/wisig_w01_common.py','scripts/prepare_wisig_w01.py','scripts/train_wisig_w0.py','scripts/analyze_wisig_w1.py','scripts/launch_wisig_w0.py','docs/wisig_w01/EXECUTION_PLAN.md','configs/wisig_w01/registration.json']
 dump(out/'pre_execution_seal.json',dict(before_fitting=True,files_sha256={p:sha(ROOT/p) for p in sources},preflight_sha256=sha(out/'preflight.json')))
 print('Data validity and frozen split passed',flush=True)
if __name__=='__main__':main()
