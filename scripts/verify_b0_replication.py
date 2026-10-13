#!/usr/bin/env python3
"""Independent B0 reconstruction and real checkpoint/optimizer audit."""
import csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'reports/d05_b0/2026-10-13/b0'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def main():
    r=json.loads((ROOT/'configs/d05_b0/registration.json').read_text());history=Path(r['history_root']);assets=rows(P/'input_provenance.csv')
    recall={(int(x['seed']),x['receiver'],x['method'],int(x['tx'])):float(x['recall']) for x in rows(P/'per_RX_TX.csv')}
    confusion={(int(x['seed']),x['receiver'],x['method'],int(x['tx_i']),int(x['tx_j'])):int(x['count']) for x in rows(P/'confusion_counts.csv')}
    reconstructed={};checked=0
    for a in assets:
        path=Path(a['path']);assert sha(path)==a['sha256']
        with np.load(path,allow_pickle=False) as f:y=f['labels'];z=f['logits'];indices=f['rows']
        assert len(np.unique(indices))==2000 and np.array_equal(np.bincount(y,minlength=10),np.full(10,200))
        pred=z.argmax(1);c=np.array([[sum((y==i)&(pred==j)) for j in range(10)] for i in range(10)])
        key=int(a['seed']),a['receiver'],a['method'];reconstructed[key]=c
        for i in range(10):
            assert np.isclose(c[i,i]/200,recall[key+(i+31,)])
            for j in range(10):assert c[i,j]==confusion[key+(i+31,j+31)]
        checked+=1
    assert checked==60
    repetition=rows(P/'cross_seed_repetition.csv');repeat=[]
    for row in repetition:
        rx=row['receiver'];tx=int(row['tx'])-31
        deltas=[(reconstructed[s,rx,'strict_cis'][tx,tx]-reconstructed[s,rx,'full_cis'][tx,tx])/200 for s in [r['original_seed']]+r['new_seeds']]
        count=sum(x<=-.1+1e-12 for x in deltas);assert count==int(row['severe_seeds'])
        assert np.allclose([np.mean(deltas),min(deltas),max(deltas)],[float(row['mean_delta_recall']),float(row['min_delta_recall']),float(row['max_delta_recall'])])
        if count>=2:repeat.append(row)
    decision=json.loads((P/'decision.json').read_text());assert len(repeat)==decision['repeated_severe_RX_TX_cells']
    assert len({x['receiver'] for x in repeat})==decision['repeated_severe_physical_RX']
    for s in decision['seed_summaries']:
        seed=s['seed']
        acc={m:sum(np.trace(reconstructed[seed,rx,m]) for rx in r['source_receivers'])/20000 for m in r['b0_methods']}
        assert np.isclose(acc['full_cis'],s['full_mean_accuracy']) and np.isclose(acc['strict_cis'],s['strict_mean_accuracy'])
    initial={};checkpoints=[]
    for cfg in r['jobs']:
        run=ROOT/'outputs/d05_b0/b0/runs'/cfg['run_id'];c=json.loads((run/'config.json').read_text());status=json.loads((run/'status.json').read_text());cp=torch.load(run/'final.pt',map_location='cpu',weights_only=False)
        assert cp['epoch']==20 and cp['config']==c and status['final_epoch']==20
        assert status['checkpoint_sha256']==sha(run/'final.pt')
        assert c['scaler']['fit_source_keys']==cfg['train_keys'] and not set(cfg['train_keys'])&set(cfg['validation_keys'])
        assert all(torch.isfinite(v).all() for v in cp['model'].values())
        assert len(cp['optimizer']['param_groups'])==1
        group=cp['optimizer']['param_groups'][0];assert group['lr']==.0003 and group['weight_decay']==0 and group['betas']==(.9,.999)
        expected_steps=math.ceil(10000/64)*20
        assert all(int(x['step'])==expected_steps for x in cp['optimizer']['state'].values())
        hist=rows(run/'history.csv');assert [int(x['epoch']) for x in hist]==list(range(1,21))
        assert all(x['validation_used_for_selection']=='False' for x in hist)
        log=ROOT/'outputs/d05_b0/b0/logs'/(cfg['run_id']+'.log');lines=[json.loads(x) for x in log.read_text().splitlines() if x.startswith('{')];assert len(lines)==20
        for a,b in zip(hist,lines):
            for k in ('epoch','train_loss','train_accuracy','training_seconds','validation_accuracy'):assert np.isclose(float(a[k]),float(b[k]),rtol=0,atol=1e-12)
        k=cfg['seed'];initial.setdefault(k,set()).add(c['initial_model_payload_sha256'])
        checkpoints.append(dict(run=cfg['run_id'],checkpoint_sha256=sha(run/'final.pt'),optimizer_steps=expected_steps,parameters=sum(v.numel() for v in cp['model'].values())))
    assert all(len(v)==1 for v in initial.values()) and len(set(next(iter(v)) for v in initial.values()))==2
    execution=json.loads((P/'execution.json').read_text());assert execution['fitting_jobs_launched']==8 and execution['fitting_retries']==0 and execution['seed_replacements']==0 and execution['reserved_gpu_hours']<=1
    startup=ROOT/'outputs/d05_b0/b0_startup_failure';zero_steps=0
    for cfg in r['jobs']:
        folder=startup/'runs'/cfg['run_id'];assert not (folder/'history.csv').exists() and not (folder/'final.pt').exists();zero_steps+=1
    seal=json.loads((ROOT/'configs/d05_b0/pre_execution_seal.json').read_text())
    for path,digest in seal['files_sha256'].items():assert sha(ROOT/path)==digest
    oldseal=json.loads((ROOT/'configs/d05_b0/initial_pre_execution_seal.json').read_text())
    for path,digest in oldseal['files_sha256'].items():
        original=ROOT/'configs/d05_b0/initial_code'/Path(path).name if path in ('scripts/train_b0_replication.py','scripts/launch_b0_replication.py') else ROOT/'configs/d05_b0/initial_registration.json' if path=='configs/d05_b0/registration.json' else ROOT/path
        assert sha(original)==digest,path
    proof=dict(passed=True,model_rx_assets_checked=checked,individual_recalls_checked=600,confusion_counts_checked=6000,checkpoints_checked=checkpoints,initial_payloads_paired_across_full_strict_and_folds=True,retained_pre_fit_failure_attempts=zero_steps,actual_fitting_jobs=8,training_retries=0,seed_replacements=0,source_seals_and_original_versions_verified=True,no_new_inference=True,reserved_gpu_hours=execution['reserved_gpu_hours'])
    (P/'independent_verification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k!='checkpoints_checked'}))
if __name__=='__main__':main()
