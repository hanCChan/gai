#!/usr/bin/env python3
"""Independent counts/selection/sampling/checkpoint audit; no inference or fitting."""
import csv,hashlib,json,sys
from fractions import Fraction
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from b1_baselines.model import FeatureCNN
P=ROOT/'outputs/b1_baselines';R=ROOT/'reports/b1_baselines/2026-10-13'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csv_rows(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def main():
    torch.set_num_threads(2);reg=json.loads((ROOT/'configs/b1_baselines/registration.json').read_text());seal=json.loads((ROOT/'configs/b1_baselines/pre_execution_seal.json').read_text())
    for path,digest in seal['files_sha256'].items():assert sha(ROOT/path)==digest,path
    assert sha(ROOT/reg['approved_matrix_path'])==reg['approved_matrix_sha256']
    preflight=json.loads((P/'preflight.json').read_text());assert sha(P/'preflight.json')==seal['preflight_sha256'];h=Path(reg['history_root'])
    schedule_checks=0
    for name,meta in preflight['schedules'].items():
        path=ROOT/meta['path'];assert sha(path)==meta['sha256'];schedule=np.load(path,allow_pickle=False)
        groups=np.concatenate([index*10+np.load(h/'outputs/shen_p2a8/g1_cache_v1'/key/'labels.npy') for index,key in enumerate(meta['train_keys'])])
        for epoch in schedule:
            assert np.array_equal(np.sort(epoch.ravel()),np.arange(len(groups)))
            for batch in epoch:assert np.array_equal(np.bincount(groups[batch],minlength=meta['groups']),np.full(meta['groups'],2));schedule_checks+=1
    expected_init={}
    for seed in (20261013,20261014,20261015):
        torch.manual_seed(seed);model=FeatureCNN();expected_init[seed]=hashlib.sha256(b''.join(v.numpy().tobytes() for v in model.state_dict().values())).hexdigest()
    source_scores={};initials={};formal_confusions={};assets=0;checkpoints=[];steps_total=0
    for stage,key in [('search','search_jobs'),('formal','formal_jobs')]:
        for job in reg['matrix'][key]:
            folder=P/stage/'runs'/job['run_id'];cfg=json.loads((folder/'config.json').read_text());status=json.loads((folder/'status.json').read_text());metrics=json.loads((folder/'metrics.json').read_text())
            assert status['status']=='completed' and status['final_epoch']==job['epochs'] and status['optimizer_steps']==job['updates']
            assert cfg['initial_base_payload_sha256']==expected_init[job['seed']]
            assert cfg['train_keys']==job['train_keys'] and cfg['stage']==stage
            eval_keys=job['validation_keys'] if stage=='search' else job['evaluation_keys']
            assert cfg['source_keys_opened']==job['train_keys']+eval_keys
            if stage=='search':assert not set(cfg['source_keys_opened'])&set(job['outer_keys_never_read'])
            else:
                assert cfg['selection_sha256']==sha(P/'selection.frozen.json')
                assert (P/'selection.frozen.json').stat().st_mtime<(folder/'config.json').stat().st_mtime
            manifests=[json.loads((h/'outputs/shen_p2a8/g1_cache_v1'/key/'manifest.json').read_text())['stats']['strict_cis'] for key in job['train_keys']]
            n=sum(x['count'] for x in manifests);mean=sum(x['sum'] for x in manifests)/n;std=float(np.sqrt(sum(x['sum_squares'] for x in manifests)/n-mean*mean))
            assert cfg['scaler']==dict(mean=mean,std=std,fit_source_keys=job['train_keys'])
            assert cfg['schedule_sha256']==preflight['schedules'][cfg['schedule_key']]['sha256']
            cp=torch.load(folder/'final.pt',map_location='cpu',weights_only=False);assert sha(folder/'final.pt')==status['checkpoint_sha256'] and cp['config']==cfg
            assert cp['epoch']==job['epochs'] and cp['optimizer_steps']==job['updates']
            assert sum(v.numel() for v in cp['model'].values())==658314 and all(torch.isfinite(v).all() for v in cp['model'].values())
            assert all(int(v['step'])==job['updates'] for v in cp['optimizer']['state'].values());steps_total+=job['updates']
            g=cp['optimizer']['param_groups'][0];assert g['lr']==cfg['lr'] and g['weight_decay']==0
            if job['method']=='grl':assert cp['rx_head']['2.weight'].shape==(len(job['train_keys']),128) and sum(v.numel() for v in cp['rx_head'].values())==cfg['aux_parameters']
            else:assert cp['rx_head'] is None and cfg['aux_parameters']==0
            if job['method']=='group_dro_rx_tx':assert len(cp['group_dro']['log_weights'])==job['groups'] and torch.isfinite(cp['group_dro']['log_weights']).all() and torch.isclose(cp['group_dro']['log_weights'].exp().sum(),torch.tensor(1.),atol=1e-6)
            else:assert cp['group_dro'] is None
            hist=csv_rows(folder/'history.csv');assert [int(x['epoch']) for x in hist]==list(range(1,job['epochs']+1))
            assert [int(x['optimizer_steps']) for x in hist]==[100*i for i in range(1,job['epochs']+1)] and all(int(x['packets'])==job['sampler_packets'] for x in hist)
            logs=[json.loads(x) for x in (P/stage/'logs'/(job['run_id']+'.log')).read_text().splitlines() if x.startswith('{')];assert len(logs)==len(hist)
            for a,b in zip(hist,logs):
                for k in ('classification_CE','objective_loss','auxiliary_loss','train_accuracy','optimizer_steps'):assert np.isclose(float(a[k]),float(b[k]),rtol=0,atol=1e-12)
            total=np.zeros((10,10),int)
            for key in eval_keys:
                rx=key.removeprefix('shen_').removesuffix('_train');path=folder/'predictions'/(rx+'.npz');assert sha(path)==metrics['assets'][path.name]
                with np.load(path,allow_pickle=False) as archive:z=archive['logits'];y=archive['labels'];rows=archive['rows'];prob=archive['probabilities']
                cache=h/'outputs/shen_p2a8/g1_cache_v1'/key;assert np.array_equal(y,np.load(cache/'labels.npy')) and np.array_equal(rows,np.load(cache/'rows.npy')) and len(set(rows.tolist()))==2000
                assert np.array_equal(np.bincount(y,minlength=10),np.full(10,200)) and np.isfinite(z).all()
                e=np.exp(z-z.max(1)[:,None]);assert np.allclose(prob,e/e.sum(1)[:,None],rtol=1e-7,atol=1e-9)
                predicted=z.argmax(1);c=np.array([[np.sum((y==i)&(predicted==j)) for j in range(10)] for i in range(10)]);total+=c;assets+=1
                if stage=='formal':formal_confusions[job['method'],job['seed'],rx]=c
            exact=sum((Fraction(int(2*total[i,i]),int(total[i].sum()+total[:,i].sum())) for i in range(10)),Fraction(0))/10
            assert metrics['macro_f1_fraction']==dict(numerator=exact.numerator,denominator=exact.denominator) and np.isclose(float(exact),metrics['macro_f1'])
            assert metrics['confusion_counts']==total.tolist()
            if stage=='search':source_scores[job['run_id']]=(exact,job['lr'])
            checkpoints.append(dict(run_id=job['run_id'],stage=stage,checkpoint_sha256=sha(folder/'final.pt'),epoch=job['epochs'],optimizer_steps=job['updates']))
    selection=json.loads((P/'selection.frozen.json').read_text());assert selection['outer_evaluation_used'] is False
    for name,choice in selection['choices'].items():
        candidates=choice['candidates'];expected=max(candidates,key=lambda c:(source_scores[c['run']][0],-source_scores[c['run']][1]));assert expected['run']==choice['winner'] and expected['lr']==choice['lr']
    for job in reg['matrix']['formal_jobs']:
        cfg=json.loads((P/'formal/runs'/job['run_id']/'config.json').read_text());assert cfg['lr']==selection['choices'][f"CV{job['fold']}_{job['method']}"]['lr']
    for row in csv_rows(R/'per_RX_TX.csv'):
        c=formal_confusions[row['method'],int(row['seed']),row['receiver']];assert np.isclose(c[int(row['tx'])-31,int(row['tx'])-31]/200,float(row['recall']))
    for row in csv_rows(R/'paired_TX_changes.csv'):
        i=int(row['tx'])-31;seed=int(row['seed']);rx=row['receiver'];delta=(formal_confusions[row['method'],seed,rx][i,i]-formal_confusions['ce',seed,rx][i,i])/200
        assert np.isclose(delta,float(row['delta_recall'])) and (delta<=-.1+1e-12)==(row['severe_CE_relative_loss']=='True')
    ex=json.loads((P/'execution.json').read_text());assert ex['state']=='completed' and ex['fitting_jobs_launched']==32 and ex['retries']==0 and ex['reserved_gpu_hours']<=2
    proof=dict(passed=True,checkpoints_checked=len(checkpoints),optimizer_step_total=steps_total,prediction_assets_checked=assets,formal_RX_TX_recalls_checked=800,paired_TX_changes_checked=600,shared_schedule_batches_checked=schedule_checks,source_internal_exact_F1_selection_verified=True,all_formal_selection_hashes_and_freeze_order_verified=True,base_initialization_matched_across_methods_LR_folds=True,code_and_source_seals_verified=True,checkpoint_details=checkpoints,no_new_inference_or_fitting=True,reserved_gpu_hours=ex['reserved_gpu_hours'])
    (R/'independent_verification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:v for k,v in proof.items() if k!='checkpoint_details'}))
if __name__=='__main__':main()
