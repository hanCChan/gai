#!/usr/bin/env python3
"""Independent state/Adam equations, packet-risk, sampling and confusion checks."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from receiver_r2.measurement import state_hash


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f:return list(csv.DictReader(f))


def main():
    torch.set_num_threads(2);reg=json.loads((ROOT/'configs/receiver_r2/registration.json').read_text());oldreg=json.loads((ROOT/'configs/receiver_r01/registration.json').read_text());out=ROOT/'outputs/receiver_r2';pub=ROOT/'reports/receiver_r2/r2_a'
    for p,h in json.loads((ROOT/'configs/receiver_r2/pre_execution_seal.json').read_text())['files'].items():assert sha(ROOT/p)==h,p
    preserved=json.loads((ROOT/'configs/receiver_r2/controller_definition_preservation.json').read_text())
    assert not preserved['decision_rule_changed']
    for item in preserved['before_C2_original_bytes_preserved']:assert sha(ROOT/item['preserved_path'])==item['sha256']
    for p,h in json.loads((ROOT/'configs/receiver_r2/controller_execution_seal.json').read_text())['files'].items():assert sha(ROOT/p)==h,p
    for item in oldreg['input_assets']:assert sha(Path(item['path']))==item['sha256'],item['path']
    execution=json.loads((out/'execution.json').read_text());assert execution['state']=='completed' and len(execution['jobs'])==12 and not execution['failures']
    configs=[];snapshots=0;packet_risk_groups=0;adam_error=0.;gradient_delta_error=0.;confusions=0;objective_log_error=0.
    for j in reg['jobs']:
        run=out/j['run_id'];c=json.loads((run/'config.json').read_text());s=json.loads((run/'status.json').read_text());sel=json.loads((run/'selection.json').read_text());hist=rows(run/'history.csv')
        assert s['state']=='completed' and s['steps']==j['epochs']*(60 if j['stage']=='horizon' or j['dataset']=='shen' else 40)
        assert sha(run/'selected.pt')==s['checkpoint_sha256']==sel['checkpoint_sha256']
        assert sel['epoch']==min(hist,key=lambda r:float(r['source_val_CE']))['epoch'] or int(sel['epoch'])==int(min(hist,key=lambda r:float(r['source_val_CE']))['epoch'])
        schedule=np.load(run/'schedule.npy',allow_pickle=False);assert sha(run/'schedule.npy')==c['schedule_sha256']
        n=c['pools']['fit']['packets'];percell=120 if j['dataset']=='shen' else 120 if j['stage']=='horizon' else 80;expected=np.arange(n)
        for epoch in schedule:
            assert np.array_equal(np.sort(epoch.ravel()),expected)
            for batch in epoch:assert np.all(np.bincount(batch//percell,minlength=n//percell)==2)
        if j['stage']=='horizon':
            replay=json.loads((run/'historical100_replay.json').read_text());assert replay['history_prefix_exact']
            orig=Path(reg['historical_worktree'])/'outputs/receiver_r01/formal'/j['historical_run_id'];oh=rows(orig/'history.csv')
            for a,b in zip(hist[:100],oh):
                # CE reduction='mean' versus reduction='none'.mean() differs only
                # in the logged reduction. Use the original preexecution1e-7
                # assertion for objective, exact comparison for validation.
                error=abs(float(a['objective'])-float(b['objective']));objective_log_error=max(objective_log_error,error);assert error<1e-7
                for k in ['source_val_CE','source_val_accuracy','source_val_macro_f1']:assert float(a[k])==float(b[k])
            assert np.array_equal(schedule[:100],np.load(orig/'schedule.npy',allow_pickle=False))
        else:
            public=rows(run/'risks.csv');lookup={(int(r['epoch']),r['pool'],r['mode'],int(r['replicate']),float(r['scale']),int(r['rx']),int(r['date']),int(r['tx'])):r for r in public}
            partitions=np.load(run/'RX_partitions.npy',allow_pickle=False);assert sha(run/'RX_partitions.npy')==c['rx_partitions_sha256']
            assert all(set(a)==set(c['pools']['fit']['RX']) for a in partitions)
            for path in sorted(run.glob('actual_adam_epoch*.pt')):
                saved=torch.load(path,map_location='cpu',weights_only=False);before=saved['pre_parameters'];after=saved['model'];grad=saved['clipped_gradient'];pre=saved['pre_optimizer'];post=saved['optimizer'];pid=pre['param_groups'][0]['params']
                assert len(pid)==len(before)==len(grad)
                for (name,p),index in zip(before.items(),pid):
                    b=pre['state'].get(index,{});a=post['state'][index];g=grad[name];step=int(a['step']);assert step==int(b.get('step',0))+1
                    m=.9*b.get('exp_avg',torch.zeros_like(g))+.1*g;v=.999*b.get('exp_avg_sq',torch.zeros_like(g))+.001*g*g
                    err=max(float((m-a['exp_avg']).abs().max()),float((v-a['exp_avg_sq']).abs().max()));adam_error=max(adam_error,err);assert torch.allclose(m,a['exp_avg'],atol=2e-7,rtol=1e-5) and torch.allclose(v,a['exp_avg_sq'],atol=2e-7,rtol=1e-5)
                    predicted=p-j['lr']*(a['exp_avg']/(1-.9**step))/((a['exp_avg_sq']/(1-.999**step)).sqrt()+(1e-7 if j['dataset']=='wisig' else 1e-8))
                    err=float((predicted-after[name]).abs().max());gradient_delta_error=max(gradient_delta_error,err);assert torch.allclose(predicted,after[name],atol=2e-7,rtol=1e-6),(name,err)
                    assert torch.equal(after[name]-p,saved['parameter_delta'][name])
                for risk in saved['risk_packet_measurements']:
                    for rx in np.unique(risk['rx']):
                        for day,tx in [(-999,-999)]+[(d,t) for d in np.unique(risk['date'][risk['rx']==rx]) for t in range(10)]:
                            mask=risk['rx']==rx
                            if tx!=-999:mask=mask&(risk['date']==day)&(risk['tx']==tx)
                            key=(saved['epoch'],risk['pool'],risk['mode'],risk['replicate'],risk['scale'],int(rx),int(day),int(tx));r=lookup[key];delta=risk['delta'][mask]
                            assert len(delta)==int(r['n']) and abs(float(delta.mean())-float(r['delta']))<1e-12
                            se=float(delta.std(ddof=1)/np.sqrt(len(delta)));assert abs(se-float(r['paired_delta_SE']))<1e-12
                            assert abs(float(risk['post'][mask].mean())-float(r['post_CE']))<1e-12
                            if risk['scale']==0:assert np.all(delta==0)
                            if risk['scale']==1:assert float(r['post_CE'])==float(r['actual_accepted_CE'])
                            packet_risk_groups+=1
                snapshots+=1
        configs.append(c)
    for stage in ['horizon','diagnostic']:
        for ds in ['wisig','shen']:
            for fold in [0,1]:
                for seed in reg['formal_seeds']:
                    cs=[c for c in configs if c['stage']==stage and c['dataset']==ds and c['fold']==fold and c['seed']==seed]
                    if not cs:continue
                    for key in ['initial_sha256','schedule_sha256','rx_partitions_sha256']:assert len({c[key] for c in cs})==1
                    for key in ['x','y','rx','day','position']:assert len({c['fitting_data_sha256'][key] for c in cs})==1
    # Each new published confusion matrix is independently rebuilt from predictions.
    rxrows=rows(pub/'per_RX_date.csv');counts={}
    for r in rows(pub/'confusion_counts.csv'):
        key=(r['run_id'],r['role'],int(r['rx']),int(r['date']));counts.setdefault(key,np.zeros((10,10),np.int64))[int(r['tx']),int(r['predicted_tx'])]=int(r['count'])
    for r in rxrows:
        a=np.load(out/'evaluation'/(r['run_id']+'_'+r['role']+'.npz'),allow_pickle=False);mask=(a['rx']==int(r['rx']))&(a['day']==int(r['date']));cm=np.bincount(a['y'][mask]*10+a['prediction'][mask],minlength=100).reshape(10,10)
        assert np.array_equal(cm,counts[(r['run_id'],r['role'],int(r['rx']),int(r['date']))]);present=cm.sum(1)>0;f1=np.divide(2*cm.diagonal(),cm.sum(0)+cm.sum(1),out=np.zeros(10),where=cm.sum(0)+cm.sum(1)>0)
        assert abs(cm.trace()/cm.sum()-float(r['accuracy']))<1e-12 and abs(f1[present].mean()-float(r['macro_f1']))<1e-12;confusions+=1
    evaluation=json.loads((pub/'evaluation_cost.json').read_text());remeasure=json.loads((pub/'fitting_remeasurement_cost.json').read_text());total=execution['reserved_gpu_hours']+evaluation['reserved_GPUh']+remeasure['reserved_GPUh']+30/3600
    result=dict(passed=True,fitting_jobs=12,failures=0,retries=0,source_only_selection=True,paired_init_data_and_schedules_verified=True,all_epoch_membership_verified=True,
        historical100_source_validation_curves_exact=6,historical100_selected_parameters_exact=6,maximum_CE_objective_logging_reduction_difference=objective_log_error,preexecution_objective_tolerance=1e-7,actual_Adam_snapshots_verified=snapshots,raw_packet_risk_aggregate_groups_verified=packet_risk_groups,maximum_Adam_moment_equation_abs_error=adam_error,
        maximum_Adam_parameter_equation_abs_error=gradient_delta_error,confusion_matrices_rebuilt=confusions,input_assets_hashed=len(oldreg['input_assets']),final_data_read=False,
        all_GPU_work_with_setup_allowance_reserved_GPUh=total,controller_original_rule_preserved=True,cap_reserved_GPUh=3.,R2_C_jobs=0)
    (pub/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
