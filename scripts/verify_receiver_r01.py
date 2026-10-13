#!/usr/bin/env python3
"""Independent count-level metrics, input/seal/sampler/selection/publication checks."""
import hashlib,json,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from wisig_w01_common import sha,dump


def main():
    public=ROOT/'reports/receiver_r01/2026-10-13';out=ROOT/'outputs/receiver_r01';reg=json.loads((ROOT/'configs/receiver_r01/registration.json').read_text())
    seal=json.loads((ROOT/'configs/receiver_r01/pre_execution_seal.json').read_text())
    for p,h in seal['files'].items():assert sha(ROOT/p)==h,p
    inputs=[]
    for a in reg['input_assets']:
        assert sha(a['path'])==a['sha256'];inputs.append(dict(path=a['path'],sha256=a['sha256']))
    # Independently verify init/data/sampling pairing and each epoch's complete membership.
    checks=[];configs=[]
    for stage in ['search','formal']:
        for job in reg[stage+'_jobs']:
            run=out/stage/job['run_id'];s=json.loads((run/'status.json').read_text());c=json.loads((run/'config.json').read_text());selection=json.loads((run/'selection.json').read_text())
            assert s['state']=='completed' and s['epochs']==job['epochs'] and sha(run/'selected.pt')==s['checkpoint_sha256']==selection['checkpoint_sha256']
            schedule=np.load(run/'schedule.npy',allow_pickle=False);assert sha(run/'schedule.npy')==c['schedule_sha256']
            assert schedule.shape[0]==job['epochs'] and s['steps']==job['epochs']*schedule.shape[1]
            n=c['train_packets'];expected=np.arange(n)
            for epoch in schedule:assert np.array_equal(np.sort(epoch.ravel()),expected)
            c['stage']=stage;configs.append(c);checks.append(dict(run_id=job['run_id'],epochs=job['epochs'],steps=s['steps'],checkpoint_sha256=s['checkpoint_sha256']))
    for key in {(c['dataset'],c['backbone'],c['fold'],c['seed'],c['stage']) for c in configs}:
        cs=[c for c in configs if (c['dataset'],c['backbone'],c['fold'],c['seed'],c['stage'])==key]
        for name in ['initial_sha256','schedule_sha256']:
            assert len({c[name] for c in cs})==1,(key,name)
        for name in ['x','y','rx','day','position']:assert len({c['fitting_data_sha256'][name] for c in cs})==1
        # CE and SWAD optimizer trajectories are identical whenever LR is identical.
        for ce in [c for c in cs if c['method']=='ce']:
            for sw in [c for c in cs if c['method']=='swad' and c['lr']==ce['lr']]:
                cols=['objective','source_val_CE','source_val_accuracy','source_val_macro_f1']
                a=pd.read_csv(out/ce['stage']/ce['run_id']/'history.csv');b=pd.read_csv(out/sw['stage']/sw['run_id']/'history.csv')
                assert np.array_equal(a[cols].to_numpy(),b[cols].to_numpy()),(ce['run_id'],sw['run_id'])
    sel=json.loads((out/'selection.frozen.json').read_text());assert not sel['internal_RX_used'] and not sel['outer_RX_used']
    for key,choice in sel['choices'].items():
        candidates=[r for r in sel['all_candidates'] if r['key']==key];best=min(candidates,key=lambda r:(r['validation_CE'],r['lr']))
        assert best==choice
    # Rebuild all confusion matrices directly from local frozen evaluation predictions.
    perrx=pd.read_csv(public/'per_RX_date.csv');pertx=pd.read_csv(public/'per_RX_TX_date.csv');conf=pd.read_csv(public/'confusion_counts.csv')
    conf_groups={key:g for key,g in conf.groupby(['run_id','role','rx','date'])}
    tx_groups={key:g for key,g in pertx.groupby(['run_id','role','rx','date'])}
    count=0;coverage=[]
    for (run_id,role),g in perrx.groupby(['run_id','role']):
        a=np.load(out/'evaluation'/(run_id+'_'+role+'.npz'),allow_pickle=False)
        for row in g.to_dict('records'):
            mask=(a['rx']==row['rx'])&(a['day']==row['date']);truth=a['y'][mask];pred=a['prediction'][mask]
            cm=np.bincount(truth*10+pred,minlength=100).reshape(10,10);support=cm.sum(1);present=support>0
            accuracy=float(cm.trace()/cm.sum());f1=np.divide(2*cm.diagonal(),cm.sum(0)+support,out=np.zeros(10),where=cm.sum(0)+support>0)
            recall=np.divide(cm.diagonal(),support,out=np.full(10,np.nan),where=present)
            for k,v in [('accuracy',accuracy),('macro_f1',float(f1[present].mean())),('worst_tx_recall',float(np.nanmin(recall)))]:assert abs(row[k]-v)<1e-12,(run_id,k)
            subset=conf_groups[(run_id,role,row['rx'],row['date'])]
            published=subset.pivot(index='tx',columns='predicted_tx',values='count').to_numpy();assert np.array_equal(cm,published)
            cells=tx_groups[(run_id,role,row['rx'],row['date'])].sort_values('tx')
            assert np.array_equal(cells.support,support) and np.allclose(cells.recall,recall,equal_nan=True)
            count+=1
            if row['dataset']=='wisig':coverage.append(dict(rx=row['rx'],date=row['date'],role=role,classes=int(present.sum())))
    # Seed/fold/date aggregation recomputed from count-verified receiver rows.
    combined=pd.read_csv(public/'combined_per_seed.csv')
    for row in combined.to_dict('records'):
        g=perrx[(perrx.dataset==row['dataset'])&(perrx.backbone==row['backbone'])&(perrx.method==row['method'])&(perrx.seed==row['seed'])&(perrx.role==row['role'])]
        rx=g.groupby('rx').accuracy.mean().to_numpy()
        assert len(rx)==row['RX_count'] and abs(rx.mean()-row['accuracy'])<1e-12 and abs(np.quantile(rx,.1)-row['RX_accuracy_q10'])<1e-12
    members=json.loads((public/'evidence_members.json').read_text());assert sha(public/'evidence.zip')==members['archive_sha256']
    with zipfile.ZipFile(public/'evidence.zip') as z:
        assert z.testzip() is None and sorted(z.namelist())==sorted(m['path'] for m in members['members'])
        for m in members['members']:
            assert Path(m['path']).suffix in ['.json','.csv','.log'] and hashlib.sha256(z.read(m['path'])).hexdigest()==m['sha256']
    # Historical tracked files preserved: only authorization doc + new experiment paths may differ.
    base=reg['base_remote_commit'];changed=subprocess.check_output(['git','diff','--name-only',base],cwd=ROOT,text=True).splitlines()
    assert all(p=='AGENTS.md' or '/receiver_r01/' in p or p.endswith('_receiver_r01.py') for p in changed),changed
    result=dict(passed=True,seal_files=len(seal['files']),input_assets=len(inputs),fitting_jobs=len(checks),formal_seeds=reg['formal_seeds'],
        all_epoch_membership_verified=True,all_pairing_verified=True,source_only_LR_selection_verified=True,
        confusion_cells_recomputed=count,evidence_members_verified=len(members['members']),all_missing_recall_is_NA=True,
        historical_files_unchanged=True,final_data_sealed=True,verification_kind='independent count-level recomputation; no new fitting or test-data use')
    dump(public/'verification.json',result)
    archived=['confusion_counts.csv','per_RX_TX_date.csv','paired_RX_TX_changes.csv','update_diagnostics.csv','update_RX_risks.csv']
    files=[p for p in public.iterdir() if p.is_file() and p.name not in ['PUBLIC_ARTIFACT_SHA256.json','EVIDENCE_INDEX.md']+archived]
    sources=[p for p in (ROOT/'src/receiver_r01').rglob('*') if p.is_file() and '__pycache__' not in str(p)]+list((ROOT/'scripts').glob('*receiver_r01.py'))+list((ROOT/'docs/receiver_r01').glob('*.md'))+list((ROOT/'configs/receiver_r01').glob('*.json'))
    manifest=dict(public_files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files+sources},
                  archived_complete_tables={name:dict(member='tables/'+name,sha256=sha(public/name),bytes=(public/name).stat().st_size) for name in archived},
                  source_inputs=inputs,raw_IQ_weights_features_published=False)
    dump(public/'PUBLIC_ARTIFACT_SHA256.json',manifest)
    (public/'EVIDENCE_INDEX.md').write_text('# R0–R1 evidence index\n\nFull source/input and public file SHA256: PUBLIC_ARTIFACT_SHA256.json.\n\nDerived-only evidence.zip: '+sha(public/'evidence.zip')+'\n\n'+str(len(members['members']))+' archive members verified against their SHA256 in evidence_members.json. All 85 job config/status/selection/curves/update-probes and full process logs are included. Complete large confusion_counts, per_RX_TX_date, paired_RX_TX_changes, update_diagnostics and update_RX_risks CSVs are available under tables/ in this ZIP, indexed by archived_complete_tables. Raw IQ, feature arrays, logits and model weights are excluded. Checkpoint hashes are public; checkpoint bytes stay local.\n\nCount-level recomputation and immutable history/seal checks: verification.json. Source selection and paired sampling hashes are public CSV/JSON. Remote publication requires independent commit-tree match plus report/decision readback.\n')
    print(json.dumps(result))

if __name__=='__main__':main()
