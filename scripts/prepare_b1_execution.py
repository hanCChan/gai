#!/usr/bin/env python3
"""Hash-check approved sources and save one common schedule per stage/fold/seed."""
import datetime,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from b1_baselines.data import schedule_from_groups,validate_schedule

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def main():
    r=json.loads((ROOT/'configs/b1_baselines/registration.json').read_text())
    assert sha(ROOT/r['approved_matrix_path'])==r['approved_matrix_sha256']
    out=ROOT/'outputs/b1_baselines';out.mkdir(parents=True,exist_ok=False);(out/'schedules').mkdir();checks=[]
    h=Path(r['history_root'])
    for rel,expected in r['feature_assets'].items():
        actual=sha(h/rel);assert actual==expected['sha256'],rel
        checks.append(dict(path=rel,sha256=actual,bytes=(h/rel).stat().st_size))
    schedules={}
    for stage,jobkey in [('search','search_jobs'),('formal','formal_jobs')]:
        for job in r['matrix'][jobkey]:
            key=f"{stage}_CV{job['fold']}_seed{job['seed']}"
            if key in schedules:continue
            assert all(k in [f'shen_{rx}_train' for rx in r['source_receivers']] for k in job['train_keys'])
            labels=[np.load(h/'outputs/shen_p2a8/g1_cache_v1'/k/'labels.npy',allow_pickle=False) for k in job['train_keys']]
            groups=np.concatenate([rx*10+y for rx,y in enumerate(labels)])
            batches=schedule_from_groups(groups,job['epochs'],job['seed']);validate_schedule(batches,groups,job['epochs'])
            file=out/'schedules'/(key+'.npy');np.save(file,batches,allow_pickle=False)
            schedules[key]=dict(path=str(file.relative_to(ROOT)),sha256=sha(file),shape=list(batches.shape),train_keys=job['train_keys'],epochs=job['epochs'],seed=job['seed'],groups=job['groups'],packets=job['sampler_packets'],group_id_sha256=hashlib.sha256(groups.tobytes()).hexdigest(),all_packets_once_per_epoch=True,each_group_twice_each_batch=True)
    assert len(schedules)==6
    provenance=dict(prepared_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),feature_assets_checked=checks,schedules=schedules,no_model_fit=True,source_receivers=10,no_raw_IQ_access=True)
    (out/'preflight.json').write_text(json.dumps(provenance,indent=2)+'\n')
    files=['configs/b1_baselines/registration.json','docs/b1_baselines/PREREGISTRATION.md','scripts/prepare_b1_execution.py','scripts/train_b1_baseline.py','scripts/launch_b1_baselines.py','scripts/test_b1_baselines.py','src/b1_baselines/__init__.py','src/b1_baselines/objectives.py','src/b1_baselines/data.py','src/b1_baselines/model.py','src/shen_p2a/models.py','src/shen_evidence/models.py','src/shen_evidence/runner.py','src/shen_p2a8/g1_data.py']
    seal=dict(sealed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),before_first_B1_optimization=True,files_sha256={p:sha(ROOT/p) for p in files},preflight_sha256=sha(out/'preflight.json'))
    (ROOT/'configs/b1_baselines/pre_execution_seal.json').write_text(json.dumps(seal,indent=2)+'\n')
    print('Preflight passed:',len(checks),'source assets,6 verified shared schedules; code/definitions sealed')
if __name__=='__main__':main()
