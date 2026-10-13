#!/usr/bin/env python3
"""CPU-only extraction from80 existing Adam snapshots; no fitting/inference."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(2);reg=json.loads((ROOT/'configs/receiver_c0/registration.json').read_text());rows=[];index=[]
    for entry in reg['diagnostic_runs']:
        job=entry['job'];run=Path(entry['run_path']);up={int(r['epoch']):r for r in csv.DictReader((run/'updates.csv').open())};parts=np.load(run/'RX_partitions.npy');schedule=np.load(run/'schedule.npy');nC=4 if job['dataset']=='wisig' else 2
        for path in sorted(run.glob('actual_adam_epoch*.pt')):
            digest=hashlib.sha256(path.read_bytes()).hexdigest();s=torch.load(path,map_location='cpu',weights_only=False);e=s['epoch'];assert digest==up[e]['snapshot_sha256'];C=parts[(e-1)*schedule.shape[1],-nC:]
            index.append(dict(run_id=job['run_id'],dataset=job['dataset'],method=job['method'],fold=job['fold'],epoch=e,sha256=digest,private_state_bytes=path.stat().st_size,C_RX=sorted(map(int,C))))
            for risk in s['risk_packet_measurements']:
                if risk['mode']!='eval' or risk['replicate']!=0:continue
                assert set(np.unique(risk['rx']))==set(C)
                for rx in sorted(C):
                    for day in [-999]+sorted(set(risk['date'][risk['rx']==rx])):
                        mask=risk['rx']==rx
                        if day!=-999:mask &= risk['date']==day
                        d=risk['delta'][mask];a=risk['pre'][mask];n=len(d)
                        rows.append(dict(run_id=job['run_id'],dataset=job['dataset'],method=job['method'],fold=job['fold'],epoch=e,phase='early' if e<=job['epochs']/3 else 'middle' if e<=2*job['epochs']/3 else 'late_short',pool=risk['pool'],rx=int(rx),date=int(day),scale=risk['scale'],n=n,delta=float(d.mean()),paired_SE=float(d.std(ddof=1)/np.sqrt(n)),resolution=1e-7+1e-6*float(np.abs(a).mean()),pre_CE=float(a.mean())))
        print('extracted',job['run_id'],flush=True)
    assert len(index)==80
    pub=ROOT/'reports/receiver_c0'
    with (pub/'snapshot_RX_date_risks.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (pub/'snapshot_index.json').write_text(json.dumps(dict(snapshots=index,count=80,no_weights_or_signals_published=True),indent=2)+'\n');print('snapshots',len(index),'RX/date risk rows',len(rows))

if __name__=='__main__':main()
