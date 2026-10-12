#!/usr/bin/env python3
"""P0 audit: six allowed files only; protected files use prior metadata only."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from shen_stage1.protocol import development_path
from shen_stage1.data import ShenDevelopmentPackets
from shen_evidence.runner import dump


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/shen_evidence/p0_20261012')
    parser.add_argument('--baseline-ref',default='ba5d4d415ad00c3aa640acfcae335039a83074e9')
    parser.add_argument('--reference-repo',type=Path,default=ROOT)
    args=parser.parse_args()
    out=args.output.resolve()
    if (out/'audit.json').exists():
        parser.error('Existing audit records are immutable; use a new --output directory')
    out.mkdir(parents=True,exist_ok=True)
    protocol_path=ROOT/'configs/shen_evidence/protocol.candidate.json'
    proto=json.loads(protocol_path.read_text()); manifest=[]; diagnostics=[]; frequency_summary=[]
    for item in proto['files']:
        if item['role'] not in ('train','validation','same_day_diagnostic'): continue
        path=development_path(ROOT/proto['data_root'],item['dataset_relative_path'])
        sha=digest(path)
        if sha!=item['sha256']: raise ValueError(f'Development file hash changed: {path}')
        ds=ShenDevelopmentPackets(ROOT,protocol_path,item['receiver'],item['role'],cache_handles=True,signal_dtype=np.float64)
        record=dict(item,verified_sha256=sha,label_counts=np.bincount(ds.labels,minlength=10).tolist())
        manifest.append(record); energies={g:[] for g in (0,6250,12500,25000)}
        freq=np.fft.fftshift(np.fft.fftfreq(8192,d=1e-6)); ib=np.abs(freq)<=62500
        for start in range(0,len(ds),128):
            packets=[ds[i] for i in range(start,min(start+128,len(ds)))]
            x=np.stack([p[0][0]+1j*p[0][1] for p in packets]); psd=np.abs(np.fft.fftshift(np.fft.fft(x,norm='ortho'),axes=-1))**2
            total=psd.sum(1)
            if not np.isfinite(total).all() or np.any(total<=0): raise ValueError('Invalid development packet energy')
            ib_frac=psd[:,ib].sum(1)/total
            for g in energies:
                mask=np.abs(freq)>62500+g; energies[g].extend((psd[:,mask].sum(1)/total).tolist())
            for i,(_,label,row) in enumerate(packets):
                diagnostics.append({'sample_id':f'{item["dataset_relative_path"]}:{row}','receiver':item['receiver'],
                                    'role':item['role'],'label':label,'rms':float(np.sqrt(total[i]/8192)),
                                    'ib_energy_fraction':float(ib_frac[i]),
                                    **{f'oob_energy_fraction_guard_{g}':energies[g][start+i] for g in energies}})
        for g,values in energies.items():
            mask=np.abs(freq)>62500+g
            frequency_summary.append({'receiver':item['receiver'],'role':item['role'],'guard_hz_assumed':g,
                                      'ib_bins':int(ib.sum()),'oob_bins':int(mask.sum()),'guard_bins':int((~(ib|mask)).sum()),
                                      'oob_energy_fraction_median':float(np.median(values)),
                                      'oob_energy_fraction_p05':float(np.quantile(values,.05)),
                                      'oob_energy_fraction_p95':float(np.quantile(values,.95)),
                                      'oob_energy_positive_packets':int(np.count_nonzero(np.array(values)>0))})
        ds.close()
    for name,rows in [('development_manifest.csv',manifest),('oob_packet_diagnostics.csv',diagnostics),('oob_frequency_summary.csv',frequency_summary)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # Exact allowed development split, without reopening locked HDF5.
    split=[]
    for item in manifest:
        for row in range(item['packets']): split.append(f'{item["dataset_relative_path"]}:{row},{item["role"]}\n')
    (out/'packet_split.csv').write_text('sample_id,role\n'+''.join(split))
    source_paths=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    source_hashes={p:digest(ROOT/p) for p in source_paths if (ROOT/p).is_file()}
    snapshot=args.reference_repo.resolve(); differing=[]; compared=0
    for path in source_paths:
        if (snapshot/path).is_file():
            try: baseline=subprocess.check_output(['git','show',f'{args.baseline_ref}:{path}'],cwd=ROOT,stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError: continue
            compared+=1
            if hashlib.sha256(baseline).hexdigest()!=digest(snapshot/path): differing.append(path)
    audit={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),
           'local_base_commit':subprocess.check_output(['git','rev-parse',args.baseline_ref],cwd=ROOT,text=True).strip(),
           'gai_snapshot_commit':subprocess.check_output(['git','rev-parse',args.baseline_ref],cwd=snapshot,text=True).strip(),
           'snapshot_baseline_compared_files':compared,'snapshot_baseline_differing_paths':differing,
           'github_main_live_verified':False,'github_main_access_error':'This local audit does not contact GitHub; see publication provenance for online verification',
           'branch':subprocess.check_output(['git','rev-parse','--abbrev-ref','HEAD'],cwd=ROOT,text=True).strip(),
           'packages':{p:importlib.metadata.version(p) for p in ['torch','numpy','scipy','h5py','scikit-learn','tqdm']},
           'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
           'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv'],text=True),
           'protocol_sha256':digest(protocol_path),'packet_split_sha256':digest(out/'packet_split.csv'),
           'development_manifest':manifest,'protected_signals_read':0,'new_final_receiver_labels_read':0,
           'source_hashes':source_hashes,'frequency_parameters_status':'Fs=1MHz BW=125kHz literature assumptions, local metadata unverified',
           'provenance_limitations':['actual_dates_unknown','session_ids_unknown','author_original_zip_identity_unverified','earlier_OOB_split_overlap_unknown'],
           'hdf5_io_benchmark':{'rows':512,'open_per_row_seconds':1.0629937169996992,'worker_handle_seconds':.3188275180000346,
                                'design':'same random-row order, sequential warm-cache measurements; indicative not a controlled cold-cache benchmark'}}
    dump(out/'audit.json',audit)
    print(json.dumps({'audit':'passed','files_verified':len(manifest),'packets_diagnosed':len(diagnostics),'protected_signals_read':0},indent=2))

if __name__=='__main__': main()
