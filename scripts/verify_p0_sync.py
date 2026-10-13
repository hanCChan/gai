#!/usr/bin/env python3
"""Independent direct correlation of 100 already selected source packets.

Does not import the detector. Recomputes shape/phase with a dense template
matrix instead of its FFT path; verifies all recorded scopes and seal hashes.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path
import h5py
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(folder):
    reg=json.loads((ROOT/'configs/p0_sync/registration.json').read_text())
    seal=json.loads((ROOT/'configs/p0_sync/seal.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in seal['files'].items())
    packets=list(csv.DictReader((folder/'packets.csv').open()))
    sample=json.loads((ROOT/'configs/p0_sync/sample_manifest.json').read_text())
    keys=lambda r:(r['receiver'],int(r['tx_raw_label']),int(r['row']))
    assert set(map(keys,packets))==set(map(keys,sample)) and len(packets)==2000
    access=json.loads((folder/'signal_access_log.json').read_text())
    assert len(access)==10 and {x['receiver'] for x in access}==set(reg['source_receivers'])
    assert all(Path(x['path']).name==x['receiver']+'_train.h5' and x['mode']=='read_only' for x in access)
    n=1024;j=np.arange(n)
    templates=((j[None,:]-j[:,None])%n+.5)/n-.5
    norm=np.sum(templates[0]**2);errors=[];representatives=[];quiet=[]
    offsets=np.arange(-4,4)
    for rx in reg['source_receivers']:
        item=next(i for i in reg['files'] if i['receiver']==rx)
        path=Path(reg['data_root'])/item['relative']
        assert path.resolve()==path and not path.is_symlink()
        with h5py.File(path,'r') as f:
            for tx in range(31,41):
                p=min((x for x in packets if x['receiver']==rx and int(x['tx_raw_label'])==tx),key=lambda x:int(x['row']))
                raw=np.asarray(f['data'][int(p['row'])]);x=raw[:8192]+1j*raw[8192:]
                freq=np.angle(x[1:]*np.conj(x[:-1]))/(2*np.pi)
                freq=np.concatenate((freq,[2*freq[-1]-freq[-2]])).reshape(8,n)
                profile=np.median(freq,axis=0)
                smooth=profile[(j[:,None]+offsets)%n].mean(axis=1);smooth-=smooth.mean()
                correlations=templates@smooth
                sign=1 if correlations.max()>=-correlations.min() else -1
                phase=int((sign*correlations).argmax())
                r2=float(correlations[phase]**2/(norm*np.sum(smooth**2)))
                coh=float(abs(np.sum(np.conj(x[:-n])*x[n:]))/np.sqrt(np.sum(abs(x[:-n])**2)*np.sum(abs(x[n:])**2)))
                err=dict(receiver=rx,tx_raw_label=tx,row=int(p['row']),phase_difference_samples=phase-int(p['chirp_phase_modulo_samples']),r2_error=abs(r2-float(p['chirp_r2'])),coherence_error=abs(coh-float(p['nominal_coherence'])))
                assert err['phase_difference_samples']==0 and err['r2_error']<1e-12 and err['coherence_error']<1e-12
                errors.append(err)
            for p in (z for z in packets if z['receiver']==rx and z['left_quiet_candidate']=='True'):
                raw=np.asarray(f['data'][int(p['row'])]);x=raw[:8192]+1j*raw[8192:];power=abs(x)**2
                quiet.append(dict(receiver=rx,tx_raw_label=int(p['tx_raw_label']),row=int(p['row']),leading_exact_zero_count=int(np.argmax(power!=0)) if np.any(power!=0) else len(x),first_256_energy_16_sample_blocks=power[:256].reshape(16,16).mean(axis=1).tolist(),prefix_energy_ratio=float(p['prefix_energy_ratio']),interpretation='Low leading energy alone cannot distinguish frame onset from fading/AGC; no pre-window record or SFD reference.'))
    report={'passed':True,'sealed_files_match':True,'sample_manifest_exactly_matches':True,'signal_access_scope_verified':True,'packets_recomputed':100,'all_100_rx_tx_cells_covered':True,'new_unique_packets_outside_2000':0,'phase_difference_max_samples':max(abs(x['phase_difference_samples']) for x in errors),'r2_absolute_error_max':max(x['r2_error'] for x in errors),'coherence_absolute_error_max':max(x['coherence_error'] for x in errors),'quiet_prefix_cases':quiet,'method':'Dense direct ramp correlation; explicit eight-point mean; independent formulas; no detector import','per_cell':errors}
    report['raw_row_reads']=100+len(quiet)
    report['raw_access_log']=[{'receiver':rx,'path':str(Path(reg['data_root'])/(rx+'_train.h5')),'mode':'read_only','stage':'independent_verification','recomputed_cells':10,'quiet_case_reads':sum(z['receiver']==rx for z in quiet)} for rx in reg['source_receivers']]
    (folder/'independent_verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('per_cell','quiet_prefix_cases')}))


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
