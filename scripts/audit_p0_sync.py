#!/usr/bin/env python3
"""Run sealed source-only P0-A1. No torch import, weights, optimizer or GPU."""
import argparse
import csv
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from p0_sync.detector import analyze, distance, synthetic_chirps


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''): h.update(chunk)
    return h.hexdigest()


def js(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_csv(path, rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def checked_raw_path(reg, item):
    rx=item['receiver']
    if rx not in reg['source_receivers'] or item['role']!='train' or item['relative']!=rx+'_train.h5':
        raise ValueError('Outside ten source train files')
    base=Path(reg['data_root']).resolve()
    requested=base/item['relative']
    if requested.is_symlink() or requested.resolve()!=requested:
        raise ValueError('Raw file alias/escape prohibited')
    return requested


def synthetic(out,reg):
    cfg=reg['detector'];s=reg['synthetic_validation'];rng=np.random.default_rng(s['seed'])
    trials=[];summaries=[]
    for snr in s['snr_db']:
        for cfo in s['cfo_cycles_per_sample']:
            rows=[]
            for i in range(s['repetitions_per_cell']):
                shift=int(rng.integers(s['shift_samples_inclusive'][0],s['shift_samples_inclusive'][1]+1))
                x=synthetic_chirps(shift,cfo,snr,rng)
                m,*_=analyze(x,cfg)
                row=dict(snr_db=snr,cfo_cycles_per_sample=cfo,trial=i,true_phase_samples=shift%1024,
                         estimated_phase_samples=m['chirp_phase_modulo_samples'],
                         error_samples=float(distance(m['chirp_phase_modulo_samples'],shift,1024)),
                         reliable=m['reliable_symbol_phase'],r2=m['chirp_r2'])
                trials.append(row);rows.append(row)
            q95=float(np.quantile([r['error_samples'] for r in rows],.95))
            coverage=float(np.mean([r['reliable'] for r in rows]))
            summaries.append(dict(snr_db=snr,cfo_cycles_per_sample=cfo,n=len(rows),q95_error_samples=q95,
                                  reliable_coverage=coverage,passed=q95<=s['maximum_q95_error_samples'] and coverage>=s['minimum_reliable_coverage_each_cell']))
    negative=[]
    for kind in s['negative_control_types']:
        count=0
        for i in range(s['negative_controls_per_type']):
            if kind=='white_noise':x=rng.normal(size=8192)+1j*rng.normal(size=8192)
            elif kind=='tone':
                freq=float(rng.uniform(-.2,.2));x=np.exp(2j*np.pi*freq*np.arange(8192))+.01*(rng.normal(size=8192)+1j*rng.normal(size=8192))
            elif kind=='downchirp':x=synthetic_chirps(int(rng.integers(-511,512)),0,30,rng,direction=-1)
            else:raise ValueError(kind)
            m,*_=analyze(x,cfg);count+=m['reliable_symbol_phase']
        fraction=count/s['negative_controls_per_type']
        negative.append(dict(kind=kind,n=s['negative_controls_per_type'],false_acceptances=int(count),fraction=fraction,
                             passed=fraction<=s['maximum_false_acceptance_fraction_each_type']))
    write_csv(out/'synthetic_trials.csv',trials)
    report=dict(cells=summaries,negative_controls=negative,
                passed=all(z['passed'] for z in summaries+negative),
                scope=s['scope'],real_packets_read_before_validation=0)
    js(out/'synthetic_validation.json',report)
    print('synthetic',json.dumps(report),flush=True)
    return report


def run(out):
    if out.exists():raise ValueError('Use a new output directory')
    reg_path=ROOT/'configs/p0_sync/registration.json'
    reg=json.loads(reg_path.read_text());seal=json.loads((ROOT/'configs/p0_sync/seal.json').read_text())
    for rel,expected in seal['files'].items():
        if sha(ROOT/rel)!=expected:raise ValueError('Frozen file changed: '+rel)
    selected=json.loads((ROOT/'configs/p0_sync/sample_manifest.json').read_text())
    if len(selected)!=reg['initial_packets']:raise ValueError('Sample count changed')
    out.mkdir(parents=True);js(out/'registration.json',reg);js(out/'seal.json',seal)
    start=time.monotonic();access=[]
    synth=synthetic(out,reg)
    if not synth['passed']:
        js(out/'status.json',dict(stage='P0-A1',status='STOP_NO_OBSERVABLE_SYNC',reason='Frozen detector synthetic validation failed',raw_packets_read=0))
        return
    metrics=[];source_info=[];profiles=[];curves=[];blocks=[]
    for item in reg['files']:
        path=checked_raw_path(reg,item);rx=item['receiver']
        access.append(dict(receiver=rx,path=str(path),role='source_train_raw',mode='read_only',operation='provenance_hash_and_200_selected_rows',utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        js(out/'signal_access_log.json',access)
        before=path.stat();actual=sha(path)
        if actual!=item['expected_sha256']:raise ValueError('Raw SHA mismatch '+rx)
        cache_info=next(i for i in reg['historical_index_files'] if i['receiver']==rx)
        for kind in ('rows','labels','manifest'):
            if sha(cache_info[kind+'_path'])!=cache_info[kind+'_sha256']:raise ValueError('Historical index changed')
        hist_rows=np.load(cache_info['rows_path']);hist_labels=np.load(cache_info['labels_path'])
        entries=[s for s in selected if s['receiver']==rx];entries=sorted(entries,key=lambda e:e['row'])
        if len(entries)!=200 or len({e['row'] for e in entries})!=200:raise ValueError('Expected 200 distinct rows per RX')
        for e in entries:
            if hist_rows[e['historical_cache_index']]!=e['row'] or hist_labels[e['historical_cache_index']]+31!=e['tx_raw_label']:raise ValueError('Not a subset of frozen G1 indices')
        ids=np.array([e['row'] for e in entries])
        with h5py.File(path,'r') as f:
            for key in ('data','label','CFO','SNR'):
                if not isinstance(f.get(key,getlink=True),h5py.HardLink) or f[key].is_virtual or f[key].external:raise ValueError('External/VDS inputs prohibited')
            if f['data'].shape!=(8000,16384):raise ValueError('Unexpected data geometry')
            labels=np.asarray(f['label'][:]).reshape(-1)
            if not np.array_equal(labels[ids],[e['tx_raw_label'] for e in entries]):raise ValueError('Raw label mismatch')
            stored=np.asarray(f['data'][ids],dtype='float64')
            x=stored[:,:8192]+1j*stored[:,8192:]
            snr=np.asarray(f['SNR'][:]).reshape(-1)[ids];cfo=np.asarray(f['CFO'][:]).reshape(-1)[ids]
            source_info.append(dict(receiver=rx,path=str(path),sha256=actual,bytes=before.st_size,
                                    attrs={k:str(v) for k,v in f.attrs.items()},
                                    fields={k:dict(shape=list(f[k].shape),dtype=str(f[k].dtype),attrs={a:str(v) for a,v in f[k].attrs.items()}) for k in ('data','label','CFO','SNR')},
                                    selected_rows=ids.tolist(),fold=entries[0]['fold'],packets_analyzed=200,
                                    mtime_ns_before=before.st_mtime_ns))
            for i,e in enumerate(entries):
                m,profile,curve,fit=analyze(x[i],reg['detector'])
                metrics.append(dict(**e,snr_stored=float(snr[i]),cfo_stored=float(cfo[i]),**m))
                for j,b in enumerate(fit):blocks.append(dict(receiver=rx,tx_raw_label=e['tx_raw_label'],row=e['row'],block=j,**b))
                # One predeclared representative per TX/RX: first selected row.
                if not any(z['receiver']==rx and z['tx_raw_label']==e['tx_raw_label'] for z in profiles):
                    profiles.append(dict(receiver=rx,tx_raw_label=e['tx_raw_label'],row=e['row'],profile=profile.tolist()))
                    curves.append(dict(receiver=rx,tx_raw_label=e['tx_raw_label'],row=e['row'],r2=curve.tolist()))
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Input changed during audit')
        print(rx,'200 packets complete; reliable phase',sum(m['reliable_symbol_phase'] for m in metrics if m['receiver']==rx),flush=True)
    write_csv(out/'packets.csv',metrics);write_csv(out/'chirp_blocks.csv',blocks)
    js(out/'source_manifest.json',source_info);js(out/'representative_profiles.json',profiles);js(out/'representative_candidate_curves.json',curves)
    js(out/'status.json',dict(stage='P0-A1',status='completed_pending_evidence_decision',raw_packets_analyzed=len(metrics),synthetic_positive_trials=360,synthetic_negative_trials=300,
                             new_training_runs=0,model_loads=0,development_signal_reads=0,locked_signal_reads=0,day3_day4_reads=0,osu_signal_reads=0,elapsed_seconds=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output.resolve())
