#!/usr/bin/env python3
"""Curated public code/result evidence; raw IQ, caches and checkpoints stay local."""
import argparse,hashlib,json,shutil,csv,datetime,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from report_shen_p2a import read_csv,write_csv

def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pipeline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
    out.mkdir(parents=True,exist_ok=False);manifest=[]
    for name in ('preflight_v2.json','iq_cache_verified.json','unit_tests_v4.log','feature_construction_cost.json','P1_all32_preservation_check.json'):
        src=ROOT/'outputs/shen_p2a'/name
        if src.exists():copy(src,out/'preflight'/name)
    for name in ('four_gpu_smoke_v1','four_gpu_smoke_v2','integration_v1'):
        src=ROOT/'outputs/shen_p2a'/name
        for f in src.rglob('*'):
            if f.is_file() and f.suffix in ('.json','.log'):copy(f,out/'preflight'/name/f.relative_to(src))
    old=ROOT/'outputs/shen_p2a/p2a_preregistered_v1'
    for f in old.rglob('*'):
        if f.is_file() and f.suffix in ('.json','.log'):copy(f,out/'failed_attempt_v1'/f.relative_to(old))
    diagnostics=ROOT/'outputs/shen_p2a/diagnostics_v1'
    for name in ('status.json','P1_CIS_rechecked.csv','P1_RTL_optimization_recheck.csv','per_device_feature_quality.csv','per_device_waveform_quality.csv','centroid_shifts.csv','frequency_profiles.csv','ring_descriptive_separability.csv','synthetic_inband_tone_leakage.csv','train_fitted_error_clusters.csv'):
        copy(diagnostics/name,out/'diagnostics'/name)
    for f in (diagnostics/'interpretation').iterdir():copy(f,out/'diagnostics'/'interpretation'/f.name)
    for f in diagnostics.glob('*confusion.csv'):copy(f,out/'diagnostics'/f.name)
    energy=read_csv(diagnostics/'packet_ring_energy_window_probe.csv');group={}
    for r in energy:group.setdefault((r['receiver'],r['role'],r['raw_label'],r['band']),[]).append(r)
    summarized=[]
    for key,rows in sorted(group.items()):
        for k in ('rect_energy_fraction','raw_hann_before_fft_energy_fraction_NONSTRICT'):
            x=np.array([float(r[k]) for r in rows]);summarized.append(dict(receiver=key[0],role=key[1],raw_label=key[2],band=key[3],metric=k,packets=len(x),mean=float(x.mean()),median=float(np.median(x)),p05=float(np.quantile(x,.05)),p95=float(np.quantile(x,.95))))
    write_csv(out/'diagnostics/ring_window_energy_summary.csv',summarized)
    for name in ('pipeline_status.json','selection.frozen.json','confirmation_matrix.json'):
        src=a.pipeline/name
        if src.exists():copy(src,out/name)
    for phase in ('screening','confirmation'):
        src=a.pipeline/phase
        if not src.exists():continue
        for f in src.glob('*.json'):copy(f,out/phase/f.name)
        for run in src.iterdir():
            if not run.is_dir():continue
            status=json.loads((run/'status.json').read_text())
            names=('config.json','metrics.json','status.json','history.csv','confusion_matrix.csv','per_device.csv','shadow_per_device.csv','shadow_confusion_matrix.csv') if status['status']=='completed' else ('config.json','status.json')
            for name in names:
                if (run/name).exists():copy(run/name,out/phase/run.name/name)
    report=a.pipeline/'report'
    if report.exists():
        for f in report.iterdir():
            if f.is_file():copy(f,out/'report'/f.name)
    for f in sorted(out.rglob('*')):
        if f.is_file():manifest.append({'path':str(f.relative_to(out)),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    state=json.loads((a.pipeline/'pipeline_status.json').read_text())
    (out/'publication_manifest.json').write_text(json.dumps({'snapshot_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pipeline_status':state,'files':manifest,'raw_IQ_weights_caches_published':False,'full_packet_predictions_remain_local':True,'final_signal_access':False},indent=2)+'\n')
    print(json.dumps({'files':len(manifest),'bytes':sum(r['bytes'] for r in manifest),'status':state['status']}))
