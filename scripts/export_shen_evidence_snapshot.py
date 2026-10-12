#!/usr/bin/env python3
"""Freeze completed development results for publication; never reads IQ/checkpoints."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--staging',type=Path,required=True)
    args=parser.parse_args()
    project=args.project.resolve(); output=args.output.resolve(); staging=args.staging.resolve()
    output.mkdir(parents=True,exist_ok=True); staging.mkdir(parents=True,exist_ok=False)
    if (output/'publication_manifest.json').exists(): parser.error('Existing snapshot is immutable; use a new output')
    evidence=project/'outputs/shen_evidence'; final=evidence/'p1_final_balanced'
    selection=final/'selection.json'
    sources=[evidence/'p1_seed20261012',evidence/'p1_lr0003',final/'confirmation_runs']
    snapshot=[]; completed=[]
    for group,folder in enumerate(sources):
        for run in sorted(folder.iterdir()):
            if not run.is_dir() or not (run/'status.json').exists(): continue
            try: state=json.loads((run/'status.json').read_text())
            except json.JSONDecodeError: state={'status':'updating'}
            snapshot.append({'run':run.name,'phase':'screening' if group<2 else 'confirmation','status':state['status'],
                             'epoch':state.get('epoch'),'best_macro_f1':state.get('best_macro_f1')})
            if state['status']!='completed': continue
            local=staging/'completed_runs'/run.name; local.mkdir(parents=True)
            for filename in ['config.json','metrics.json','status.json','predictions.csv','errors.csv','history.csv','confusion_matrix.csv']:
                shutil.copy2(run/filename,local/filename)
            public=output/'runs'/run.name; public.mkdir(parents=True)
            for filename in ['config.json','metrics.json','status.json','history.csv','confusion_matrix.csv']:
                shutil.copy2(local/filename,public/filename)
            completed.append(run.name)
    subprocess.run([sys.executable,str(ROOT/'scripts/report_shen_evidence.py'),'--runs',str(staging/'completed_runs'),
                    '--selection',str(selection),'--output',str(staging/'summary')],check=True)
    for path in (staging/'summary').iterdir(): shutil.copy2(path,output/path.name)
    shutil.copy2(selection,output/'selection.frozen.json')
    shutil.copy2(final/'pipeline_status.json',output/'pipeline_status_at_snapshot.json')
    shutil.copy2(final/'decision.json',output/'confirmation_decision.json')
    p0=output/'p0'; p0.mkdir()
    original=evidence/'p0_20261012'
    for name in ['audit.json','development_manifest.csv','oob_frequency_summary.csv','protocol_regression_tests.log',
                 'pipeline_balanced_tests.log','regression_tests.log']:
        shutil.copy2(original/name,p0/name)
    shutil.copytree(original/'cis_equivalence',p0/'cis_equivalence')
    for folder in ['four_gpu_smoke','four_gpu_smoke_v2']:
        target=p0/folder; target.mkdir()
        shutil.copy2(original/folder/'launcher_results.json',target/'launcher_results.json')
        for logfile in (original/folder).glob('*.log'): shutil.copy2(logfile,target/logfile.name)
        for gpu in range(4):
            for name in ['checks.json','passed.json']:
                path=original/folder/f'gpu{gpu}'/name
                if path.exists():
                    (target/f'gpu{gpu}').mkdir(exist_ok=True); shutil.copy2(path,target/f'gpu{gpu}'/name)
    for csv_file in output.rglob('*.csv'):
        csv_file.write_bytes(csv_file.read_bytes().replace(b'\r\n',b'\n'))
    meta={'csv_line_endings':'LF, numerical contents unchanged','snapshot_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'timezone':'Asia/Shanghai',
          'local_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),
          'github_main_verified_at_publication':'ba5d4d415ad00c3aa640acfcae335039a83074e9',
          'screening_budget':16,'confirmation_budget':16,'completed_runs':completed,'observed_statuses':snapshot,
          'final_test_signals_read':0,'P2_P3_started':False,'original_iq_uploaded':False,'checkpoints_uploaded':False,
          'full_packet_predictions_uploaded':False,'packet_predictions_retained_at':str(staging/'completed_runs'),
          'portability_adjustments':['frozen protocol bundled in configs, unchanged content/SHA',
                                     'confirmation configs match chosen learning rates',
                                     'auditor baseline/ref made explicit for clone use']}
    meta['published_file_sha256']={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'publication_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'snapshot':str(output),'completed':len(completed),'screening_completed':sum(s['status']=='completed' and s['phase']=='screening' for s in snapshot),
                      'confirmation_completed':sum(s['status']=='completed' and s['phase']=='confirmation' for s in snapshot)},indent=2))

if __name__=='__main__': main()
