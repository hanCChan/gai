#!/usr/bin/env python3
"""Execute source-only CV, freeze epochs, then three-seed four-baseline G1."""
import argparse,concurrent.futures,datetime,json,math,os,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from shen_p2a.cache import write_json
from shen_p2a8.g1_data import sha

def launch(jobs,out):
    matrix=[]
    for i,cfg in enumerate(jobs):
        path=out/'configs'/(cfg['run_id']+'.json');write_json(path,cfg);matrix.append(dict(run_id=cfg['run_id'],gpu=i%4,config=str(path)))
    write_json(out/('execution_matrix_'+jobs[0]['phase']+'.json'),matrix)
    def queue(gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        for job in matrix:
            if job['gpu']!=gpu:continue
            cmd=[sys.executable,'scripts/train_shen_p2a8_g1.py','--config',job['config'],'--output',str(out/'runs'/job['run_id'])]
            with (out/'logs'/(job['run_id']+'.log')).open('w') as log:
                subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(job['run_id'],'completed',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(queue,range(4)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['shen','osu'],required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    for name in ('configs','runs','logs'):(out/name).mkdir()
    regpath=ROOT/'configs/shen_p2a8/g1_registration.json';reg=json.loads(regpath.read_text());digest=sha(regpath)
    state=lambda status,**kw:write_json(out/'pipeline_status.json',dict(status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),dataset=a.dataset,**kw))
    try:
        gate=ROOT/f'outputs/shen_p2a8/g1_{a.dataset}_preflight.json';assert json.loads(gate.read_text())['passed'] is True
        state('source_cv_running' if a.dataset=='shen' else 'fixed_epoch_OSU_running')
        common=dict(registration_sha256=digest,batch_size=64,workers=2,classes=10 if a.dataset=='shen' else 25)
        jobs=[]
        if a.dataset=='shen':
            for fold,heldout in enumerate(reg['source_receiver_folds']):
                training=[r for r in reg['train_receivers'] if r not in heldout]
                for method in reg['methods']:
                    jobs.append(dict(common,run_id=f'CV{fold}_{method}',phase='cv',method=method,seed=reg['cv_seed'],epochs=reg['max_cv_epochs'],train_keys=['shen_'+r+'_train' for r in training],validation_keys=['shen_'+r+'_train' for r in heldout]))
            launch(jobs,out);choices={}
            for method in reg['methods']:
                epochs=[json.loads((out/'runs'/f'CV{i}_{method}'/'status.json').read_text())['best_source_epoch'] for i in range(2)]
                choices[method]=dict(fold_best_epochs=epochs,final_epoch=int(math.ceil(np.mean(epochs))))
            write_json(out/'source_epoch_selection.frozen.json',dict(choices=choices,development_used=False,frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            state('confirmation_running',choices=choices);jobs=[]
            evaluation=[x['key'] for x in reg['files'] if x['role']!='train']
            for seed in reg['confirmation_seeds']:
                for method in reg['methods']:
                    jobs.append(dict(common,run_id=f'G1_shen_{method}_seed{seed}',phase='confirmation',method=method,seed=seed,epochs=choices[method]['final_epoch'],train_keys=['shen_'+r+'_train' for r in reg['train_receivers']],evaluation_keys=evaluation))
        else:
            for direction in ('RX1','RX2'):
                target='RX2' if direction=='RX1' else 'RX1'
                for seed in reg['confirmation_seeds']:
                    for method in reg['methods']:
                        jobs.append(dict(common,run_id=f'G1_osu_{direction}_{method}_seed{seed}',phase='osu',method=method,seed=seed,epochs=reg['osu_epochs'],train_keys=['osu_'+direction],evaluation_keys=['osu_'+direction,'osu_'+target]))
        launch(jobs,out);state('reporting')
        subprocess.run([sys.executable,'scripts/report_shen_p2a8_g1.py','--pipeline',str(out),'--dataset',a.dataset],cwd=ROOT,check=True)
        state('completed',formal_runs=20 if a.dataset=='shen' else 24,locked_accessed=False,G2_G3_started=False)
    except BaseException as e:state('failed',error=repr(e));raise
