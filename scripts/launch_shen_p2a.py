#!/usr/bin/env python3
"""Four designated P2-A queues; all failure attempts are retained."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','train'],required=True)
    p.add_argument('--matrix',type=Path,default=ROOT/'configs/shen_p2a/screening_matrix.json')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke-dir',type=Path)
    a=p.parse_args();jobs=json.loads(a.matrix.read_text())
    if not isinstance(jobs,list) or any(j.get('stage')!='P2-A' for j in jobs):p.error('Only P2-A is authorized here')
    if a.stage=='train':
        preflight=ROOT/'outputs/shen_p2a/preflight_v2.json'
        if not preflight.exists() or json.loads(preflight.read_text()).get('passed') is not True:p.error('Preflight gate is required')
        if a.smoke_dir is None:p.error('Training requires all four smoke gates')
        for gpu in range(4):
            info=a.smoke_dir/f'gpu{gpu}'/'passed.json'
            if not info.exists() or json.loads(info.read_text()).get('passed') is not True:p.error(f'GPU{gpu} has not passed')
    a.output.mkdir(parents=True,exist_ok=False)
    def queue(gpu):
        group=[j for j in jobs if j['gpu']==gpu];results=[]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        if a.stage=='smoke':
            # One actual-data overfit/checkpoint check for every unique transform/architecture.
            signatures=set();configs=[]
            for j in group:
                cfg=json.loads((ROOT/j['config']).read_text())
                keys=('receiver','family','representation','map_normalization','model','band','coordinate','normalization','window','guard_hz','fusion','aux_band')
                signature=tuple(cfg.get(k) for k in keys)
                if signature not in signatures:configs.append(str((ROOT/j['config']).resolve()));signatures.add(signature)
            config_list=a.output/f'gpu{gpu}_configs.json';config_list.write_text(json.dumps(configs,indent=2)+'\n')
            command=[sys.executable,str(ROOT/'scripts/smoke_shen_p2a.py'),'--configs',str(config_list.resolve()),'--output',str((a.output/f'gpu{gpu}').resolve())]
            with (a.output/f'gpu{gpu}.log').open('w') as log:code=subprocess.call(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            return [{'gpu':gpu,'returncode':code,'stage':'smoke','configs':len(configs)}]
        for j in group:
            command=[sys.executable,str(ROOT/'scripts/train_shen_p2a.py'),'--config',str((ROOT/j['config']).resolve()),'--output',str((a.output/j['run_id']).resolve())]
            with (a.output/f'{j["run_id"]}.log').open('w') as log:code=subprocess.call(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            results.append({'gpu':gpu,'run_id':j['run_id'],'returncode':code,'command':command})
            if code:break
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:results=list(executor.map(queue,range(4)))
    (a.output/'launcher_results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2),flush=True)
    if any(j['returncode'] for group in results for j in group):sys.exit(1)
