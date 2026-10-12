#!/usr/bin/env python3
"""Check only the six authorized sources, pinned environment, cache, and P1 record."""
import argparse, hashlib, json, sys, subprocess
from pathlib import Path
import numpy as np
import scipy, torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shen_stage1.protocol import development_path,DEVELOPMENT_FILES
from shen_stage1.data import ShenDevelopmentPackets
from shen_stage1.representations import channel_independent_spectrogram
from shen_p2a.cache import cache_dir,write_json

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Never overwrite preflight evidence')
    env={'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'python':sys.executable}
    assert (env['torch'],env['numpy'],env['scipy'])==('2.5.1+cu121','1.26.4','1.13.1')
    protocol=ROOT/'configs/shen_evidence/protocol.candidate.json'; info=json.loads(protocol.read_text())
    entries=[i for i in info['files'] if i['signal_access']];assert len(entries)==6
    assert {i['dataset_relative_path'] for i in entries}==DEVELOPMENT_FILES
    records=[]
    for item in entries:
        path=development_path(ROOT/info['data_root'],item['dataset_relative_path'])
        actual=digest(path);assert actual==item['sha256']
        ds=ShenDevelopmentPackets(ROOT,protocol,item['receiver'],item['role'],signal_dtype=np.float64)
        assert len(ds)==item['packets'] and set(ds.labels)==set(range(10))
        folder=cache_dir(ROOT,item['receiver'],item['role']);m=json.loads((folder/'manifest.json').read_text())
        assert m['numpy_version']==np.__version__ and m['scipy_version']==scipy.__version__
        np.testing.assert_array_equal(np.load(folder/'labels.npy'),ds.labels)
        cis=np.load(folder/'cis.npy',mmap_mode='r')
        for label in range(10):
            row=int(np.flatnonzero(ds.labels==label)[0]);x=ds[row][0];expected=channel_independent_spectrogram((x[0]+1j*x[1])[None],dtype=np.float32)[0]
            np.testing.assert_array_equal(cis[row],expected)
        ds.close();records.append({'path':item['dataset_relative_path'],'sha256':actual,'packets':len(ds),'cache_author_cis_exact':True})
    smoke=ROOT/'outputs/shen_p2a/four_gpu_smoke_v2'
    for gpu in range(4):assert json.loads((smoke/f'gpu{gpu}/passed.json').read_text())['passed'] is True
    selection=json.loads((ROOT/'configs/shen_p2a/p1_artifacts.json').read_text())
    p1=[]
    for choice in selection['selection']['choices']:
        runs=[Path(choice['screening_run'])]+[Path(selection['confirmation_root'])/f'{choice["config"]["experiment_id"]}_{choice["receiver"]}_seed{s}' for s in (20261013,20261014)]
        for run in runs:
            assert json.loads((run/'status.json').read_text())['status']=='completed'
            p1.append({'run':str(run),'predictions_sha256':digest(run/'predictions.csv'),'metrics_sha256':digest(run/'metrics.json')})
    write_json(a.output,{'passed':True,'environment':env,'authorized_source_files':records,'P1_selected_runs':p1,'protocol_sha256':digest(protocol),'final_files_opened':0,'gpu_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,memory.total','--format=csv'],text=True)})
    print(json.dumps({'passed':True,'files':len(records),'P1_selected_runs':len(p1)}))
