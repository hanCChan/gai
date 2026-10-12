#!/usr/bin/env python3
"""Durable sequential handoff: prepared cache → preflight → source CV → report."""
import argparse,datetime,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shen_p2a.cache import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['shen','osu'],required=True);a=p.parse_args();out=ROOT/f'outputs/shen_p2a8/g1_{a.dataset}_orchestration.json'
    state=lambda status,**kw:write_json(out,dict(status=status,dataset=a.dataset,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
    def run(command):subprocess.run([sys.executable,*command],cwd=ROOT,check=True)
    try:
        if a.dataset=='shen':
            state('waiting_for_cache')
            while not (ROOT/'outputs/shen_p2a8/g1_shen_cache_status.json').exists():time.sleep(10)
            assert json.loads((ROOT/'outputs/shen_p2a8/g1_shen_cache_status.json').read_text())['status']=='completed'
        else:
            from shen_p2a8.g1_data import sha,build_osu
            state('waiting_for_official_capture_download_and_hashes')
            targets=json.loads((ROOT/'outputs/shen_p2a8/osu_download/targets.json').read_text())
            # Presence of aria2 control means a sparse partially downloaded file, even at its final size.
            while not all((ROOT/i['path']).exists() and (ROOT/i['path']).stat().st_size==i['bytes'] and not Path(str(ROOT/i['path'])+'.aria2').exists() for i in targets):time.sleep(20)
            assert all(sha(ROOT/i['path'])==i['sha256'] for i in targets)
            write_json(ROOT/'outputs/shen_p2a8/osu_download/status.json',dict(status='completed',captures=50,bytes=8000000000,all_sha256_match=True))
            state('building_periodic_window_cache');reg=json.loads((ROOT/'configs/shen_p2a8/g1_registration.json').read_text())
            for rx in ('RX1','RX2'):build_osu(ROOT,rx,reg)
        state('preflight_running');run(['scripts/preflight_shen_p2a8_g1.py','--dataset',a.dataset])
        state('baselines_running');run(['scripts/run_shen_p2a8_g1.py','--dataset',a.dataset,'--output',f'outputs/shen_p2a8/g1_{a.dataset}_v1'])
        state('completed',G2_G3_started=False,old_P2_B_P3_started=False)
    except BaseException as e:state('failed',error=repr(e));raise
