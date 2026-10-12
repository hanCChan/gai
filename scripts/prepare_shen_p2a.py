#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shen_p2a.cache import build_cache

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    protocol=a.root/'configs/shen_evidence/protocol.candidate.json'
    for receiver in ('n210_1','rtl_6'):
        for role in ('train','validation','same_day_diagnostic'):
            path=build_cache(a.root,protocol,receiver,role)
            print(json.dumps({'cache':str(path),'completed':True}),flush=True)
