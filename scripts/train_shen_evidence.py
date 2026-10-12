#!/usr/bin/env python3
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from shen_evidence.runner import train

if __name__=='__main__':
    p=argparse.ArgumentParser(description='Day1 train / Day2 development only')
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=ROOT)
    a=p.parse_args()
    print(json.dumps(train(json.loads(a.config.read_text()),a.root,a.output),indent=2))
