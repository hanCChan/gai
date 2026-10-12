#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shen_p2a.iq_cache import build
if __name__=='__main__':
    for rx in ('n210_1','rtl_6'):
        for role in ('train','validation'):print(build(ROOT,ROOT/'configs/shen_evidence/protocol.candidate.json',rx,role),flush=True)
