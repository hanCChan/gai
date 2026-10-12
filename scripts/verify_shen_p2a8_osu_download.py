#!/usr/bin/env python3
"""Reject sparse partial captures and independently check all 50 original hashes."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shen_p2a8.g1_data import sha
from shen_p2a.cache import write_json
if __name__=='__main__':
    folder=ROOT/'outputs/shen_p2a8/osu_download';targets=json.loads((folder/'targets.json').read_text());rows=[]
    for item in targets:
        p=ROOT/item['path']
        if not p.is_file() or p.stat().st_size!=item['bytes'] or Path(str(p)+'.aria2').exists():sys.exit(2)
        actual=sha(p)
        if actual!=item['sha256']:raise ValueError('Official capture differs from prior inventory: '+item['path'])
        rows.append(dict(path=item['path'],bytes=p.stat().st_size,sha256=actual,passed=True))
    write_json(folder/'independent_verification.json',dict(passed=True,captures=50,files=rows));write_json(folder/'status.json',dict(status='completed',captures=50,bytes=sum(r['bytes'] for r in rows),all_sha256_match=True))
