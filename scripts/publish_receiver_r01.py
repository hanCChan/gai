#!/usr/bin/env python3
"""Prepare safe content chunks for GitHub connector Git objects, no credentials."""
import argparse,base64,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARCHIVED=['confusion_counts.csv','per_RX_TX_date.csv','paired_RX_TX_changes.csv','update_diagnostics.csv','update_RX_risks.csv']


def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',action='store_true');p.add_argument('--path');p.add_argument('--offset',type=int,default=0);p.add_argument('--length',type=int,default=30000);a=p.parse_args()
    if a.inventory:
        paths=subprocess.check_output(['git','diff','--cached','--name-only','e64c5a3fe855684b082ff29183001f01b74730be'],cwd=ROOT,text=True).splitlines();items=[]
        for path in paths:
            file=ROOT/path;assert file.is_file() and file.suffix not in ['.pt','.npy','.npz','.pkl','.h5']
            assert not (file.parent==ROOT/'reports/receiver_r01/2026-10-13' and file.name in ARCHIVED)
            content=file.read_bytes();blob=hashlib.sha1(('blob '+str(len(content))+'\0').encode()+content).hexdigest()
            items.append(dict(path=path,mode='100644',type='blob',sha=blob,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
        print(json.dumps(dict(files=items,desired_tree=subprocess.check_output(['git','write-tree'],cwd=ROOT,text=True).strip())))
    else:
        relative=Path(a.path);assert not relative.is_absolute() and '..' not in relative.parts
        assert relative.parts[0] in ['src','scripts','docs','configs','reports'] or str(relative)=='AGENTS.md'
        data=(ROOT/relative).read_bytes()[a.offset:a.offset+a.length]
        print(base64.b64encode(data).decode())

if __name__=='__main__':main()
