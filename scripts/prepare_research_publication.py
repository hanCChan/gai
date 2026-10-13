#!/usr/bin/env python3
"""Prepare exact Git blobs for publishing through the connected GitHub app.

No credential handling and no remote writes. The orchestration workflow uses
GitHub create_blob/create_tree/create_commit/create_branch or update_ref, then
fetches the result to prove file-tree equality. Large blobs are sent in chunks
to the orchestrator; no shell interpolation of file contents is needed.
"""
import argparse
import base64
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def git(*args):return subprocess.check_output(['git','-C',str(ROOT),*args])


def main():
    p=argparse.ArgumentParser();p.add_argument('--base');p.add_argument('--output',type=Path)
    p.add_argument('--blob');p.add_argument('--offset',type=int,default=0);p.add_argument('--chunk',type=int,default=16000)
    a=p.parse_args()
    if a.blob:
        if not re.fullmatch('[0-9a-f]{40}',a.blob) or a.offset<0 or not 1<=a.chunk<=16000:raise ValueError('Invalid blob/chunk')
        data=git('cat-file','blob',a.blob)
        print(base64.b64encode(data).decode()[a.offset:a.offset+a.chunk]);return
    if not a.base or not a.output:raise ValueError('Provide --base and --output')
    if git('status','--porcelain').strip():raise ValueError('Commit completed research before preparing publication')
    base=git('rev-parse',a.base).decode().strip();head=git('rev-parse','HEAD').decode().strip()
    entries=[]
    for path in git('diff','--name-only',base,head).decode().splitlines():
        if not path:continue
        if any(part in {'data','jugaoyijian','.venv','credentials'} for part in Path(path).parts):raise ValueError('Private/raw path: '+path)
        if Path(path).suffix.lower() in {'.h5','.hdf5','.dat','.pt','.pth','.ckpt','.env','.pem','.key'}:raise ValueError('Private/raw artifact: '+path)
        if Path(path).suffix=='.pdf' and not (path.startswith('reports/') and '/figures/' in path):raise ValueError('Unreviewed PDF: '+path)
        if git('diff','--diff-filter=D','--name-only',base,head,'--',path).strip():raise ValueError('Deletion needs separate review: '+path)
        obj=git('rev-parse',head+':'+path).decode().strip();data=git('cat-file','blob',obj)
        line=git('ls-tree',head,'--',path).decode().split();mode=line[0]
        entries.append(dict(path=path,mode=mode,type='blob',sha=obj,size=len(data),sha256=hashlib.sha256(data).hexdigest()))
    value=dict(repository='hanCChan/gai',local_root=str(ROOT),local_head=head,remote_parent=base,
               base_tree=git('rev-parse',base+'^{tree}').decode().strip(),
               expected_tree=git('rev-parse',head+'^{tree}').decode().strip(),entries=entries,
               verification_required='All blob SHA match; new remote tree equals expected_tree; read back decision/report')
    if a.output.exists():raise ValueError('Do not overwrite publication manifest')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k!='entries'}));print('entries',len(entries))


if __name__=='__main__':main()
