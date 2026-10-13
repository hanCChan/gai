#!/usr/bin/env python3
"""Whitelisted connector blob inventory and content chunks."""
import argparse,base64,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE='aaa45ea16f16672a11b695a97607d02b24b238d3'


def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',action='store_true');p.add_argument('--path');p.add_argument('--offset',type=int,default=0);p.add_argument('--length',type=int,default=30000);a=p.parse_args()
    if a.inventory:
        files=[]
        for name in subprocess.check_output(['git','diff','--cached','--name-only',BASE],cwd=ROOT,text=True).splitlines():
            path=ROOT/name;assert name in ['AGENTS.md','.gitignore'] or '/receiver_c0/' in name or 'receiver_c0' in path.name
            assert path.suffix not in ['.pt','.pth','.npy','.npz','.pkl','.h5','.hdf5','.key','.pem'];b=path.read_bytes();files.append(dict(path=name,mode='100644',type='blob',sha=hashlib.sha1(('blob '+str(len(b))+'\0').encode()+b).hexdigest(),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
        print(json.dumps(dict(files=files,desired_tree=subprocess.check_output(['git','write-tree'],cwd=ROOT,text=True).strip())))
    else:
        path=Path(a.path);assert not path.is_absolute() and '..' not in path.parts;assert path.parts[0] in ['src','scripts','configs','docs','reports'] or str(path) in ['AGENTS.md','.gitignore'];print(base64.b64encode((ROOT/path).read_bytes()[a.offset:a.offset+a.length]).decode())

if __name__=='__main__':main()
