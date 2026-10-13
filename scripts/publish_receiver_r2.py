#!/usr/bin/env python3
"""Safe GitHub connector inventory and base64 chunks for this isolated stage."""
import argparse,base64,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE='84a360bbae2787bb78c00a3590cdb9be2151434f'


def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',action='store_true');p.add_argument('--path');p.add_argument('--offset',type=int,default=0);p.add_argument('--length',type=int,default=60000);a=p.parse_args()
    if a.inventory:
        paths=subprocess.check_output(['git','diff','--cached','--name-only',BASE],cwd=ROOT,text=True).splitlines();items=[]
        for name in paths:
            file=ROOT/name;assert file.is_file() and file.suffix not in ['.pt','.pth','.ckpt','.npy','.npz','.h5','.hdf5','.dat','.pem','.key','.pickle','.pkl']
            assert name in ['AGENTS.md','.gitignore'] or '/receiver_r2/' in name or 'receiver_r2' in file.name,name
            data=file.read_bytes();items.append(dict(path=name,mode='100644',type='blob',sha=hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest(),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
        print(json.dumps(dict(files=items,desired_tree=subprocess.check_output(['git','write-tree'],cwd=ROOT,text=True).strip())))
    else:
        name=Path(a.path);assert not name.is_absolute() and '..' not in name.parts
        assert name.parts[0] in ['src','scripts','docs','configs','reports'] or str(name) in ['AGENTS.md','.gitignore']
        print(base64.b64encode((ROOT/name).read_bytes()[a.offset:a.offset+a.length]).decode())

if __name__=='__main__':main()
