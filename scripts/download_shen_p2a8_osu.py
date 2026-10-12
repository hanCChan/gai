#!/usr/bin/env python3
"""Fetch just 50 official OSU cross-RX captures; verify prior inventory SHA256."""
import concurrent.futures,json,hashlib,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from shen_p2a.cache import write_json
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def download(item):
    path=ROOT/item['path'];path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.stat().st_size==item['bytes'] and sha(path)==item['sha256']:return dict(item,status='existing_verified')
    if path.exists():raise ValueError('Existing capture differs; refusing to overwrite')
    partial=path.with_suffix('.dat.partial')
    url='https://research.engr.oregonstate.edu/hamdaoui/RFFP-dataset/LoRa-Dataset/'+item['path'].split('data/raw/osu_lora/',1)[1]
    subprocess.run(['curl','-fL','--retry','3','--connect-timeout','20','--max-time','300','-C','-','-o',str(partial),url],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True)
    if partial.stat().st_size!=item['bytes'] or sha(partial)!=item['sha256']:raise ValueError('Capture hash/size mismatch: '+str(path))
    partial.replace(path);print(path.name,path.parent.name,'verified',flush=True)
    return dict(item,status='downloaded_verified',url=url)
if __name__=='__main__':
    out=ROOT/'outputs/shen_p2a8/osu_download';out.mkdir(parents=True,exist_ok=True)
    items=json.loads(Path('/home/keyan/code-y/inventory/bundle_checksums.json').read_text())
    items=[r for r in items if 'Diff_Receivers_Setup_Indoor_SameTx' in r['path'] and r['path'].endswith('.dat')]
    assert len(items)==50 and sum(i['bytes'] for i in items)==8000000000
    write_json(out/'targets.json',items);results=[]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            for r in pool.map(download,items):results.append(r);write_json(out/'verified.json',results)
        write_json(out/'status.json',dict(status='completed',captures=50,bytes=8000000000,all_sha256_match=True))
    except BaseException as e:write_json(out/'status.json',dict(status='failed',error=repr(e),verified=len(results)));raise
