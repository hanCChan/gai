"""Bit-identical float32 IQ cache for I/O; no normalization or extra signals."""
import hashlib,json,os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from shen_stage1.data import ShenDevelopmentPackets
from .cache import write_json

def folder(root,rx,role):return Path(root)/'outputs/shen_p2a/iq_cache'/f'{rx}_{role}'
def build(root,protocol,rx,role):
    ds=ShenDevelopmentPackets(root,protocol,rx,role,cache_handles=True);out=folder(root,rx,role);out.mkdir(parents=True,exist_ok=False)
    a=np.lib.format.open_memmap(out/'iq.npy',mode='w+',dtype=np.float32,shape=(len(ds),2,8192));h=hashlib.sha256()
    try:
        for start in range(0,len(ds),128):
            batch=np.stack([ds[i][0] for i in range(start,min(start+128,len(ds)))]);a[start:start+len(batch)]=batch
            np.testing.assert_array_equal(a[start:start+len(batch)],batch);h.update(batch.tobytes())
        a.flush();np.save(out/'labels.npy',ds.labels)
        item=next(i for i in json.loads(Path(protocol).read_text())['files'] if i['receiver']==rx and i['role']==role)
        write_json(out/'manifest.json',{'status':'completed','packets':len(ds),'shape':list(a.shape),'dtype':'float32','source_relative':item['dataset_relative_path'],'source_sha256':item['sha256'],'protocol_sha256':hashlib.sha256(Path(protocol).read_bytes()).hexdigest(),'float32_payload_sha256':h.hexdigest(),'all_packets_bit_identical_to_development_adapter':True,'normalization_applied':False})
        return out
    except BaseException as exc:write_json(out/'manifest.json',{'status':'failed','error':repr(exc)});raise
    finally:ds.close()
class CachedIQ(Dataset):
    def __init__(self,root,protocol,rx,role):
        checked=ShenDevelopmentPackets(root,protocol,rx,role);self.labels=checked.labels.copy();checked.close();out=folder(root,rx,role)
        m=json.loads((out/'manifest.json').read_text());item=next(i for i in json.loads(Path(protocol).read_text())['files'] if i['receiver']==rx and i['role']==role)
        if m['status']!='completed' or m['source_sha256']!=item['sha256'] or m['protocol_sha256']!=hashlib.sha256(Path(protocol).read_bytes()).hexdigest():raise ValueError('IQ cache changed or incomplete')
        np.testing.assert_array_equal(self.labels,np.load(out/'labels.npy'));self.path=out/'iq.npy';self._a=None
    def __len__(self):return len(self.labels)
    def __getitem__(self,i):
        if self._a is None:self._a=np.load(self.path,mmap_mode='r')
        x=np.array(self._a[i],copy=True)
        if x.shape!=(2,8192) or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('IQ cache invalid')
        return x,int(self.labels[i]),int(i)
    def __getstate__(self):s=self.__dict__.copy();s['_a']=None;return s
