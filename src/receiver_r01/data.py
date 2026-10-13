"""Only previously audited WiSig development caches and ten Shen Strict CIS caches."""
import json
from pathlib import Path
import numpy as np


def wisig(reg,rxs,days,positions):
    old=reg['wisig'];x=[];y=[];rx=[];day=[];pos=[]
    allowed=old['roles']['source']+old['roles']['internal_rx']
    for ri in rxs:
        assert ri in allowed and ri not in old['roles']['final_rx']
        for di in days:
            assert di in (0,1,2)
            p=Path(reg['wisig_cache'])/f'rx{ri:02d}_day{di}.npz'
            with np.load(p,allow_pickle=False) as a:
                mask=(a['position']>=positions[0])&(a['position']<positions[1])
                n=int(mask.sum());x.append(a['x'][mask]);y.append(a['y'][mask]);pos.append(a['position'][mask])
                rx.append(np.full(n,ri));day.append(np.full(n,di))
    return dict(x=np.concatenate(x),y=np.concatenate(y),rx=np.concatenate(rx),day=np.concatenate(day),position=np.concatenate(pos))


def shen(reg,keys,stats=None,partition='train'):
    assert set(keys)<=set(reg['shen_allowed'])
    root=Path(reg['shen_cache']);parts=[]
    for key in keys:
        p=root/key;info=json.loads((p/'manifest.json').read_text());assert info['source']['role']=='train'
        parts.append(p)
    x=[];y=[];rx=[];pos=[]
    for key,p in zip(keys,parts):
        a=np.load(p/'strict_cis.npy',allow_pickle=False);b=np.load(p/'labels.npy',allow_pickle=False)
        assert a.shape==(2000,1,52,126) and np.array_equal(np.bincount(b,minlength=10),np.full(10,200))
        ids=np.concatenate([np.flatnonzero(b==t)[slice(0,160) if partition=='train' else slice(160,200) if partition=='val' else slice(None)] for t in range(10)])
        x.append(a[ids]);y.append(b[ids])
        pos.append(np.load(p/'rows.npy',allow_pickle=False)[ids]);rx.append(np.full(len(ids),reg['shen_allowed'].index(key)))
    xx=np.concatenate(x)
    if stats is None:
        assert partition=='train'
        # Accumulate only actual fitting packets; validation and outer RX never enter this scaler.
        mean=float(np.mean(xx,dtype=np.float64));std=float(np.std(xx,dtype=np.float64))
        stats=dict(mean=mean,std=std,fit_keys=keys,partition='first160_per_TX')
    return dict(x=((xx-stats['mean'])/stats['std']).astype(np.float32),y=np.concatenate(y),rx=np.concatenate(rx),day=np.full(sum(map(len,y)),-1),position=np.concatenate(pos)),stats


def groups(data):
    cells=np.stack((data['rx'],data['day'],data['y']),axis=1)
    return np.unique(cells,axis=0,return_inverse=True)[1]
