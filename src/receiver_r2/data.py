"""Predeclared source-only diagnostic fitting, C1, C2 and validation pools."""
import hashlib
from pathlib import Path
import numpy as np
from receiver_r01.data import wisig,groups


def subset(data,mask):return {k:v[mask] for k,v in data.items()}


def identities(data):
    return set(zip(data['rx'].tolist(),data['day'].tolist(),data['y'].tolist(),data['position'].tolist()))


def hashes(data):return {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in data.items()}


def diagnostic_data(reg,job):
    if job['dataset']=='wisig':
        rxs=reg['wisig']['roles']['source']
        assert not set(rxs)&set(reg['wisig']['roles']['final_rx'])
        bounds=[(0,80),(80,100),(100,120),(120,160)]
        parts=[wisig(reg,rxs,[0,1],b) for b in bounds];stats=None
    else:
        fold=reg['shen_folds'][job['fold']];keys=fold['inner_train']+fold['inner_validation']
        assert len(keys)==5 and not set(keys)&set(fold['outer_evaluation'])
        assert set(keys)<=set(reg['shen_allowed']);parts=[]
        for lo,hi in [(0,120),(120,140),(140,160),(160,200)]:
            xx=[];yy=[];rr=[];pp=[]
            for key in keys:
                path=Path(reg['shen_cache'])/key
                x=np.load(path/'strict_cis.npy',allow_pickle=False);y=np.load(path/'labels.npy',allow_pickle=False);pos=np.load(path/'rows.npy',allow_pickle=False)
                ids=np.concatenate([np.flatnonzero(y==t)[lo:hi] for t in range(10)])
                xx.append(x[ids]);yy.append(y[ids]);pp.append(pos[ids]);rr.append(np.full(len(ids),reg['shen_allowed'].index(key)))
            parts.append(dict(x=np.concatenate(xx),y=np.concatenate(yy),rx=np.concatenate(rr),day=np.full(sum(map(len,yy)),-1),position=np.concatenate(pp)))
        fit=parts[0]['x'];stats=dict(mean=float(np.mean(fit,dtype=np.float64)),std=float(np.std(fit,dtype=np.float64)),fit_keys=keys,partition='first120_per_TX')
        for data in parts:data['x']=((data['x']-stats['mean'])/stats['std']).astype(np.float32)
    pools=dict(zip(['fit','C1','C2','validation'],parts))
    for name,a in pools.items():
        assert np.isfinite(a['x']).all()
        for other,b in pools.items():
            if name!=other:assert not identities(a)&identities(b),(name,other)
        cell=groups(a);counts=np.bincount(cell)
        expected=(80 if job['dataset']=='wisig' else 120) if name=='fit' else 40 if name=='validation' else 20
        assert np.all(counts==expected),(name,counts)
    return pools,stats
