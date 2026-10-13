#!/usr/bin/env python3
"""Read-only hashed source IQ cache, synthetic checks, and pre-fit seal."""
import json
import datetime
from pathlib import Path
import h5py
import numpy as np
from receiver_transform_common import dump, sha, mappings, distances
ROOT=Path(__file__).resolve().parents[1]

def main():
    reg=json.loads((ROOT/'configs/receiver_transform/registration.json').read_text())
    out=ROOT/'outputs/receiver_transform'
    out.mkdir(parents=True,exist_ok=False)
    cache=out/'iq_cache';cache.mkdir()
    checks=[]
    b1=json.loads((ROOT/'configs/b1_baselines/registration.json').read_text())
    for item in reg['source_files']:
        rx=item['receiver'];assert rx in reg['source_receivers']
        assert item['relative']==rx+'_train.h5'
        raw=Path(reg['raw_root'])/item['relative']
        assert raw.resolve()==raw and sha(raw)==item['expected_sha256']
        old=Path(reg['history_root'])/'outputs/shen_p2a8/g1_cache_v1'/item['key']
        for name in ['labels.npy','rows.npy']:
            rel=str((old/name).relative_to(reg['history_root']))
            assert sha(old/name)==b1['feature_assets'][rel]['sha256']
        rows=np.load(old/'rows.npy',allow_pickle=False)
        labels=np.load(old/'labels.npy',allow_pickle=False)
        assert np.array_equal(np.bincount(labels,minlength=10),np.full(10,200))
        x=np.empty((10,200,2,8192),np.float32)
        with h5py.File(raw,'r') as hf:
            assert isinstance(hf.get('data',getlink=True),h5py.HardLink)
            assert isinstance(hf.get('label',getlink=True),h5py.HardLink)
            yy=np.asarray(hf['label'][:]).reshape(-1).astype(int)-31
            np.testing.assert_array_equal(yy[rows],labels)
            assert hf['data'].shape[1]==16384
            for tx in range(10):
                ids=rows[labels==tx];assert np.all(np.diff(ids)>0)
                for begin in range(0,200,40):
                    value=np.asarray(hf['data'][ids[begin:begin+40]],np.float64)
                    z=value[:,:8192]+1j*value[:,8192:]
                    scale=np.sqrt(np.mean(abs(z)**2,axis=1))
                    assert np.all(np.isfinite(z)) and np.all(scale>0)
                    z=z/scale[:,None]
                    x[tx,begin:begin+len(ids[begin:begin+40]),0]=z.real
                    x[tx,begin:begin+len(ids[begin:begin+40]),1]=z.imag
        path=cache/(rx+'.npy');np.save(path,x,allow_pickle=False)
        checks.append(dict(receiver=rx,raw_sha256=item['expected_sha256'],raw_bytes=raw.stat().st_size,
            rows_sha256=sha(old/'rows.npy'),labels_sha256=sha(old/'labels.npy'),
            cache_path=str(path),cache_sha256=sha(path),shape=list(x.shape),dtype=str(x.dtype)))
        print('Prepared source-only RMS IQ:',rx,flush=True)
    rng=np.random.default_rng(20261013)
    z=rng.normal(size=(8,40,32));offset=rng.normal(size=32)
    maps=mappings(z,z+offset)
    for method in ['translation','lowrank']:
        a,b=maps[method];assert np.max(abs(z@a+b-(z+offset)))<1e-10
    assert distances(z[0],z[0])[1]<1e-10
    # Nontrivial shared rank-one residual should transfer to unseen synthetic identities.
    u=rng.normal(size=32);v=rng.normal(size=32)
    a=np.eye(32)+np.outer(u,v)/64
    full=rng.normal(size=(10,200,32))+rng.normal(size=(10,1,32))*2
    fit=mappings(full[:8],full[:8]@a+offset)['lowrank']
    errors=[distances(full[i],full[i]@a+offset)[0] for i in [8,9]]
    mapped=[distances(full[i]@fit[0]+fit[1],full[i]@a+offset)[0] for i in [8,9]]
    assert np.mean(mapped)<np.mean(errors)
    dump(out/'preflight.json',dict(inputs=checks,synthetic=dict(identity_distance=True,
        shared_translation_exact=True,known_lowrank_generalizes=True),signals_published=False))
    files=['configs/receiver_transform/registration.json','docs/receiver_transform/EXECUTION_PLAN.md',
           'scripts/receiver_transform_common.py','scripts/prepare_receiver_transform.py',
           'scripts/train_receiver_transform_gate.py','scripts/analyze_receiver_transform_gate.py',
           'scripts/launch_receiver_transform_gate.py']
    dump(out/'pre_execution_seal.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        files_sha256={p:sha(ROOT/p) for p in files},preflight_sha256=sha(out/'preflight.json'),
        before_fitting=True))

if __name__=='__main__':main()
