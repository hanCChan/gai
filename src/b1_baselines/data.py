"""Read-only source CIS and a shared, saved balanced epoch schedule."""
import hashlib,json
from pathlib import Path
import numpy as np
from shen_p2a8.g1_data import G1Dataset,fit_scaler

class SourceCIS(G1Dataset):
    def __init__(self,root,keys,allowed,stats):
        assert keys and set(keys)<=set(allowed)
        for key in keys:
            folder=Path(root)/'outputs/shen_p2a8/g1_cache_v1'/key
            m=json.loads((folder/'manifest.json').read_text())
            assert m['source']['role']=='train' and m['key']==key and m['packets']==2000
            y=np.load(folder/'labels.npy',allow_pickle=False)
            assert np.array_equal(np.bincount(y,minlength=10),np.full(10,200))
        super().__init__(root,keys,'strict_cis',stats)
        self.group_ids=np.concatenate([rx*10+y for rx,y in enumerate(self.labels)])
        assert all(a.shape==(2000,1,52,126) and a.dtype==np.float32 for a in self.arrays)
    def __getitem__(self,index):
        x,y,sample=super().__getitem__(index)
        assert np.isfinite(x).all()
        return x,y,int(self.group_ids[index]),int(self.group_ids[index]//10),sample

def schedule_from_groups(group_ids,epochs,seed):
    groups=np.unique(group_ids);assert np.array_equal(groups,np.arange(len(groups)))
    members=[np.flatnonzero(group_ids==g) for g in groups]
    assert all(len(m)==200 for m in members)
    rng=np.random.default_rng(seed);result=[]
    for _ in range(epochs):
        per_group=np.stack([rng.permutation(m) for m in members]).reshape(len(groups),100,2)
        batches=[]
        for batch in range(100):batches.append(per_group[rng.permutation(len(groups)),batch,:].reshape(-1))
        result.append(batches)
    return np.asarray(result,dtype=np.int64)

def validate_schedule(schedule,group_ids,epochs):
    groups=len(np.unique(group_ids));assert schedule.shape==(epochs,100,2*groups)
    wanted=np.arange(len(group_ids))
    for epoch in schedule:
        assert np.array_equal(np.sort(epoch.reshape(-1)),wanted)
        for batch in epoch:assert np.array_equal(np.bincount(group_ids[batch],minlength=groups),np.full(groups,2))

class EpochBatches:
    def __init__(self,schedule):self.schedule=schedule;self.epoch=0
    def __len__(self):return 100
    def __iter__(self):
        for indices in self.schedule[self.epoch]:yield indices.tolist()
