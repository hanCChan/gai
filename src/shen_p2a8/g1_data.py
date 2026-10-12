"""Frozen G1 source/development access and source-only fitted map scalars."""
import hashlib,json
from pathlib import Path
import h5py
import numpy as np
from torch.utils.data import Dataset
from shen_p2a.cache import write_json
from shen_p2a.representations import paired_maps
from shen_stage1.frequency import periodic_phase_offset

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def checked_shen_path(root,item,reg):
    rx=item['receiver'];assert rx in reg['train_receivers']+reg['development_receivers']
    allowed={f'{r}_{k}.h5' for r in reg['train_receivers']+reg['development_receivers'] for k in ('train','test')}
    allowed|={f'Test/{r}_day{d}_wireless_test.h5' for r in ('n210_1','rtl_6') for d in (1,2)}
    rel=item['relative'];assert rel in allowed,'Locked or unregistered source'
    base=(Path(root)/'data/external/shen_receiver_agnostic').resolve();p=(base/rel).resolve()
    assert p==base/rel and p.is_relative_to(base),'Aliases/escaping paths forbidden'
    return p

def cache(root,key):return Path(root)/'outputs/shen_p2a8/g1_cache_v1'/key

def validate_run_roles(cfg,reg):
    assert cfg['method'] in reg['methods']
    if cfg['phase'] in ('cv','confirmation'):
        train_allowed={'shen_'+r+'_train' for r in reg['train_receivers']}
        assert cfg['train_keys'] and set(cfg['train_keys'])<=train_allowed,'Training cannot use developer/locked domains'
        assert cfg['classes']==10
        if cfg['phase']=='cv':
            assert cfg['seed']==reg['cv_seed'] and cfg['epochs']==reg['max_cv_epochs']
            assert set(cfg['validation_keys'])<=train_allowed and not set(cfg['train_keys'])&set(cfg['validation_keys'])
            assert set(cfg['train_keys'])|set(cfg['validation_keys'])==train_allowed
            assert any(set(cfg['validation_keys'])=={'shen_'+r+'_train' for r in fold} for fold in reg['source_receiver_folds'])
        else:
            assert set(cfg['train_keys'])==train_allowed and cfg['seed'] in reg['confirmation_seeds']
            assert set(cfg['evaluation_keys'])<={i['key'] for i in reg['files'] if i['role']!='train'}
            assert 1<=cfg['epochs']<=reg['max_cv_epochs']
    elif cfg['phase']=='osu':
        assert len(cfg['train_keys'])==1 and cfg['train_keys'][0] in ('osu_RX1','osu_RX2')
        assert set(cfg['evaluation_keys'])=={'osu_RX1','osu_RX2'}
        assert cfg['classes']==25 and cfg['epochs']==reg['osu_epochs'] and cfg['seed'] in reg['confirmation_seeds']
    else:raise PermissionError('Unregistered stage')

def save_features(root,key,iq,labels,rows,source):
    folder=cache(root,key);folder.mkdir(parents=True,exist_ok=False);n=len(iq)
    arrays={name:np.lib.format.open_memmap(folder/f'{name}.npy',mode='w+',dtype='float32',shape=((n,2,8192) if name=='iq' else (n,1,52,126))) for name in ('iq','full_cis','strict_cis')}
    sums={k:0. for k in ('full_cis','strict_cis')};squares=sums.copy();freq=np.fft.fftfreq(8192,d=1e-6);ib=np.abs(freq)<=62500
    psd=np.lib.format.open_memmap(folder/'psd128.npy',mode='w+',dtype='float32',shape=(n,128))
    for start in range(0,n,128):
        z=np.asarray(iq[start:start+128],dtype='complex128');assert np.isfinite(z).all()
        arrays['iq'][start:start+len(z)]=np.stack((z.real,z.imag),axis=1)
        full=paired_maps(z)[1];strict=paired_maps(np.fft.ifft(np.fft.fft(z,axis=-1)*ib,axis=-1))[1]
        for name,values in (('full_cis',full),('strict_cis',strict)):
            arrays[name][start:start+len(z)]=values;v=values.astype('float64');sums[name]+=float(v.sum());squares[name]+=float(np.square(v).sum())
        p=np.abs(np.fft.fft(z,axis=-1))**2;p/=p.sum(1,keepdims=True);psd[start:start+len(z)]=np.fft.fftshift(p,axes=-1).reshape(len(z),128,64).sum(-1)
    for a in arrays.values():a.flush()
    psd.flush();np.save(folder/'labels.npy',labels);np.save(folder/'rows.npy',rows)
    count=n*52*126
    stats={k:dict(count=count,sum=sums[k],sum_squares=squares[k]) for k in sums}
    write_json(folder/'manifest.json',dict(status='completed',key=key,packets=n,source=source,stats=stats,num_classes=len(set(labels.tolist())),row_semantics='original HDF5 row or OSU continuous capture start sample'))

def build_shen(root,item,reg):
    p=checked_shen_path(root,item,reg);actual_sha=sha(p)
    if actual_sha!=item['expected_sha256']:raise ValueError('Source SHA differs from prior audit')
    with h5py.File(p,'r') as f:
        assert all(isinstance(f.get(k,getlink=True),h5py.HardLink) for k in ('data','label'))
        raw=np.asarray(f['label'][:]).reshape(-1);assert set(raw)==set(range(31,41))
        assert f['data'].shape==(len(raw),16384)
        ids=np.arange(len(raw))
        if item['role']=='train':
            rng=np.random.default_rng(reg['sample_selection_seed']);ids=np.sort(np.concatenate([rng.choice(np.flatnonzero(raw==c),200,replace=False) for c in range(31,41)]))
        x=np.asarray(f['data'][ids],dtype='float64');iq=x[:,:8192]+1j*x[:,8192:];labels=raw[ids].astype('int64')-31
    save_features(root,item['key'],iq,labels,ids,dict(item,sha256=actual_sha,bytes=p.stat().st_size,dates_sessions='unknown except named drift files'))
    # All ten classes independently reconstructed from raw data, without relying on feature cache output.
    checks=[]
    for c in range(10):
        i=int(np.flatnonzero(labels==c)[0]);expected=paired_maps(iq[i:i+1])[1]
        np.testing.assert_array_equal(np.load(cache(root,item['key'])/'full_cis.npy',mmap_mode='r')[i:i+1],expected)
        checks.append(c)
    print(item['key'],len(iq),'prepared',flush=True)

def build_osu(root,rx,reg):
    batches=[];labels=[];starts=[];sources=[]
    base=Path(root)/'data/raw/osu_lora/Diff_Receivers_Setup/Diff_Receivers_Setup_Indoor_SameTx'/rx
    for tx in range(1,26):
        p=base/f'Device{tx}_IQ.dat';meta=json.loads(p.with_suffix('.sigmf-meta').read_text())['_metadata']
        assert meta['global']['core:datatype']=='cf32' and meta['global']['core:sample_rate']=='1 MS/s'
        assert int(meta['annotations'][0]['wines:transmitter']['ID']['Transmitter ID'])==tx
        mm=np.memmap(p,dtype='<c8',mode='r');n=len(mm)//8192;chosen=[]
        for s in range(0,n,128):
            z=np.asarray(mm[s*8192:min(s+128,n)*8192]).reshape(-1,8192)
            values=periodic_phase_offset(z,1024,1e6)['coherence'];chosen.extend((np.flatnonzero(values>=reg['osu_periodic_coherence_min'])+s).tolist())
        assert len(chosen)>=20,f'{rx} TX{tx} has only {len(chosen)} periodic windows; do not change threshold using target'
        selected=np.array(chosen)[np.linspace(0,len(chosen)-1,min(200,len(chosen)),dtype=int)]
        batches.append(np.asarray(mm.reshape(-1)[:n*8192].reshape(n,8192)[selected]).copy());labels.extend([tx-1]*len(selected));starts.extend((selected*8192).tolist())
        sources.append(dict(receiver=rx,raw_label=tx,path=str(p),metadata=meta,eligible_windows=len(chosen),selected_windows=len(selected),sha256=sha(p)))
        print(rx,tx,len(chosen),'eligible periodic windows',flush=True)
    # Row index includes class/file identity through the corresponding label in sample IDs.
    save_features(root,'osu_'+rx,np.concatenate(batches),np.array(labels,dtype='int64'),np.array(starts,dtype='int64'),sources)

class G1Dataset(Dataset):
    def __init__(self,root,keys,method,stats=None):
        self.root=Path(root);self.method=method;self.keys=keys;self.arrays=[];self.labels=[];self.rows=[];self.boundaries=[0];self.stats=stats
        name=method if method in ('full_cis','strict_cis') else 'iq'
        for key in keys:
            folder=cache(root,key);info=json.loads((folder/'manifest.json').read_text());assert info['status']=='completed'
            a=np.load(folder/f'{name}.npy',mmap_mode='r');y=np.load(folder/'labels.npy');r=np.load(folder/'rows.npy');assert len(a)==len(y)==len(r)
            self.arrays.append(a);self.labels.append(y);self.rows.append(r);self.boundaries.append(self.boundaries[-1]+len(y))
        self.labels_all=np.concatenate(self.labels)
    def __len__(self):return self.boundaries[-1]
    def __getitem__(self,index):
        part=int(np.searchsorted(self.boundaries,index,side='right')-1);local=index-self.boundaries[part]
        x=np.array(self.arrays[part][local],copy=True)
        if self.stats:x=(x-self.stats['mean'])/self.stats['std']
        y=int(self.labels[part][local]);row=int(self.rows[part][local]);sample_id=f'{self.keys[part]}:TX{y}:row{row}'
        return x.astype('float32'),y,sample_id

def fit_scaler(root,keys,method):
    if method not in ('full_cis','strict_cis'):return None
    items=[json.loads((cache(root,key)/'manifest.json').read_text())['stats'][method] for key in keys]
    n=sum(r['count'] for r in items);mean=sum(r['sum'] for r in items)/n;var=sum(r['sum_squares'] for r in items)/n-mean**2
    return dict(mean=mean,std=float(np.sqrt(var)),fit_source_keys=keys)
