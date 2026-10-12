"""Immutable, streamed derived feature cache; source access stays development-only."""
import csv
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import scipy
from torch.utils.data import Dataset
from shen_stage1.data import ShenDevelopmentPackets
from shen_stage1.frequency import periodic_phase_offset
from .representations import paired_maps,STFT_CONFIG

ROLES=('train','validation','same_day_diagnostic')
CACHE_VERSION='p2a_stft_pair_v2_pinned_environment'


def write_json(path,obj):
    path=Path(path); temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def cache_dir(root,receiver,role,strict_ib=False):
    return Path(root)/('outputs/shen_p2a/cache_strict_ib' if strict_ib else 'outputs/shen_p2a/cache')/f'{receiver}_{role}'


def metadata_array(file,name,n):
    if name not in file:return np.full(n,np.nan)
    if not isinstance(file.get(name,getlink=True),h5py.HardLink):raise ValueError('Metadata links are forbidden')
    value=np.asarray(file[name][:]).reshape(-1)
    if value.shape!=(n,):raise ValueError('Metadata row alignment changed')
    return value


def build_cache(root,protocol,receiver,role,strict_ib=False):
    root=Path(root); protocol=Path(protocol); out=cache_dir(root,receiver,role,strict_ib)
    if (np.__version__,scipy.__version__)!=('1.26.4','1.13.1'):
        raise RuntimeError('Use /home/keyan/venvs/shen-stage1/bin/python; P1 numerical environment is required')
    variant='_original_FFT_strict_IB' if strict_ib else ''
    fingerprint=hashlib.sha256((CACHE_VERSION+variant+protocol.read_text()+receiver+role+json.dumps(STFT_CONFIG,sort_keys=True)).encode()).hexdigest()
    if out.exists():
        info=json.loads((out/'manifest.json').read_text())
        if info['status']!='completed' or info['fingerprint']!=fingerprint:raise ValueError('Cache is immutable and incomplete/stale; use a new cache version')
        return out
    out.mkdir(parents=True)
    write_json(out/'manifest.json',{'status':'building','fingerprint':fingerprint})
    try:
        ds=ShenDevelopmentPackets(root,protocol,receiver,role,cache_handles=True,signal_dtype=np.float64)
        n=len(ds); arrays={name:np.lib.format.open_memmap(out/f'{name}.npy',mode='w+',dtype=np.float32,shape=(n,1,52,126)) for name in ('spec','cis')}
        vectors=np.lib.format.open_memmap(out/'diagnostic_vectors.npy',mode='w+',dtype=np.float32,shape=(n,208))
        np.save(out/'labels.npy',ds.labels)
        with h5py.File(ds.path,'r') as f:cfo=metadata_array(f,'CFO',n); snr=metadata_array(f,'SNR',n)
        sums={k:0. for k in arrays}; squares={k:0. for k in arrays}; stats=[]
        for start in range(0,n,128):
            rows=[ds[i] for i in range(start,min(n,start+128))]
            x=np.stack([r[0][0]+1j*r[0][1] for r in rows])
            if strict_ib:
                allowed=np.abs(np.fft.fftfreq(8192,d=1e-6))<=62500.
                x=np.fft.ifft(np.fft.fft(x,axis=-1)*allowed,axis=-1)
            spec,cis,denominator=paired_maps(x)
            maps={'spec':spec,'cis':cis}
            for k,a in maps.items():
                arrays[k][start:start+len(rows)]=a
                sums[k]+=float(a.astype(np.float64).sum()); squares[k]+=float(np.square(a.astype(np.float64)).sum())
            vectors[start:start+len(rows)]=np.concatenate([spec[:,0].mean(2),spec[:,0].std(2),cis[:,0].mean(2),cis[:,0].std(2)],axis=1)
            energy=np.abs(denominator)**2
            relative=energy/energy.mean((1,2),keepdims=True)
            phase=periodic_phase_offset(x,lag=1024,sample_rate_hz=1e6)
            for j,(_,label,row) in enumerate(rows):
                record={'receiver':receiver,'role':role,'row':row,'label':label,'raw_label':label+31,
                        'raw_rms':float(np.sqrt(np.mean(np.abs(x[j])**2))),
                        'CFO_metadata_raw_unknown_units':float(cfo[row]) if np.isfinite(cfo[row]) else '',
                        'SNR_metadata_raw_unknown_units':float(snr[row]) if np.isfinite(snr[row]) else '',
                        'cis_mean':float(cis[j].mean()),'cis_std':float(cis[j].std()),'cis_min':float(cis[j].min()),'cis_max':float(cis[j].max()),
                        'cis_abs_q99':float(np.quantile(np.abs(cis[j]),.99)),
                        'cis_fraction_abs_gt_3':float((np.abs(cis[j])>3).mean()),'cis_fraction_abs_gt_6':float((np.abs(cis[j])>6).mean()),
                        'spec_mean':float(spec[j].mean()),'spec_std':float(spec[j].std()),
                        'denominator_power_q01':float(np.quantile(energy[j],.01)),
                        'denominator_fraction_relative_lt_1e6':float((relative[j]<1e-6).mean()),
                        'denominator_fraction_relative_lt_1e4':float((relative[j]<1e-4).mean()),
                        'denominator_fraction_relative_lt_1e2':float((relative[j]<1e-2).mean()),
                        'conditional_periodic_phase_offset_hz':float(phase['all_pairs_offset_hz_under_assumed_fs'][j]),
                        'periodic_coherence':float(phase['coherence'][j])}
                stats.append(record)
        for a in arrays.values():a.flush()
        vectors.flush()
        with (out/'packet_diagnostics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(stats[0]),lineterminator='\n');w.writeheader();w.writerows(stats)
        count=n*52*126
        mean_std={k:{'mean':sums[k]/count,'std':float(np.sqrt(max(squares[k]/count-(sums[k]/count)**2,1e-12)))} for k in arrays}
        source_item=next(i for i in json.loads(protocol.read_text())['files'] if i['receiver']==receiver and i['role']==role)
        write_json(out/'manifest.json',{'status':'completed','version':CACHE_VERSION,'fingerprint':fingerprint,'receiver':receiver,'role':role,
                                       'packets':n,'shape':[n,1,52,126],'source_sha256_from_frozen_audit':source_item['sha256'],
                                       'source_relative_path':source_item['dataset_relative_path'],'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
                                       'stft':STFT_CONFIG,'mean_std':mean_std,'numpy_version':np.__version__,'scipy_version':scipy.__version__,
                                       'strict_original_ib_projection':strict_ib,'diagnostics_raw_rms_is_projected_ib_if_strict':strict_ib,
                                       'relative_denominator_thresholds':{'denominator_fraction_relative_lt_1e6':1e-6,'denominator_fraction_relative_lt_1e4':1e-4,'denominator_fraction_relative_lt_1e2':1e-2},'raw_signal_mutated':False,'metadata_units':'unknown',
                                       'phase_offset_alias_range_hz_under_assumptions':[-488.28125,488.28125]})
        ds.close();return out
    except BaseException as exc:
        write_json(out/'manifest.json',{'status':'failed','fingerprint':fingerprint,'error':repr(exc)})
        raise


class MapPackets(Dataset):
    def __init__(self,root,protocol,receiver,role,representation,normalization='none',strict_ib=False):
        if role not in ROLES:raise PermissionError('Final role cannot be cached or loaded')
        # Revalidate fixed protocol, path and actual labels before using derivatives.
        checked=ShenDevelopmentPackets(root,protocol,receiver,role)
        self.labels=checked.labels.copy();checked.close()
        folder=cache_dir(root,receiver,role,strict_ib); info=json.loads((folder/'manifest.json').read_text())
        if info.get('strict_original_ib_projection',False)!=strict_ib:raise ValueError('Cache information source mismatch')
        if info['status']!='completed' or info['protocol_sha256']!=hashlib.sha256(Path(protocol).read_bytes()).hexdigest():raise ValueError('Stale/incomplete cache')
        np.testing.assert_array_equal(np.load(folder/'labels.npy'),self.labels)
        self.path=folder/f'{representation}.npy';self._array=None
        self.mean=0.;self.std=1.
        if normalization=='train_scalar':
            train=json.loads((cache_dir(root,receiver,'train',strict_ib)/'manifest.json').read_text())
            self.mean=train['mean_std'][representation]['mean'];self.std=train['mean_std'][representation]['std']
        elif normalization!='none':raise ValueError('Only fixed train-only or no feature standardization is allowed')
    def __len__(self):return len(self.labels)
    def __getitem__(self,index):
        if self._array is None:self._array=np.load(self.path,mmap_mode='r')
        x=np.array(self._array[index],copy=True)
        x=(x-self.mean)/self.std
        return x.astype(np.float32),int(self.labels[index]),int(index)
    def __getstate__(self):
        state=self.__dict__.copy();state['_array']=None;return state
