"""Uniform original-FFT counterfactuals and train-derived feature scaling."""
import hashlib,json
from pathlib import Path
import numpy as np
from shen_stage1.data import ShenDevelopmentPackets
from shen_p2a.cache import MapPackets,cache_dir,write_json
from shen_p2a.representations import paired_maps,STFT_CONFIG

def removed_mask(variant,registration):
    f=np.fft.fftfreq(8192,d=1e-6)
    if variant=='full':return np.zeros(8192,dtype=bool)
    if variant=='strict_ib':return np.abs(f)>62500
    settings=registration['notch']
    negative=(f>=settings['negative_low_hz'])&(f<settings['negative_high_hz'])
    if variant=='notch_negative':return negative
    if variant=='notch_positive':
        # Exact reflection of FFT-bin set, rather than a separately rounded interval.
        return negative[(-np.arange(8192))%8192]
    if variant=='notch_random':
        return (f>=settings['random_low_hz'])&(f<settings['random_high_hz'])
    raise ValueError(variant)

def project(iq,variant,registration):
    return np.fft.ifft(np.fft.fft(iq,axis=-1)*(~removed_mask(variant,registration)),axis=-1)

def feature_folder(root,rx,role,variant):
    if variant in ('full','strict_ib'):return cache_dir(root,rx,role,variant=='strict_ib')
    return Path(root)/'outputs/shen_p2a8/cache_v1'/f'{rx}_{role}_{variant}'

def build(root,rx,role,variant,registration):
    assert role in ('train','validation')
    root=Path(root);protocol=root/'configs/shen_evidence/protocol.candidate.json'
    folder=feature_folder(root,rx,role,variant)
    folder.mkdir(parents=True,exist_ok=False)
    ds=ShenDevelopmentPackets(root,protocol,rx,role,True,np.float64)
    a=np.lib.format.open_memmap(folder/'spec.npy',mode='w+',dtype='float32',shape=(len(ds),1,52,126))
    total=squares=0.;count=len(ds)*52*126
    for start in range(0,len(ds),128):
        rows=[ds[i] for i in range(start,min(start+128,len(ds)))]
        iq=np.stack([r[0][0]+1j*r[0][1] for r in rows])
        maps=paired_maps(project(iq,variant,registration))[0]
        a[start:start+len(rows)]=maps;values=maps.astype('float64');total+=values.sum();squares+=np.square(values).sum()
    a.flush();np.save(folder/'labels.npy',ds.labels);ds.close()
    item=next(i for i in json.loads(protocol.read_text())['files'] if i['receiver']==rx and i['role']==role)
    write_json(folder/'manifest.json',dict(status='completed',variant=variant,receiver=rx,role=role,packets=len(a),shape=list(a.shape),
        stft=STFT_CONFIG,source_sha256_from_frozen_audit=item['sha256'],protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        registration_sha256=hashlib.sha256((root/'configs/shen_p2a8/registration.json').read_bytes()).hexdigest(),
        mean_std={'spec':dict(mean=float(total/count),std=float(np.sqrt(squares/count-(total/count)**2)))}))

def scaler(root,rx,variant,mode):
    def stats(v):return json.loads((feature_folder(root,rx,'train',v)/'manifest.json').read_text())['mean_std']['spec']
    if mode=='independent':return stats(variant)
    if mode!='common':raise ValueError(mode)
    # Equal-weight pooled Full + Strict train maps, per receiver. No Day2 statistics.
    a,b=stats('full'),stats('strict_ib');mean=(a['mean']+b['mean'])/2
    variance=(a['std']**2+a['mean']**2+b['std']**2+b['mean']**2)/2-mean**2
    return dict(mean=mean,std=float(np.sqrt(variance)))

class CounterfactualMaps(MapPackets):
    def __init__(self,root,protocol,rx,role,variant,mode):
        checked=ShenDevelopmentPackets(root,protocol,rx,role);self.labels=checked.labels.copy();checked.close()
        folder=feature_folder(root,rx,role,variant);info=json.loads((folder/'manifest.json').read_text())
        if info['status']!='completed' or info['protocol_sha256']!=hashlib.sha256(Path(protocol).read_bytes()).hexdigest():raise ValueError('Invalid cache')
        np.testing.assert_array_equal(self.labels,np.load(folder/'labels.npy'))
        self.path=folder/'spec.npy';self._array=None
        stats=scaler(root,rx,variant,mode);self.mean=stats['mean'];self.std=stats['std']
