#!/usr/bin/env python3
"""Fresh ten-class raw IQ rebuild of Full/Strict CIS for every permitted Shen cache."""
import json,sys
from pathlib import Path
import h5py,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shen_p2a8.g1_data import cache,checked_shen_path
from shen_p2a.representations import paired_maps
from shen_p2a.cache import write_json
if __name__=='__main__':
    reg=json.loads((ROOT/'configs/shen_p2a8/g1_registration.json').read_text());mask=np.abs(np.fft.fftfreq(8192,d=1e-6))<=62500;checks=[]
    for item in reg['files']:
        folder=cache(ROOT,item['key']);labels=np.load(folder/'labels.npy');rows=np.load(folder/'rows.npy');indices=np.array([np.flatnonzero(labels==c)[0] for c in range(10)])
        with h5py.File(checked_shen_path(ROOT,item,reg),'r') as f:
            raw=np.stack([np.asarray(f['data'][int(rows[i])]) for i in indices]);y=np.array([int(f['label'][0,int(rows[i])])-31 for i in indices]);np.testing.assert_array_equal(y,labels[indices])
        z=raw[:,:8192]+1j*raw[:,8192:];strict=np.fft.ifft(np.fft.fft(z,axis=-1)*mask,axis=-1)
        for name,value in (('full_cis',z),('strict_cis',strict)):
            expected=paired_maps(value)[1];actual=np.load(folder/f'{name}.npy',mmap_mode='r')[indices];np.testing.assert_array_equal(expected,actual)
        np.testing.assert_array_equal(np.stack((z.real,z.imag),axis=1).astype('float32'),np.load(folder/'iq.npy',mmap_mode='r')[indices])
        checks.append(dict(key=item['key'],ten_class_labels_exact=True,ten_class_IQ_float32_exact=True,full_cis_exact=True,strict_cis_exact=True))
    write_json(ROOT/'outputs/shen_p2a8/g1_fresh_raw_map_verification.json',dict(passed=True,checks=checks,source_files=28,locked_accessed=False))
