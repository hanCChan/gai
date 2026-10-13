#!/usr/bin/env python3
"""Read compact dataset structure with inert NumPy placeholders, never arrays.

Only three NumPy pickle constructors are mapped to local inert classes.
All other globals and persistent IDs are rejected. Signal buffers are consumed
for structural byte-length validation, not converted or statistically analyzed.
"""
import argparse
import hashlib
import json
import math
import pickle
import zipfile
from pathlib import Path

class MetadataDtype:
    def __init__(self, code, *args):
        if code not in ('f8','f4','i8','i4','c8','c16','<f8','<f4'):
            raise ValueError('Unexpected dtype')
        self.code=code
        self.itemsize=int(''.join(x for x in code if x.isdigit()))
    def __setstate__(self, state):
        self.state=state

class MetadataArray:
    def __setstate__(self, state):
        version,shape,dtype,fortran,buffer=state
        if version not in (0,1) or not isinstance(dtype,MetadataDtype):
            raise ValueError('Unexpected array state')
        if not isinstance(buffer,(bytes,bytearray)):
            raise ValueError('Object array payloads are forbidden')
        if not isinstance(shape,tuple) or not all(isinstance(x,int) and 0<=x<=2048 for x in shape):
            raise ValueError('Unexpected shape')
        if len(buffer)!=math.prod(shape)*dtype.itemsize:
            raise ValueError('Array payload byte-length mismatch')
        self.shape=shape;self.dtype=dtype.code;self.payload_bytes=len(buffer)

def reconstruct(kind,shape,code):
    if kind is not MetadataArray or shape!=(0,) or code!=b'b':
        raise ValueError('Unexpected array reconstruction')
    return MetadataArray()

class MetadataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name=='_reconstruct':return reconstruct
        if module=='numpy' and name=='ndarray':return MetadataArray
        if module=='numpy' and name=='dtype':return MetadataDtype
        raise pickle.UnpicklingError('Disallowed global: '+module+'.'+name)
    def persistent_load(self, identifier):
        raise pickle.UnpicklingError('Persistent identifiers forbidden')

def inspect(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist()==['ManyRx.pkl']
        with archive.open('ManyRx.pkl') as stream:
            root=MetadataOnlyUnpickler(stream).load()
            assert stream.read(1)==b''
    expected={'tx_list','rx_list','capture_date_list','equalized_list','max_sig','data'}
    assert type(root) is dict and expected<=set(root)
    tx=root['tx_list'];rx=root['rx_list'];dates=root['capture_date_list'];eq=root['equalized_list']
    assert len(tx)==10 and len(set(tx))==10
    assert len(rx)==32 and len(set(rx))==32
    assert len(dates)==4 and len(set(dates))==4 and eq==[0,1]
    assert root['max_sig']==200
    data=root['data'];assert len(data)==10
    inventory=[]
    for ti,group in enumerate(data):
        assert len(group)==32
        for ri,rdays in enumerate(group):
            assert len(rdays)==4
            for di,ed in enumerate(rdays):
                assert len(ed)==2
                for ei,array in enumerate(ed):
                    assert type(array) is MetadataArray and array.shape[1:]==(256,2)
                    assert 0<=array.shape[0]<=200
                    inventory.append(dict(tx_index=ti,rx_index=ri,day_index=di,equalized=eq[ei],
                        packets=array.shape[0],shape=list(array.shape),dtype=array.dtype,payload_bytes=array.payload_bytes))
    counts={str(e):sum(x['packets'] for x in inventory if x['equalized']==e) for e in eq}
    return dict(state='STRUCTURE_CONSISTENT_WITH_OFFICIAL_MANYRX',source_kind='third_party_kaggle',
        official_byte_equivalence_proven=False,fields=sorted(root),tx_count=10,rx_count=32,date_count=4,
        dates=dates,equalized_list=eq,max_sig=200,array_count=len(inventory),packets_per_version=counts,
        empty_arrays=sum(x['packets']==0 for x in inventory),
        fewer_than_200_arrays=sum(x['packets']<200 for x in inventory),
        payload_bytes=sum(x['payload_bytes'] for x in inventory),
        numerical_arrays_constructed=False,signal_statistics_computed=False,
        arbitrary_pickle_globals_executed=False,signal_payload_bytes_consumed_for_structure=True,
        sealed_for_research=True,arrays=inventory)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=inspect(args.archive)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='arrays'}))
