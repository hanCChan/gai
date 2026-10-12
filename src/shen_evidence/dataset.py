import numpy as np
from shen_stage1.data import ShenDevelopmentPackets
from shen_stage1.representations import channel_independent_spectrogram


class EvidencePackets(ShenDevelopmentPackets):
    def __init__(self,root,protocol,receiver,role,kind,cache_handles=True):
        super().__init__(root,protocol,receiver,role,cache_handles=cache_handles,
                         signal_dtype=np.float64 if kind=='cis' else np.float32)
        self.kind=kind
    def __getitem__(self,index):
        iq,label,row=super().__getitem__(index)
        if self.kind=='cis':
            z=(iq[0]+1j*iq[1])[None,:]
            iq=channel_independent_spectrogram(z,dtype=np.float32)[0]
        return iq,label,row
