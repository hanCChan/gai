import sys
import unittest
import tempfile
import json
import pickle
import numpy as np
import h5py
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from sklearn.metrics import f1_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rfhstu.losses import macro_f1_from_logits
from rfhstu.cnn_baseline import build_cnn_input
from rfhstu.models import RFPatchEmbedder, DeviceClassifier
from shen_evidence.inputs import build_input,frequency_masks
from shen_stage1.protocol import development_path
from shen_stage1.data import ShenDevelopmentPackets


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4); torch.set_num_threads(2)
    def test_macro_f1_unbalanced_and_partition_invariance(self):
        labels=torch.tensor([0]*9+[1,2,2]); pred=torch.tensor([0]*8+[1,0,0,2])
        logits=torch.nn.functional.one_hot(pred,3).float()
        expected=f1_score(labels,pred,labels=[0,1,2],average='macro',zero_division=0)
        self.assertAlmostEqual(macro_f1_from_logits(logits,labels,3),expected,places=6)
        self.assertNotAlmostEqual(expected,f1_score(labels,pred,average='weighted'),places=3)
        parts=[logits[:3],logits[3:10],logits[10:]]
        self.assertAlmostEqual(macro_f1_from_logits(torch.cat(parts),labels,3),expected,places=6)
    def test_frequency_isolation_and_no_excluded_scale_leak(self):
        n=8192; t=torch.arange(n)/1e6
        z=torch.exp(2j*torch.pi*(50*1e6/n)*t)+.2*torch.exp(2j*torch.pi*(2000*1e6/n)*t)
        iq=torch.stack((z.real,z.imag))[None]
        _,ib,oob=frequency_masks(n,iq.device)
        strict=build_input(iq,'strict_ib')
        sp=torch.fft.fftshift(torch.fft.fft(torch.complex(strict[:,0],strict[:,1])),dim=-1)
        self.assertLess(float(sp[:,~ib].abs().max()/sp.abs().max()),1e-6)
        oo=build_input(iq,'oob_only')
        self.assertEqual(float(oo[:,:,~oob].abs().max()),0.)
        self.assertFalse(bool((ib&oob).any()))
        # Add purely excluded evidence: the allowed representation stays fixed.
        spec=torch.randn(1,n,dtype=torch.complex64)
        modified=spec+100*torch.randn(1,n,dtype=torch.complex64)*ib
        def channels(s):
            zz=torch.fft.ifft(torch.fft.ifftshift(s,dim=-1),norm='ortho')
            return torch.stack((zz.real,zz.imag),dim=1)
        torch.testing.assert_close(build_input(channels(spec),'oob_only'),build_input(channels(modified),'oob_only'),atol=5e-5,rtol=5e-5)
    def test_legacy_fft_oob_really_excludes_ib(self):
        iq=torch.randn(2,2,8192)
        full=build_cnn_input(iq,'fft'); oob=build_cnn_input(iq,'fft_oob')
        _,ib,_=frequency_masks(8192,iq.device,guard_hz=0)
        self.assertEqual(float(oob[:,:,ib].abs().max()),0.)
        self.assertFalse(torch.allclose(full,oob))
    def test_dropout_clears_copied_concat_for_both_embedders(self):
        for embed_type in ('linear','cnn_stem'):
            embed=RFPatchEmbedder(window_size=512,patch_size=32,dim=16,patch_embed_type=embed_type,oob_fusion_type='concat_oob')
            model=DeviceClassifier(embed,10,dim=16,depth=1,oob_dropout=1.).train()
            iq=torch.randn(2,2,512)
            captured=[]
            handle=model.encoder.input_proj.register_forward_pre_hook(lambda _,args:captured.append(args[0].detach().clone()))
            dropped=model(iq)['embedding'].detach(); handle.remove()
            self.assertEqual(float(captured[0][...,-32:].abs().max()),0.)
            model.oob_dropout=0.; keep=model(iq)['embedding'].detach()
            self.assertFalse(torch.allclose(dropped,keep))
    def test_worker_handles_are_private_readonly_and_pickle_safe(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); data=root/'data'/'Train'; data.mkdir(parents=True)
            path=data/'n210_1_day1_wireless_train.h5'
            signals=np.arange(10*16384,dtype=np.float64).reshape(10,16384)
            with h5py.File(path,'w') as f:
                f['data']=signals; f['label']=np.arange(31,41)
            protocol={'data_root':'data','label_map':{str(i+31):i for i in range(10)},
                      'files':[{'receiver':'n210_1','role':'train','dataset_relative_path':'Train/n210_1_day1_wireless_train.h5','bytes':path.stat().st_size}]}
            pp=root/'protocol.json'; pp.write_text(json.dumps(protocol))
            ds=ShenDevelopmentPackets(root,pp,'n210_1','train',cache_handles=True)
            self.assertEqual(ds[3][1],3)
            restored=pickle.loads(pickle.dumps(ds))
            self.assertIsNone(restored._handle)
            for iq,label,row in DataLoader(ds,batch_size=3,num_workers=2):
                for j in range(len(row)):
                    torch.testing.assert_close(iq[j],torch.from_numpy(signals[int(row[j])].reshape(2,8192).astype(np.float32)))
                    self.assertEqual(int(label[j]),int(row[j]))
            ds.close(); restored.close()
            with h5py.File(path,'r') as f: np.testing.assert_array_equal(f['data'][:],signals)

    def test_allowed_filename_cannot_alias_locked_file(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); (root/'Test').mkdir()
            locked=root/'Test'/'n210_1_day3_wireless_test.h5'; locked.touch()
            allowed=root/'Test'/'n210_1_day2_wireless_test.h5'; allowed.symlink_to(locked)
            with self.assertRaises(PermissionError):
                development_path(root,'Test/n210_1_day2_wireless_test.h5')

    def test_final_data_access_denied(self):
        for rel in ('Test/n210_1_day3_wireless_test.h5','Test/rtl_6_day4_wireless_test.h5','rtl_6_test.h5','../x'):
            with self.assertRaises(PermissionError): development_path(Path('/tmp'),rel)

if __name__=='__main__': unittest.main()
