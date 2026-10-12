import sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shen_p2a.representations import paired_maps
from shen_p2a.bands import masks,spectrum,selected_view,transform
from shen_p2a.models import FactorizedAuthorCNN,SimpleFusion,SingleIB
from shen_evidence.inputs import build_input
from shen_evidence.models import AuthorCISCNN,PacketCNN
from shen_stage1.representations import channel_independent_spectrogram

class P2Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(10);torch.set_num_threads(2)
    def test_author_cis_unchanged_and_spec_geometry(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(3,8192))+1j*rng.normal(size=(3,8192))
        spec,cis,den=paired_maps(x)
        np.testing.assert_array_equal(cis,channel_independent_spectrogram(x,dtype=np.float32))
        self.assertEqual(spec.shape,cis.shape);self.assertEqual(den.shape,(3,52,126))
        self.assertTrue(np.isfinite(spec).all())
    def test_factorized_capacity_preserves_encoder_and_classifier(self):
        torch.manual_seed(2);big=AuthorCISCNN()
        torch.manual_seed(2);light=FactorizedAuthorCNN()
        for name,p in big.named_parameters():
            if not name.startswith('feature.'):
                torch.testing.assert_close(p,dict(light.named_parameters())[name],rtol=0,atol=0)
        count=lambda m:sum(p.numel() for p in m.parameters())
        self.assertLess(abs(count(light)-count(PacketCNN()))/count(PacketCNN()),.01)
    def test_ring_masks_disjoint_and_cover_oob(self):
        _,m=masks(8192,'cpu')
        for a,b in [('near','mid'),('mid','far'),('near','far'),('ib','oob')]:self.assertFalse(bool((m[a]&m[b]).any()))
        self.assertTrue(torch.equal(m['near']|m['mid']|m['far'],m['oob']))
    def test_legacy_ib_and_oob_are_identical_in_canonical_case(self):
        x=torch.randn(3,2,8192);s=spectrum(x)
        torch.testing.assert_close(selected_view(s,'ib',coordinate='time'),build_input(x,'strict_ib'))
        torch.testing.assert_close(selected_view(s,'oob'),build_input(x,'oob_only'))
    def test_excluded_bins_cannot_leak_through_window_or_independent_scale(self):
        n=8192;s=torch.randn(2,n,dtype=torch.complex128);_,m=masks(n,'cpu')
        for band in ('ib','near','mid','far','oob'):
            altered=s+30*torch.randn_like(s)*(~m[band])
            for window in ('rect','projection_hann'):
                for coordinate in ('time','frequency'):
                    torch.testing.assert_close(selected_view(s,band,coordinate=coordinate,window=window),
                                               selected_view(altered,band,coordinate=coordinate,window=window),rtol=0,atol=0)
        with self.assertRaises(ValueError):selected_view(s,'oob',window='hann_before_projection')
    def test_relative_ib_scale_is_explicit_and_energy_ratio_preserved(self):
        s=torch.randn(2,8192,dtype=torch.complex64);_,m=masks(8192,'cpu')
        out=selected_view(s,'oob',normalization='ib_relative')
        expected=(s[:,m['oob']].abs().square().sum(1)/s[:,m['ib']].abs().square().sum(1))
        actual=out.square().sum(1).mean(1)
        torch.testing.assert_close(actual,expected)
    def test_fusion_main_invariant_to_auxiliary_and_parameter_controls(self):
        n=8192;s=torch.randn(2,n,dtype=torch.complex128);_,m=masks(n,'cpu')
        def iq(a):
            z=torch.fft.ifft(torch.fft.ifftshift(a,dim=-1),norm='ortho');return torch.stack((z.real,z.imag),1)
        cfg={'family':'fusion','aux_band':'oob','sample_rate_hz':1e6,'bandwidth_hz':125e3,'guard_hz':12500.}
        a=transform(iq(s),cfg);b=transform(iq(s+20*torch.randn_like(s)*m['oob']),cfg)
        torch.testing.assert_close(a['main'],b['main'],atol=1e-13,rtol=1e-13)
        self.assertFalse(torch.allclose(a['aux'],b['aux']))
        concat=SimpleFusion('concat');gate=SimpleFusion('scalar_gate')
        count=lambda x:sum(p.numel() for p in x.parameters())
        self.assertLess(abs(count(concat)-count(gate))/count(concat),.001)
        for model in (concat,gate,SingleIB()):
            x={k:v.float() for k,v in a.items()};out=model(x);out['logits'].sum().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

if __name__=='__main__':unittest.main()
