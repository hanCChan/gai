import sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from shen_next.fair_attention import FairAttention
from shen_next.multi_receiver import Reverse,ReceiverAdversarial,SpecSiamese,nt_xent,cross_receiver_pairs,fractional_tdl_awgn
class PreparedTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(2)
    def test_same_tokenizer_capacity_and_no_oob_main_leak(self):
        h=FairAttention('hstu');torch.manual_seed(2);t=FairAttention('transformer')
        for k,p in h.main_tokens.named_parameters():torch.testing.assert_close(p,dict(t.main_tokens.named_parameters())[k],rtol=0,atol=0)
        count=lambda m:sum(p.numel() for p in m.parameters())
        self.assertLess(abs(count(h)-count(t))/count(h),.01)
        for e in ('hstu','transformer'):
            cross=FairAttention(e,'cross_attention');control=FairAttention(e,main_capacity_hidden=723)
            self.assertLess(abs(count(cross)-count(control))/count(cross),.001)
            self.assertEqual(count(cross),count(FairAttention(e,'cross_attention',aux_band='ib_control')))
        x=torch.randn(2,2,8192);torch.testing.assert_close(h.tokenize(x),t.tokenize(x),rtol=0,atol=0)
        x=x.double();z=torch.fft.fftshift(torch.fft.fft(torch.complex(x[:,0],x[:,1])),dim=-1)
        f=torch.fft.fftshift(torch.fft.fftfreq(8192,d=1e-6));altered=z+100*torch.randn_like(z)*(f.abs()>75000.)
        q=torch.fft.ifft(torch.fft.ifftshift(altered,dim=-1));changed=torch.stack((q.real,q.imag),1)
        h=h.double();torch.testing.assert_close(h.tokenize(x),h.tokenize(changed),atol=1e-12,rtol=1e-12)
        h=h.float();x=x.float()
        for m in (h,t,FairAttention('hstu','cross_attention'),FairAttention('transformer','cross_attention'),FairAttention('hstu',main_capacity_hidden=723),FairAttention('transformer','cross_attention',aux_band='ib_control')):
            out=m(x);self.assertEqual(out.shape,(2,10));out.sum().backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
    def test_grl_gradient_and_multi_receiver_constraint(self):
        x=torch.randn(3,requires_grad=True);Reverse.apply(x,.4).sum().backward();torch.testing.assert_close(x.grad,torch.full_like(x,-.4))
        with self.assertRaises(ValueError):ReceiverAdversarial(1)
        m=ReceiverAdversarial(10);out=m(torch.randn(2,1,52,126));self.assertEqual(out['receiver_logits'].shape,(2,10))
    def test_pairing_and_contrastive_positive_indices(self):
        y=np.repeat(np.arange(10),4);r=np.tile([0,0,1,1],10);pairs=cross_receiver_pairs(y,r)
        for a,b,label in pairs:self.assertEqual(y[a],y[b]);self.assertNotEqual(r[a],r[b])
        z=torch.eye(10);good=torch.cat((z,z)).requires_grad_();bad=torch.cat((z,z.roll(1,0)))
        self.assertLess(nt_xent(good),nt_xent(bad));nt_xent(good).backward();self.assertTrue(torch.isfinite(good.grad).all())
    def test_subsample_delay_survives_and_reproducible(self):
        x=np.exp(2j*np.pi*.13*np.arange(8192));a,info=fractional_tdl_awgn(x,delay_spread_ns=5.,seed=5);b,_=fractional_tdl_awgn(x,delay_spread_ns=5.,seed=5)
        np.testing.assert_array_equal(a,b);self.assertTrue(np.isfinite(a).all());self.assertTrue(any(0<d<1 for d in info['delay_samples']));self.assertAlmostEqual(info['effective_discrete_kernel_energy'],1.)
if __name__=='__main__':unittest.main()
