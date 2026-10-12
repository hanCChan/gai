import json,tempfile,unittest
from pathlib import Path
import numpy as np
from shen_p2a8.g1_data import checked_shen_path,fit_scaler,cache,validate_run_roles
from shen_p2a8.features import removed_mask,project
from shen_p2a8.reliability import source_statistics,fit_scaler as fit_vector_scaler,fit_ridge,ridge_logits,band_columns,rank_correlation

class MechanismAccessTests(unittest.TestCase):
    def test_reliability_uses_conditional_receiver_variation(self):
        result=source_statistics(np.array([[0.],[2.],[1.],[3.]]),np.array([0,1,0,1]),np.array([0,0,1,1]),classes=2)
        self.assertAlmostEqual(result['D_between'],4.)
        self.assertAlmostEqual(result['V_RX_given_TX'],.25)
        self.assertAlmostEqual(result['reliability_score'],16.,places=5)
        with self.assertRaises(ValueError):source_statistics(np.array([[0.],[2.]]),np.array([0,1]),np.array([0,0]),classes=2)

    def test_degenerate_features_do_not_receive_high_reliability(self):
        result=source_statistics(np.zeros((4,2)),np.array([0,1,0,1]),np.array([0,0,1,1]),classes=2)
        self.assertTrue(result['degenerate_feature']);self.assertIsNone(result['reliability_score'])
        self.assertIsNone(rank_correlation([1,1,1],[0,1,2]))

    def test_source_ridge_classifies_separated_classes_and_equal_bands_cover_all_features(self):
        x=np.array([[0.],[2.],[1.],[3.]]);weights=fit_ridge(x,np.array([0,1,0,1]),classes=2)
        np.testing.assert_array_equal(ridge_logits(x,weights).argmax(1),[0,1,0,1])
        columns=np.concatenate([band_columns(b) for b in range(4)])
        np.testing.assert_array_equal(np.sort(columns),np.arange(104))

    def test_confirmation_cannot_train_on_developer_receiver(self):
        reg=dict(methods=['full_iq'],train_receivers=['rtl_1'],confirmation_seeds=[1],max_cv_epochs=20,files=[dict(key='shen_rtl_4_test',role='evaluation')])
        cfg=dict(method='full_iq',phase='confirmation',train_keys=['shen_rtl_4_test'],evaluation_keys=['shen_rtl_4_test'],classes=10,epochs=10,seed=1)
        with self.assertRaises(AssertionError):validate_run_roles(cfg,reg)
        cfg['train_keys']=['shen_rtl_1_train'];validate_run_roles(cfg,reg)
    def test_locked_and_aliased_receivers_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);base=root/'data/external/shen_receiver_agnostic';base.mkdir(parents=True)
            reg=dict(train_receivers=['rtl_1'],development_receivers=['rtl_4'])
            with self.assertRaises(AssertionError):checked_shen_path(root,dict(receiver='b200_2',relative='b200_2_test.h5'),reg)
            (base/'rtl_1_train.h5').symlink_to(base/'b200_2_test.h5')
            with self.assertRaises(AssertionError):checked_shen_path(root,dict(receiver='rtl_1',relative='rtl_1_train.h5'),reg)
            with self.assertRaises(AssertionError):checked_shen_path(root,dict(receiver='rtl_1',relative='../rtl_1_train.h5'),reg)

    def test_scaler_is_weighted_by_source_only_sample_count(self):
        with tempfile.TemporaryDirectory() as d:
            for key,n,total,squared in [('sourceA',2,2.,2.),('sourceB',1,4.,16.),('target',100,1e9,1e18)]:
                folder=cache(d,key);folder.mkdir(parents=True)
                (folder/'manifest.json').write_text(json.dumps({'stats':{'full_cis':{'count':n,'sum':total,'sum_squares':squared}}}))
            fitted=fit_scaler(d,['sourceA','sourceB'],'full_cis')
            self.assertEqual(fitted['mean'],2.)
            self.assertAlmostEqual(fitted['std'],np.sqrt(2.))
            self.assertEqual(fitted['fit_source_keys'],['sourceA','sourceB'])

    def test_notch_removes_only_registered_information(self):
        reg=dict(notch=dict(negative_low_hz=-109375.,negative_high_hz=-93750.,random_low_hz=-85937.5,random_high_hz=-70312.5))
        negative=removed_mask('notch_negative',reg);positive=removed_mask('notch_positive',reg)
        self.assertEqual(int(negative.sum()),128);np.testing.assert_array_equal(positive,negative[(-np.arange(8192))%8192])
        rng=np.random.default_rng(1);iq=rng.normal(size=(2,8192))+1j*rng.normal(size=(2,8192))
        a=project(iq,'notch_negative',reg)
        self.assertLess(float(np.max(np.abs(np.fft.fft(a,axis=-1)[:,negative]))),1e-10)
        np.testing.assert_allclose(np.fft.fft(a,axis=-1)[:,~negative],np.fft.fft(iq,axis=-1)[:,~negative],rtol=1e-12,atol=1e-10)

if __name__=='__main__':unittest.main()
