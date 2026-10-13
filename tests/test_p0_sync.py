import json
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from p0_sync.detector import analyze, coherence, distance, synthetic_chirps
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_p0_sync import checked_raw_path


class SignalContract(unittest.TestCase):
    def test_whitelist_rejects_target_test_day_and_alias_paths(self):
        reg={'source_receivers':['rtl_1'],'data_root':'/home/keyan/code-y/data/external/shen_receiver_agnostic'}
        good={'receiver':'rtl_1','role':'train','relative':'rtl_1_train.h5'}
        self.assertEqual(checked_raw_path(reg,good).name,'rtl_1_train.h5')
        for change in ({'receiver':'rtl_4'},{'relative':'rtl_1_test.h5'},{'role':'evaluation'},
                       {'relative':'Train/rtl_1_day1_wireless_train.h5'}, {'relative':'../rtl_1_train.h5'}):
            with self.assertRaises(ValueError):checked_raw_path(reg,{**good,**change})

    def test_periodic_frequency_alias_is_not_start_detection(self):
        rng = np.random.default_rng(1)
        a = synthetic_chirps(123, .0001, 200, rng)
        b = a * np.exp(2j*np.pi*np.arange(8192)/1024)
        va = np.angle(np.vdot(a[:-1024],a[1024:]))
        vb = np.angle(np.vdot(b[:-1024],b[1024:]))
        self.assertAlmostEqual(va, vb, places=10)
        self.assertGreater(coherence(a,1024), .999)

    def test_positive_and_negative_crop_shifts(self):
        cfg = json.loads((Path(__file__).resolve().parents[1]/'configs/p0_sync/registration.json').read_text())['detector']
        for shift in (-511, -8, 0, 8, 511):
            x=synthetic_chirps(shift,.003,40,np.random.default_rng(shift+1000))
            m,*_=analyze(x,cfg)
            self.assertLessEqual(distance(m['chirp_phase_modulo_samples'],shift,1024),8)
            self.assertTrue(m['reliable_symbol_phase'])
            self.assertFalse(m['packet_start_identified'])


if __name__=='__main__':
    unittest.main()
