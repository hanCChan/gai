"""Scientific validity checks for oracle scope and leakage-sensitive features."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from eisr.diagnostics import convex_oracle, observables, receiver_splits


class EISRTests(unittest.TestCase):
    def test_soft_oracle_can_rescue_both_wrong(self):
        s=np.array([[.4,.5,.1],[.2,.7,.1]])
        f=np.array([[.4,.1,.5],[.2,.1,.7]])
        got=convex_oracle(s,f,[0,0])
        self.assertEqual(got['feasible'].tolist(), [True,False])
        self.assertTrue((s.argmax(1) != 0).all() and (f.argmax(1) != 0).all())

    def test_narrow_interval_not_captured_by_coarse_grid(self):
        c=1/3+1e-10
        s=np.array([[c,c+.0123456,1-2*c-.0123456]])
        f=np.array([[c,c-.0876544,1-2*c+.0876544]])
        got=convex_oracle(s,f,[0])
        self.assertTrue(got['feasible'][0])
        self.assertLess(got['upper'][0]-got['lower'][0], 1e-7)
        self.assertEqual(((1-got['witness'][0])*s+got['witness'][0]*f).argmax(1)[0],0)
        self.assertFalse(any(((1-g)*s+g*f).argmax(1)[0] == 0 for g in np.linspace(0,1,21)))

    def test_endpoint_and_argmax_tie_rules(self):
        s=np.array([[.5,.5],[.5,.5],[.6,.4]])
        f=np.array([[.4,.6],[.5,.5],[.4,.6]])
        result=convex_oracle(s,f,[0,1,0])
        self.assertEqual(result['feasible'].tolist(),[True,False,True])
        self.assertEqual(result['lower'][0],0)
        self.assertEqual(result['upper'][0],0)

    def test_features_invariant_to_common_class_renaming(self):
        rng=np.random.default_rng(20261012)
        s=rng.dirichlet(np.ones(10),20);f=rng.dirichlet(np.ones(10),20)
        perm=rng.permutation(10)
        np.testing.assert_allclose(observables(s,f),observables(s[:,perm],f[:,perm]),atol=1e-14)

    def test_second_level_never_crosses_expert_groups(self):
        rx=['a','b','c','d','e']
        splits=receiver_splits(rx)
        self.assertEqual(len(splits),5)
        for test,train in splits:
            self.assertNotIn(test,train)
            self.assertEqual(set(train)|{test},set(rx))
            self.assertEqual(len(train),4)


if __name__ == '__main__':unittest.main()
