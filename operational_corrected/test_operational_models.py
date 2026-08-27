import unittest

import numpy as np

import operational_models as om
from operational_evaluation import normalized_economic_loss


class TestBoundedLAD(unittest.TestCase):
    def test_hand_computed_constant_median(self):
        X=np.ones((3,1)); y=np.array([0.1,0.4,0.9])
        fit=om.fit_bounded_lad(X,y)
        np.testing.assert_allclose(fit.fitted_raw,np.repeat(0.4,3),atol=1e-8)
        self.assertTrue(fit.certified_optimal)

    def test_training_fitted_bounds(self):
        X=np.column_stack([np.ones(5),np.arange(5)])
        y=np.array([0,0.2,0.4,0.8,1.0])
        fit=om.fit_bounded_lad(X,y)
        self.assertTrue(np.all(fit.fitted_raw>=-1e-7))
        self.assertTrue(np.all(fit.fitted_raw<=1+1e-7))


class TestARDesignAndRolling(unittest.TestCase):
    def test_previous_day_block_reversed_hand_example(self):
        history=np.arange(12,dtype=float)[None,:]
        train=np.vstack([np.arange(10,22),np.arange(20,32)]).astype(float)
        X,y=om.build_ar_lag_design(history,train)
        np.testing.assert_array_equal(X[0],[1]+list(range(11,-1,-1)))
        np.testing.assert_array_equal(X[1],[1]+list(range(21,9,-1)))
        np.testing.assert_array_equal(y,train)

    def test_lags_are_not_averaged_or_summed(self):
        history=np.arange(12,dtype=float)[None,:]
        train=np.arange(12,24,dtype=float)[None,:]
        X,_=om.build_ar_lag_design(history,train)
        self.assertEqual(X.shape[1],13)
        self.assertEqual(len(np.unique(X[0,1:])),12)

    def test_rolling_has_no_future_leakage(self):
        coef=np.zeros((12,13)); coef[:,1]=1.0  # previous day's slot 11
        previous=np.arange(12,dtype=float)
        actual=np.vstack([np.arange(100,112),np.arange(200,212)]).astype(float)
        raw1=om.predict_ar_rolling(coef,previous,actual)
        changed=actual.copy(); changed[1]+=9999
        raw2=om.predict_ar_rolling(coef,previous,changed)
        np.testing.assert_array_equal(raw1[0],raw2[0])
        np.testing.assert_array_equal(raw1[1],raw2[1])
        np.testing.assert_array_equal(raw1[0],np.repeat(11.0,12))
        np.testing.assert_array_equal(raw1[1],np.repeat(111.0,12))


class TestProposedFormulation(unittest.TestCase):
    def test_w1_zero_exactly_matches_conventional(self):
        X=np.column_stack([np.ones(5),np.linspace(-1,1,5)])
        y=np.array([0.1,0.2,0.5,0.7,0.9]); DP=np.repeat(40.,5); RP=np.repeat(30.,5)
        conventional=om.fit_bounded_lad(X,y)
        proposed=om.fit_proposed_milp(X,y,DP,RP,w1=0,w2=20)
        np.testing.assert_allclose(proposed.fitted_raw,conventional.fitted_raw,rtol=0,atol=1e-10)
        np.testing.assert_allclose(proposed.coefficients,conventional.coefficients,rtol=0,atol=1e-10)

    def test_exact_milp_matches_dense_grid_constant_model(self):
        X=np.ones((2,1)); y=np.array([0.2,0.8]); DP=np.array([40.,50.]); RP=np.array([70.,20.])
        fit=om.fit_proposed_milp(X,y,DP,RP,w1=1,w2=2)
        grid=np.linspace(0,1,10001)
        brute=[]
        for q in grid:
            econ=normalized_economic_loss(y,np.repeat(q,2),DP,RP)
            brute.append(econ+2*np.mean(np.abs(y-q)))
        self.assertLessEqual(abs(fit.objective_value-min(brute)),3e-4)
        self.assertTrue(fit.certified_optimal)
        self.assertEqual(fit.formulation, "exact_sign_reduced_complementarity_milp")
        self.assertLess(fit.binary_variable_count, fit.unreduced_binary_variable_count)

    def test_negative_common_delta_cost_keeps_required_binaries(self):
        X=np.ones((2,1)); y=np.array([0.2,0.8])
        DP=np.ones(2); RP=np.full(2,100.0)
        fit=om.fit_proposed_milp(X,y,DP,RP,w1=1,w2=0.001)
        self.assertEqual(fit.binary_variable_count,2)
        self.assertEqual(fit.unreduced_binary_variable_count,2)
        self.assertTrue(fit.certified_optimal)

    def test_invalid_prices_shape_rejected(self):
        with self.assertRaises(om.ModelError):
            om.fit_proposed_milp(np.ones((2,1)),np.array([0.2,0.8]),np.array([40.]),np.array([30.,30.]))

    def test_w1_zero_ar_matches_conventional_ar(self):
        history=np.linspace(0.0,0.55,12)[None,:]
        train=np.vstack([
            np.linspace(0.05,0.60,12),
            np.linspace(0.10,0.65,12),
            np.linspace(0.15,0.70,12),
        ])
        prices=np.full_like(train,40.0)
        conventional=om.fit_conventional_ar(history,train)
        proposed=om.fit_proposed_ar(history,train,prices,np.full_like(train,30.0),w1=0,w2=20)
        np.testing.assert_allclose(proposed.fitted_raw,conventional.fitted_raw,rtol=0,atol=1e-10)
        np.testing.assert_allclose(proposed.coefficients_by_hour,conventional.coefficients_by_hour,rtol=0,atol=1e-10)

    def test_w1_zero_mlr_matches_conventional_mlr(self):
        hours=np.arange(12)
        X=np.column_stack([
            np.ones(12), np.linspace(-1,1,12), np.linspace(1,-1,12),
            np.column_stack([(hours==h).astype(float) for h in range(1,12)]),
        ])
        y=np.linspace(0.05,0.80,12)
        conventional=om.fit_conventional_mlr(X,y)
        proposed=om.fit_proposed_mlr(X,y,np.full(12,40.0),np.full(12,30.0),w1=0,w2=20)
        np.testing.assert_allclose(proposed.fitted_raw,conventional.fitted_raw,rtol=0,atol=1e-10)
        np.testing.assert_allclose(proposed.coefficients,conventional.coefficients,rtol=0,atol=1e-10)


class TestMLRInterface(unittest.TestCase):
    def test_pooled_fit_has_one_coefficient_vector(self):
        X=np.column_stack([np.ones(12),np.linspace(-1,1,12),np.linspace(1,-1,12),np.eye(12)[:,1:]])
        y=np.linspace(0,1,12)
        fit=om.fit_conventional_mlr(X,y)
        self.assertEqual(fit.coefficients.shape,(14,))
        self.assertEqual(fit.fitted_raw.shape,(12,))


class TestInputGuards(unittest.TestCase):
    def test_nan_shape_empty_rejected(self):
        cases=[
            (np.empty((0,1)),np.array([])),
            (np.ones((2,1)),np.array([0.1])),
            (np.array([[np.nan]]),np.array([0.1])),
        ]
        for X,y in cases:
            with self.subTest(X=X,y=y):
                with self.assertRaises(om.ModelError):
                    om.fit_bounded_lad(X,y)

    def test_import_has_no_data_or_solver_execution(self):
        import inspect
        source=inspect.getsource(om)
        self.assertNotIn("read_csv(",source)
        self.assertNotIn("build_operational_dataset(",source)


if __name__=="__main__":
    unittest.main(verbosity=2)
