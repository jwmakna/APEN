import unittest

import numpy as np

import operational_evaluation as oe


class TestImbalanceAndProfit(unittest.TestCase):
    def test_surplus_shortage_hand_computation(self):
        S = np.array([0.8, 0.2, 0.5])
        q = np.array([0.5, 0.6, 0.5])
        surplus, shortage = oe.imbalance_components(S, q)
        np.testing.assert_allclose(surplus, [0.3, 0.0, 0.0])
        np.testing.assert_allclose(shortage, [0.0, 0.4, 0.0])
        self.assertFalse(np.any((surplus > 0) & (shortage > 0)))

    def test_profit_hand_computation(self):
        S = np.array([0.8]); q = np.array([0.5])
        DP = np.array([40.0]); RP = np.array([30.0])
        expected = 30.0 * (40.0 * 0.5 + 30.0 * 0.3)
        self.assertAlmostEqual(oe.profit_per_observation(S, q, DP, RP)[0], expected)

    def test_negative_da_is_not_modified(self):
        S = np.array([0.0]); q = np.array([1.0])
        DP = np.array([-20.0]); RP = np.array([5.0])
        # PC=-10 exactly follows 0.5*DP; no abs/clipping is applied.
        self.assertAlmostEqual(oe.profit_per_observation(S, q, DP, RP)[0], -300.0)


class TestOracle(unittest.TestCase):
    def test_candidates_match_dense_grid(self):
        S = np.array([0.37, 0.62, 0.2])
        DP = np.array([40.0, 20.0, -5.0])
        RP = np.array([70.0, 10.0, 30.0])
        oracle, oracle_q = oe.oracle_profit_per_observation(S, DP, RP)
        grid = np.linspace(0, 1, 10001)
        for i in range(len(S)):
            brute = oe.profit_per_observation(
                np.full_like(grid, S[i]), grid,
                np.full_like(grid, DP[i]), np.full_like(grid, RP[i]),
            )
            self.assertAlmostEqual(oracle[i], float(brute.max()), places=8)
            self.assertIn(round(float(oracle_q[i]), 12), {0.0, round(float(S[i]), 12), 1.0})

    def test_oracle_loss_nonnegative(self):
        S = np.array([0.2, 0.5, 0.8]); q = np.array([0.7, 0.1, 0.8])
        DP = np.array([40.0, 50.0, 30.0]); RP = np.array([35.0, 70.0, 10.0])
        oracle, _ = oe.oracle_profit_per_observation(S, DP, RP)
        realized = oe.profit_per_observation(S, q, DP, RP)
        self.assertTrue(np.all(oracle - realized >= -1e-8))

    def test_nonpositive_oracle_denominator_blocked(self):
        S = np.array([0.0]); q = np.array([0.0])
        DP = np.array([0.0]); RP = np.array([0.0])
        with self.assertRaises(oe.EvaluationError):
            oe.normalized_economic_loss(S, q, DP, RP)


class TestScaleInvariance(unittest.TestCase):
    def test_price_scaling_invariance(self):
        S=np.array([0.2,0.7]); q=np.array([0.5,0.4]); DP=np.array([40.,50.]); RP=np.array([30.,70.])
        a=oe.normalized_economic_loss(S,q,DP,RP)
        b=oe.normalized_economic_loss(S,q,DP*100,RP*100)
        self.assertAlmostEqual(a,b,places=12)

    def test_capacity_scaling_invariance(self):
        S=np.array([0.2,0.7]); q=np.array([0.5,0.4]); DP=np.array([40.,50.]); RP=np.array([30.,70.])
        a=oe.normalized_economic_loss(S,q,DP,RP,capacity_mw=30)
        b=oe.normalized_economic_loss(S,q,DP,RP,capacity_mw=300)
        self.assertAlmostEqual(a,b,places=12)


class TestProjectionAndMetrics(unittest.TestCase):
    def test_projection_and_raw_immutability(self):
        raw=np.array([-0.2,0.4,1.3]); before=raw.copy()
        np.testing.assert_allclose(oe.feasible_unit_projection(raw),[0,0.4,1])
        np.testing.assert_array_equal(raw,before)

    def test_official_projection_is_exact_not_tolerance_relaxed(self):
        with self.assertRaises(oe.EvaluationError):
            oe.validate_official_prediction(np.array([1.0+5e-9]))

    def test_diagnostics(self):
        d=oe.prediction_diagnostics(np.array([-0.2,0.4,1.3]))
        self.assertEqual(d.raw_boundary_violation_count,2)
        self.assertEqual(d.projection_changed_count,2)
        self.assertAlmostEqual(d.projection_max_absolute_change,0.3)

    def test_rmse_nrmse_hand_computation(self):
        S=np.array([0.4,0.4]); q=np.array([0.5,0.3])
        self.assertAlmostEqual(oe.rmse(S,q),0.1)
        self.assertAlmostEqual(oe.nrmse_percent(S,q),25.0)

    def test_shape_nan_infinity_empty_rejected(self):
        bad=[
            (np.array([0.1,0.2]),np.array([0.1])),
            (np.array([np.nan]),np.array([0.1])),
            (np.array([0.1]),np.array([np.inf])),
            (np.array([]),np.array([])),
        ]
        for S,q in bad:
            with self.subTest(S=S,q=q):
                with self.assertRaises(oe.EvaluationError):
                    oe.rmse(S,q)


class TestImportHasNoExecution(unittest.TestCase):
    def test_no_io_or_solver_calls(self):
        import inspect
        source=inspect.getsource(oe)
        for marker in ("read_csv(","linprog(","milp(",".fit("):
            self.assertNotIn(marker,source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
