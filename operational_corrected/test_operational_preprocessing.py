import dataclasses
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import operational_config as cfg
from operational_preprocessing import (
    PreprocessingError,
    ScalerMetadata,
    build_mlr_design,
    build_operational_dataset,
    deaccumulate_source,
    fit_train_scaler,
    validate_source_sequence,
)


ROOT = Path(__file__).resolve().parents[1]
PREDICTORS = ROOT / "data" / "predictors15.csv"
DA = ROOT / "data" / "da_lmp_prices.csv"
RT = ROOT / "data" / "rt_lmp_prices.csv"


def synthetic_group(zone: int, run_date: str, ssrd, tsr, power=None):
    start = pd.Timestamp(run_date) + pd.Timedelta(hours=1)
    timestamps = pd.date_range(start, periods=24, freq="h")
    if power is None:
        power = np.linspace(0, 1, 24)
    return pd.DataFrame({
        "ZONEID": zone,
        "TIMESTAMP": timestamps,
        "VAR169": np.asarray(ssrd, dtype=float),
        "VAR178": np.asarray(tsr, dtype=float),
        "POWER": np.asarray(power, dtype=float),
    })


class TestSourceSequenceAndDeaccumulation(unittest.TestCase):
    def test_simple_accumulation(self):
        ssrd = [1, 3, 6] + [6] * 21
        tsr = [2, 5, 9] + [9] * 21
        raw = synthetic_group(1, "2014-01-01", ssrd, tsr)
        out, _ = deaccumulate_source(raw)
        np.testing.assert_allclose(out["dSSRD"].iloc[:3], [1, 2, 3])
        np.testing.assert_allclose(out["dTSR"].iloc[:3], [2, 3, 4])

    def test_reset_between_runs_is_not_differenced_across_runs(self):
        a = synthetic_group(1, "2014-01-01", np.arange(1, 25), np.arange(2, 50, 2))
        b = synthetic_group(1, "2014-01-02", np.arange(10, 34), np.arange(20, 68, 2))
        out, _ = deaccumulate_source(pd.concat([a, b], ignore_index=True))
        self.assertEqual(out.loc[out["forecast_step"] == 1, "dSSRD"].tolist(), [1.0, 10.0])
        self.assertEqual(out.loc[out["forecast_step"] == 1, "dTSR"].tolist(), [2.0, 20.0])

    def test_groups_do_not_mix_zones(self):
        a = synthetic_group(1, "2014-01-01", np.arange(1, 25), np.arange(2, 50, 2))
        b = synthetic_group(2, "2014-01-01", np.arange(100, 124), np.arange(200, 248, 2))
        out, _ = deaccumulate_source(pd.concat([a, b], ignore_index=True))
        first = out[out["forecast_step"] == 1].sort_values("ZONEID")
        np.testing.assert_allclose(first["dSSRD"], [1, 100])

    def test_negative_within_group_difference_is_preserved(self):
        vals = [1, 3, 2.999] + [3] * 21
        raw = synthetic_group(1, "2014-01-01", vals, vals)
        out, _ = deaccumulate_source(raw)
        self.assertAlmostEqual(out["dSSRD"].iloc[2], -0.001)

    def test_non_24_row_group_rejected(self):
        raw = synthetic_group(1, "2014-01-01", np.arange(1, 25), np.arange(1, 25)).iloc[:-1]
        with self.assertRaises(PreprocessingError):
            validate_source_sequence(raw)

    def test_broken_hourly_sequence_rejected(self):
        raw = synthetic_group(1, "2014-01-01", np.arange(1, 25), np.arange(1, 25))
        raw.loc[5, "TIMESTAMP"] += pd.Timedelta(hours=1)
        with self.assertRaises(PreprocessingError):
            validate_source_sequence(raw)


class TestScalingAndMLRDesign(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            "dSSRD": [1.0, 2.0, 3.0, 4.0],
            "dTSR": [10.0, 20.0, 30.0, 40.0],
            "hour_idx": [0, 1, 2, 3],
        })

    def test_training_only_scaler(self):
        scaler = fit_train_scaler(self.train)
        changed_test = pd.DataFrame({"dSSRD": [1000.0], "dTSR": [2000.0], "hour_idx": [0]})
        X, _ = build_mlr_design(changed_test, scaler)
        self.assertAlmostEqual(X[0, 1], (1000.0 - 2.5) / np.std([1, 2, 3, 4]))
        self.assertAlmostEqual(scaler.dSSRD_mean, 2.5)

    def test_hour_reference_and_eleven_dummies(self):
        scaler = fit_train_scaler(self.train)
        rows = pd.DataFrame({
            "dSSRD": np.repeat(2.5, 12),
            "dTSR": np.repeat(25.0, 12),
            "hour_idx": np.arange(12),
        })
        X, names = build_mlr_design(rows, scaler)
        self.assertEqual(X.shape, (12, 14))
        self.assertEqual(len([n for n in names if n.startswith("hour_")]), 11)
        np.testing.assert_array_equal(X[0, 3:], np.zeros(11))
        np.testing.assert_array_equal(X[1:, 3:], np.eye(11))

    def test_zero_training_std_rejected(self):
        bad = self.train.copy()
        bad["dSSRD"] = 1.0
        with self.assertRaises(PreprocessingError):
            fit_train_scaler(bad)

    def test_dummy_columns_are_not_standardized(self):
        scaler = fit_train_scaler(self.train)
        X, _ = build_mlr_design(self.train, scaler)
        self.assertTrue(set(np.unique(X[:, 3:])).issubset({0.0, 1.0}))


class TestStrictDatasetPolicy(unittest.TestCase):
    def test_exact_policy_passes(self):
        cfg.validate_official_dataset_policy(cfg.OFFICIAL_DATASET_POLICY)

    def test_integer_field_float_substitution_rejected(self):
        bad = dataclasses.replace(cfg.OFFICIAL_DATASET_POLICY, train_days=300.0)
        with self.assertRaises(TypeError):
            cfg.validate_official_dataset_policy(bad)

    def test_boolean_field_integer_substitution_rejected(self):
        bad = dataclasses.replace(cfg.OFFICIAL_DATASET_POLICY, requires_observed_rt=1)
        with self.assertRaises(TypeError):
            cfg.validate_official_dataset_policy(bad)

    def test_string_field_non_string_rejected(self):
        bad = dataclasses.replace(cfg.OFFICIAL_DATASET_POLICY, train_start=20130327)
        with self.assertRaises(TypeError):
            cfg.validate_official_dataset_policy(bad)


class TestOfficialDatasetIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = build_operational_dataset(PREDICTORS, DA, RT)

    def test_fixed_300_100_policy(self):
        self.assertEqual((self.dataset.train["local_date"].nunique(), len(self.dataset.train)), (300, 3600))
        self.assertEqual((self.dataset.test["local_date"].nunique(), len(self.dataset.test)), (100, 1200))

    def test_extended_price_files_are_complete_hourly_series(self):
        expected = pd.date_range("2013-03-26 00:00:00", "2014-04-30 23:00:00", freq="h")
        for path, value_column in ((DA, "DA_LMP"), (RT, "RT_LMP")):
            prices = pd.read_csv(path)
            timestamps = pd.to_datetime(prices["TIMESTAMP"], errors="raise")
            self.assertEqual(len(prices), 9624)
            np.testing.assert_array_equal(timestamps.to_numpy(), expected.to_numpy())
            self.assertFalse(timestamps.duplicated().any())
            self.assertTrue(np.isfinite(prices[value_column].to_numpy(float)).all())

    def test_every_day_has_twelve_rows(self):
        self.assertTrue((self.dataset.train.groupby("local_date").size() == 12).all())
        self.assertTrue((self.dataset.test.groupby("local_date").size() == 12).all())

    def test_local_hours_and_hour_idx_are_exact_every_day(self):
        for frame in (self.dataset.train, self.dataset.test):
            for _, day in frame.groupby("local_date", sort=False):
                np.testing.assert_array_equal(day["local_hour"], np.arange(9, 21))
                np.testing.assert_array_equal(day["hour_idx"], np.arange(12))

    def test_day_grouping_is_sydney_local_date_not_utc_date(self):
        first = self.dataset.train.iloc[0]
        self.assertEqual(first["timestamp"].isoformat(), "2013-03-26T22:00:00+00:00")
        self.assertEqual(first["timestamp_local"].isoformat(), "2013-03-27T09:00:00+11:00")
        self.assertNotEqual(first["timestamp"].date(), first["local_date"])

    def test_dst_changes_utc_mapping_automatically(self):
        before = self.dataset.alignment_audit["dst_before_example"]
        after = self.dataset.alignment_audit["dst_after_example"]
        self.assertEqual(before["timestamp_local"], "2014-04-05T09:00:00+11:00")
        self.assertEqual(before["timestamp_utc"], "2014-04-04T22:00:00+00:00")
        self.assertEqual(after["timestamp_local"], "2014-04-07T09:00:00+10:00")
        self.assertEqual(after["timestamp_utc"], "2014-04-06T23:00:00+00:00")

    def test_boundary_dates_follow_observed_support(self):
        audit = self.dataset.alignment_audit
        self.assertEqual(audit["history_local_date"], "2013-03-26")
        self.assertEqual(audit["history_source_daylight_rows"], 12)
        self.assertEqual(audit["history_complete_four_source_rows"], 10)
        self.assertNotIn(pd.Timestamp("2013-03-26").date(), set(self.dataset.train["local_date"]))
        self.assertEqual(audit["test_end_local_date"], "2014-04-30")
        self.assertEqual(audit["test_end_observed_rt_supported_rows"], 12)
        self.assertIn(pd.Timestamp("2014-04-30").date(), set(self.dataset.test["local_date"]))

    def test_power_and_prices_are_finite_and_rt_observed(self):
        for frame in (self.dataset.train, self.dataset.test):
            values = frame[["solar_power", "da_price", "rt_price"]].to_numpy(float)
            self.assertTrue(np.isfinite(values).all())

    def test_power_and_price_values_and_order_equal_direct_sources(self):
        official = pd.concat([self.dataset.train, self.dataset.test], ignore_index=True)
        raw = pd.read_csv(PREDICTORS)
        raw["TIMESTAMP"] = pd.to_datetime(
            raw["TIMESTAMP"], format="%Y%m%d %H:%M"
        ).dt.tz_localize("UTC")
        raw["local_ts"] = raw["TIMESTAMP"].dt.tz_convert(cfg.LOCAL_TIMEZONE)
        raw["local_date"] = raw["local_ts"].dt.date
        raw["local_hour"] = raw["local_ts"].dt.hour
        train_start = pd.Timestamp(cfg.TRAIN_START).date()
        test_end = pd.Timestamp(cfg.TEST_END).date()
        direct = raw[
            (raw["ZONEID"] == cfg.ZONE_ID)
            & raw["local_date"].between(train_start, test_end)
            & raw["local_hour"].between(cfg.LOCAL_HOUR_START, cfg.LOCAL_HOUR_END - 1)
        ].sort_values("TIMESTAMP")
        da = pd.read_csv(DA); rt = pd.read_csv(RT)
        da["TIMESTAMP"] = pd.to_datetime(da["TIMESTAMP"]).dt.tz_localize("UTC")
        rt["TIMESTAMP"] = pd.to_datetime(rt["TIMESTAMP"]).dt.tz_localize("UTC")
        direct = direct.merge(da, on="TIMESTAMP", validate="one_to_one")
        direct = direct.merge(rt, on="TIMESTAMP", validate="one_to_one")
        np.testing.assert_array_equal(official["timestamp"].to_numpy(), direct["TIMESTAMP"].to_numpy())
        np.testing.assert_array_equal(official["solar_power"].to_numpy(), direct["POWER"].to_numpy())
        np.testing.assert_array_equal(official["da_price"].to_numpy(), direct["DA_LMP"].to_numpy())
        np.testing.assert_array_equal(official["rt_price"].to_numpy(), direct["RT_LMP"].to_numpy())

    def test_deaccumulation_precedes_local_filter(self):
        raw = pd.read_csv(PREDICTORS)
        raw["TIMESTAMP"] = pd.to_datetime(raw["TIMESTAMP"], format="%Y%m%d %H:%M").dt.tz_localize("UTC")
        zone = raw[raw["ZONEID"].eq(cfg.ZONE_ID)].sort_values("TIMESTAMP").reset_index(drop=True)
        out, _ = deaccumulate_source(zone)
        target_utc = pd.Timestamp("2013-03-26 22:00:00", tz="UTC")
        position = int(out.index[out["TIMESTAMP"].eq(target_utc)][0])
        expected = zone.loc[position, "VAR169"] - zone.loc[position - 1, "VAR169"]
        self.assertAlmostEqual(out.loc[position, "dSSRD"], expected)
        selected = self.dataset.train.iloc[0]
        self.assertEqual(selected["timestamp"], target_utc)
        self.assertAlmostEqual(selected["dSSRD"], expected)

    def test_utc_00_to_11_window_is_not_mixed_into_local_window(self):
        before_dst = self.dataset.train[
            self.dataset.train["local_date"].eq(pd.Timestamp("2013-03-27").date())
        ]
        after_dst = self.dataset.test[
            self.dataset.test["local_date"].eq(pd.Timestamp("2014-04-07").date())
        ]
        np.testing.assert_array_equal(before_dst["timestamp"].dt.hour, [22,23,0,1,2,3,4,5,6,7,8,9])
        np.testing.assert_array_equal(after_dst["timestamp"].dt.hour, [23,0,1,2,3,4,5,6,7,8,9,10])
        self.assertFalse(np.array_equal(before_dst["timestamp"].dt.hour, np.arange(12)))

    def test_first_daylight_slot_not_forced_to_zero(self):
        first = self.dataset.train[self.dataset.train["hour_idx"] == 0]
        self.assertTrue((np.abs(first["dSSRD"]) > 0).any())
        self.assertTrue((np.abs(first["dTSR"]) > 0).any())

    def test_source_group_audit_is_24_step(self):
        self.assertEqual(self.dataset.source_sequence_audit["group_status"],
                         "STRUCTURALLY_IDENTIFIED_FROM_OFFICIAL_SOURCE_SEQUENCE")
        self.assertTrue(all(z["group_size"] == 24 for z in self.dataset.source_sequence_audit["zones"].values()))


class TestImportHasNoExecution(unittest.TestCase):
    def test_import_does_not_load_data_or_solve(self):
        import inspect
        import operational_preprocessing as op
        source = inspect.getsource(op)
        self.assertNotIn("build_operational_dataset(PREDICTORS", source)
        self.assertNotIn("milp(", source)
        self.assertNotIn("linprog(", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
