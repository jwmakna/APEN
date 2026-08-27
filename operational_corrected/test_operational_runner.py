import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import operational_config as cfg
import run_operational_corrected as runner
from operational_evaluation import evaluate_raw_predictions


def canonical_time_columns():
    local = pd.DatetimeIndex([
        pd.Timestamp(date).tz_localize("Australia/Sydney") + pd.Timedelta(hours=hour)
        for date in pd.date_range(cfg.TEST_START, cfg.TEST_END, freq="D")
        for hour in range(9, 21)
    ])
    return {
        "timestamp_utc": local.tz_convert("UTC").map(lambda value: value.isoformat()),
        "timestamp_local": local.map(lambda value: value.isoformat()),
        "local_date": [value.date().isoformat() for value in local],
        "local_hour": np.tile(np.arange(9, 21), cfg.TEST_DAYS),
        "hour_idx": np.tile(np.arange(12), cfg.TEST_DAYS),
    }


class TestIndependentMetricRecalculation(unittest.TestCase):
    def test_saved_prediction_equations_match_evaluator(self):
        rows = []
        summaries = {}
        S = np.linspace(0.05, 0.95, cfg.TEST_OBSERVATIONS)
        DP = np.linspace(20.0, 60.0, cfg.TEST_OBSERVATIONS)
        RP = np.linspace(70.0, 10.0, cfg.TEST_OBSERVATIONS)
        time_columns = canonical_time_columns()
        for i, model in enumerate(runner.MODEL_ORDER):
            raw = S + (i - 1.5) * 0.1
            summary, arrays = evaluate_raw_predictions(S, raw, DP, RP)
            summaries[model] = summary
            frame = pd.DataFrame({
                "model": model,
                **time_columns,
                "actual": arrays["actual"], "raw": arrays["raw"],
                "projected": arrays["projected"], "DA": arrays["da"],
                "RT": arrays["rt"], "penalty": arrays["penalty"],
                "realized_profit": arrays["realized_profit"],
                "oracle_profit": arrays["oracle_profit"], "oracle_q": arrays["oracle_q"],
            })
            rows.append(frame)
        predictions = pd.concat(rows, ignore_index=True)
        recomputed = runner.independently_recompute_metrics(predictions)
        verification = runner._assert_metric_agreement(summaries, recomputed)
        self.assertEqual(verification["status"], "VERIFIED")
        self.assertGreater(verification["checked_field_count"], 0)

    def test_modified_saved_projection_is_rejected_without_tolerance(self):
        S = np.repeat(0.4, cfg.TEST_OBSERVATIONS)
        raw = np.repeat(0.4, cfg.TEST_OBSERVATIONS)
        DP = np.repeat(40.0, cfg.TEST_OBSERVATIONS)
        RP = np.repeat(30.0, cfg.TEST_OBSERVATIONS)
        _, arrays = evaluate_raw_predictions(S, raw, DP, RP)
        time_columns = canonical_time_columns()
        rows = []
        for model in runner.MODEL_ORDER:
            frame = pd.DataFrame({
                "model": model,
                **time_columns,
                "actual": arrays["actual"], "raw": arrays["raw"],
                "projected": arrays["projected"], "DA": arrays["da"],
                "RT": arrays["rt"], "penalty": arrays["penalty"],
                "realized_profit": arrays["realized_profit"],
                "oracle_profit": arrays["oracle_profit"], "oracle_q": arrays["oracle_q"],
            })
            rows.append(frame)
        predictions = pd.concat(rows, ignore_index=True)
        predictions.loc[0, "projected"] += 1e-12
        with self.assertRaises(runner.OfficialRunError):
            runner.independently_recompute_metrics(predictions)

    def test_utc_window_rows_are_rejected_by_saved_prediction_check(self):
        S = np.repeat(0.4, cfg.TEST_OBSERVATIONS)
        raw = np.repeat(0.4, cfg.TEST_OBSERVATIONS)
        DP = np.repeat(40.0, cfg.TEST_OBSERVATIONS)
        RP = np.repeat(30.0, cfg.TEST_OBSERVATIONS)
        _, arrays = evaluate_raw_predictions(S, raw, DP, RP)
        time_columns = canonical_time_columns()
        rows = []
        for model in runner.MODEL_ORDER:
            frame = pd.DataFrame({
                "model": model, **time_columns,
                "actual": arrays["actual"], "raw": arrays["raw"],
                "projected": arrays["projected"], "DA": arrays["da"],
                "RT": arrays["rt"], "penalty": arrays["penalty"],
                "realized_profit": arrays["realized_profit"],
                "oracle_profit": arrays["oracle_profit"], "oracle_q": arrays["oracle_q"],
            })
            rows.append(frame)
        predictions = pd.concat(rows, ignore_index=True)
        predictions.loc[
            predictions["model"].eq(runner.MODEL_ORDER[0]), "local_hour"
        ] = np.tile(np.arange(12), cfg.TEST_DAYS)
        with self.assertRaises(runner.OfficialRunError):
            runner.independently_recompute_metrics(predictions)


class TestOfficialCommitGuards(unittest.TestCase):
    def test_existing_official_file_blocks_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / runner.OFFICIAL_FILENAMES[0]).touch()
            with self.assertRaises(runner.OfficialRunError):
                runner._assert_output_targets_absent(path)

    def test_failure_payload_contains_no_numeric_result(self):
        payload = runner._failure_payload(RuntimeError("synthetic failure"))
        self.assertFalse(payload["official_result_created"])
        self.assertEqual(payload["status"], "FAILED")
        self.assertNotIn("metrics", payload)

    def test_import_has_no_data_loading_or_solver_execution(self):
        import importlib
        results_dir = Path(__file__).parent / "results"
        before = {
            path.name: (path.stat().st_size, path.stat().st_mtime_ns)
            for path in results_dir.glob("*")
        } if results_dir.exists() else {}
        importlib.reload(runner)
        after = {
            path.name: (path.stat().st_size, path.stat().st_mtime_ns)
            for path in results_dir.glob("*")
        } if results_dir.exists() else {}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main(verbosity=2)
