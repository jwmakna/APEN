"""Run the four APEN models on the fixed 300-day/100-day split."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from operational_config import (
    BOUNDARY_TOL,
    CAPACITY_MW,
    DAYLIGHT_SLOTS_PER_DAY,
    DURATION_HOURS,
    LEGACY_OFFICIAL_EXECUTION_USE,
    LEGACY_STATUS,
    LEGACY_SUPERSEDED_BY,
    LOCAL_TIMEZONE,
    OBJECTIVE_DEFINITION,
    OFFICIAL_DATASET_POLICY,
    OFFICIAL_PREDICTION_VIEW,
    OFFICIAL_VERSION,
    PENALTY_RATE,
    RAW_PREDICTION_SOURCES,
    TEST_DAYS,
    TEST_OBSERVATIONS,
    TRAIN_DAYS,
    TRAIN_OBSERVATIONS,
    W1,
    W2,
    validate_official_dataset_policy,
)
from operational_evaluation import evaluate_raw_predictions
from operational_models import (
    ARFit,
    LinearFit,
    SolverFailure,
    fit_conventional_ar,
    fit_conventional_mlr,
    fit_proposed_ar,
    fit_proposed_mlr,
    predict_ar_rolling,
    predict_linear_raw,
)
from operational_preprocessing import (
    OperationalDataset,
    build_mlr_design,
    build_operational_dataset,
    fit_train_scaler,
    to_daily_matrix,
)


MODEL_ORDER = (
    "Conventional AR",
    "Conventional MLR",
    "Proposed AR",
    "Proposed MLR",
)
OFFICIAL_FILENAMES = (
    "operational_predictions.csv",
    "operational_results.json",
    "operational_metric_verification.json",
    "operational_preprocessed_train.csv",
    "operational_preprocessed_test.csv",
)


class OfficialRunError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(value, tmp, indent=2, sort_keys=True, allow_nan=False)
        tmp.write("\n")
        temporary = Path(tmp.name)
    os.replace(temporary, path)


def _assert_output_targets_absent(output_dir: Path) -> None:
    present = [name for name in OFFICIAL_FILENAMES if (output_dir / name).exists()]
    if present:
        raise OfficialRunError(
            "refusing to overwrite existing official artifacts: " + ", ".join(present)
        )


def _solver_record(fit: LinearFit | ARFit) -> dict:
    if isinstance(fit, LinearFit):
        if not fit.certified_optimal or fit.solver_status != 0:
            raise OfficialRunError("linear fit is not certified OPTIMAL")
        return {
            "status": "OPTIMAL",
            "status_code": fit.solver_status,
            "message": fit.solver_message,
            "formulation": fit.formulation,
            "objective_value": fit.objective_value,
            "mip_gap": fit.mip_gap,
            "binary_variable_count": fit.binary_variable_count,
            "unreduced_binary_variable_count": fit.unreduced_binary_variable_count,
        }
    if not fit.certified_optimal or any(status != 0 for status in fit.solver_statuses_by_hour):
        raise OfficialRunError("AR fit is not certified OPTIMAL for all 12 target hours")
    if len(fit.solver_statuses_by_hour) != DAYLIGHT_SLOTS_PER_DAY:
        raise OfficialRunError("AR solver record must contain 12 target-hour statuses")
    return {
        "status": "OPTIMAL",
        "status_codes_by_hour": list(fit.solver_statuses_by_hour),
        "messages_by_hour": list(fit.solver_messages_by_hour),
        "formulation": fit.formulation,
        "objective_values_by_hour": fit.objective_values_by_hour.tolist(),
        "mip_gaps_by_hour": list(fit.mip_gaps_by_hour),
        "binary_variable_counts_by_hour": list(fit.binary_variable_counts_by_hour),
        "unreduced_binary_variable_counts_by_hour": list(
            fit.unreduced_binary_variable_counts_by_hour
        ),
    }


def _prediction_frame(
    model: str,
    test: pd.DataFrame,
    raw: np.ndarray,
    summary: dict,
    arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    raw = np.asarray(raw, dtype=float).reshape(-1)
    if len(raw) != TEST_OBSERVATIONS or len(test) != TEST_OBSERVATIONS:
        raise OfficialRunError(
            f"official prediction must have exactly {TEST_OBSERVATIONS} rows"
        )
    if summary["raw_prediction_min"] > summary["raw_prediction_max"]:
        raise OfficialRunError("raw prediction min exceeds max")
    if not all(np.isfinite(summary[key]) for key in (
        "raw_prediction_min", "raw_prediction_max", "official_nrmse_percent",
        "raw_nrmse_percent_diagnostic", "absolute_gap", "realized_profit",
        "oracle_profit", "gap_percent",
    )):
        raise OfficialRunError("official summary contains non-finite numeric values")
    return pd.DataFrame({
        "model": model,
        "timestamp_utc": test["timestamp"].map(lambda value: value.isoformat()).to_numpy(),
        "timestamp_local": test["timestamp_local"].map(lambda value: value.isoformat()).to_numpy(),
        "local_date": test["local_date"].map(lambda value: value.isoformat()).to_numpy(),
        "local_hour": test["local_hour"].to_numpy(dtype=int),
        "hour_idx": test["hour_idx"].to_numpy(dtype=int),
        "actual": arrays["actual"],
        "raw": arrays["raw"],
        "projected": arrays["projected"],
        "DA": arrays["da"],
        "RT": arrays["rt"],
        "penalty": arrays["penalty"],
        "realized_profit": arrays["realized_profit"],
        "oracle_profit": arrays["oracle_profit"],
        "oracle_q": arrays["oracle_q"],
    })


def independently_recompute_metrics(predictions: pd.DataFrame) -> dict[str, dict]:
    """Recalculate saved metrics without using the evaluation module."""
    required = {
        "model", "timestamp_utc", "timestamp_local", "local_date", "local_hour",
        "hour_idx", "actual", "raw", "projected", "DA", "RT",
        "penalty", "realized_profit", "oracle_profit", "oracle_q",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise OfficialRunError(f"prediction CSV is missing columns: {missing}")
    if len(predictions) != len(MODEL_ORDER) * TEST_OBSERVATIONS:
        raise OfficialRunError(
            f"prediction CSV row count is not {len(MODEL_ORDER)}*{TEST_OBSERVATIONS}"
        )

    result: dict[str, dict] = {}
    for model in MODEL_ORDER:
        group = predictions[predictions["model"] == model].copy()
        if len(group) != TEST_OBSERVATIONS:
            raise OfficialRunError(
                f"{model} does not have exactly {TEST_OBSERVATIONS} predictions"
            )
        numeric_columns = sorted(
            required - {"model", "timestamp_utc", "timestamp_local", "local_date"}
        )
        values = group[numeric_columns].to_numpy(dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise OfficialRunError(f"{model} prediction data is empty or non-finite")
        if not np.all(
            group["local_hour"].to_numpy(int).reshape(-1, 12) == np.arange(9, 21)
        ):
            raise OfficialRunError(f"{model} saved rows do not use local hours 09,...,20")
        if not np.all(group["hour_idx"].to_numpy(int).reshape(-1, 12) == np.arange(12)):
            raise OfficialRunError(f"{model} saved hour_idx is not 0,...,11 per local day")

        S = group["actual"].to_numpy(float)
        raw = group["raw"].to_numpy(float)
        projected = group["projected"].to_numpy(float)
        DP = group["DA"].to_numpy(float)
        RP = group["RT"].to_numpy(float)
        exact_projection = np.minimum(1.0, np.maximum(0.0, raw))
        if not np.array_equal(projected, exact_projection):
            raise OfficialRunError(f"{model} saved projection is not the exact [0,1] projection")
        if np.any(projected < 0.0) or np.any(projected > 1.0):
            raise OfficialRunError(f"{model} official prediction is outside [0,1]")

        surplus = np.maximum(S - projected, 0.0)
        shortage = np.maximum(projected - S, 0.0)
        realized = CAPACITY_MW * DURATION_HOURS * (
            DP * projected + RP * surplus - PENALTY_RATE * DP * shortage
        )
        candidates = np.stack([np.zeros_like(S), S, np.ones_like(S)])
        candidate_profits = []
        for q in candidates:
            u = np.maximum(S - q, 0.0)
            v = np.maximum(q - S, 0.0)
            candidate_profits.append(CAPACITY_MW * DURATION_HOURS * (
                DP * q + RP * u - PENALTY_RATE * DP * v
            ))
        candidate_profits = np.stack(candidate_profits)
        oracle = candidate_profits.max(axis=0)
        oracle_q = np.take_along_axis(
            candidates, np.expand_dims(candidate_profits.argmax(axis=0), 0), axis=0
        )[0]
        loss = oracle - realized
        if np.any(loss < -BOUNDARY_TOL):
            raise OfficialRunError(f"{model} realized profit exceeds oracle")
        denominator = float(oracle.sum())
        mean_actual = float(S.mean())
        if denominator <= 0 or mean_actual <= 0:
            raise OfficialRunError(f"{model} has a nonpositive metric denominator")
        diff = np.abs(projected - raw)
        result[model] = {
            "official_nrmse_percent": 100.0 * float(np.sqrt(np.mean((projected - S) ** 2))) / mean_actual,
            "raw_nrmse_percent_diagnostic": 100.0 * float(np.sqrt(np.mean((raw - S) ** 2))) / mean_actual,
            "absolute_gap": float(loss.sum()),
            "realized_profit": float(realized.sum()),
            "oracle_profit": denominator,
            "gap_percent": 100.0 * float(loss.sum()) / denominator,
            "raw_prediction_min": float(raw.min()),
            "raw_prediction_max": float(raw.max()),
            "raw_boundary_violation_count": int(np.sum((raw < -BOUNDARY_TOL) | (raw > 1 + BOUNDARY_TOL))),
            "projection_changed_count": int(np.sum(diff > BOUNDARY_TOL)),
            "projection_max_absolute_change": float(diff.max()),
            "evaluated_prediction_min": float(projected.min()),
            "evaluated_prediction_max": float(projected.max()),
        }
        for column, computed in (
            ("realized_profit", realized), ("oracle_profit", oracle), ("oracle_q", oracle_q),
        ):
            if not np.allclose(group[column].to_numpy(float), computed, rtol=1e-12, atol=1e-8):
                raise OfficialRunError(f"{model} saved {column} disagrees with independent calculation")
    if tuple(result) != MODEL_ORDER:
        raise OfficialRunError("prediction CSV model set/order is invalid")
    return result


def _assert_metric_agreement(reported: dict[str, dict], recomputed: dict[str, dict]) -> dict:
    checked = []
    for model in MODEL_ORDER:
        for key, expected in reported[model].items():
            if key not in recomputed[model]:
                continue
            actual = recomputed[model][key]
            if isinstance(expected, int) and not isinstance(expected, bool):
                equal = actual == expected
            else:
                equal = np.isclose(float(actual), float(expected), rtol=1e-11, atol=1e-8)
            if not equal:
                raise OfficialRunError(
                    f"independent metric mismatch for {model}.{key}: {actual} != {expected}"
                )
            checked.append(f"{model}.{key}")
    return {
        "status": "VERIFIED",
        "method": "saved_prediction_csv_direct_equation_recalculation",
        "checked_field_count": len(checked),
        "checked_fields": checked,
        "recomputed_metrics": recomputed,
    }


def _unit_and_price_audit(dataset: OperationalDataset) -> dict:
    combined = pd.concat([dataset.train, dataset.test], ignore_index=True)
    da = combined["da_price"].to_numpy(float)
    rt = combined["rt_price"].to_numpy(float)
    if not np.all(np.isfinite(da)) or not np.all(np.isfinite(rt)):
        raise OfficialRunError("DA/RT prices are non-finite")
    # The files do not contain currency metadata, so prices are used as stored.
    return {
        "price_columns": ["DA_LMP", "RT_LMP"],
        "common_price_type": "MISO_LMP",
        "embedded_currency_metadata": "NOT_PRESENT_IN_CSV",
        "reported_profit_unit": "price_data_currency_unit",
        "capacity_mw": CAPACITY_MW,
        "duration_hours": DURATION_HOURS,
        "price_transformation": "NONE",
        "da_min": float(da.min()),
        "da_max": float(da.max()),
        "rt_min": float(rt.min()),
        "rt_max": float(rt.max()),
        "negative_da_observation_count": int(np.sum(da < 0)),
        "negative_rt_observation_count": int(np.sum(rt < 0)),
    }


def run_official(
    predictors_path: Path,
    da_path: Path,
    rt_path: Path,
    output_dir: Path,
) -> dict:
    validate_official_dataset_policy(OFFICIAL_DATASET_POLICY)
    output_dir.mkdir(parents=True, exist_ok=True)
    _assert_output_targets_absent(output_dir)

    print("[1/6] preprocessing Sydney-local 09:00-20:00 observed-RT dataset", flush=True)
    dataset = build_operational_dataset(predictors_path, da_path, rt_path)
    unit_audit = _unit_and_price_audit(dataset)
    scaler = fit_train_scaler(dataset.train)
    X_train, design_columns = build_mlr_design(dataset.train, scaler)
    X_test, design_columns_test = build_mlr_design(dataset.test, scaler)
    if design_columns_test != design_columns:
        raise OfficialRunError("train/test MLR design columns differ")

    _, history_actual = to_daily_matrix(dataset.history, "solar_power")
    _, train_actual = to_daily_matrix(dataset.train, "solar_power")
    _, test_actual = to_daily_matrix(dataset.test, "solar_power")
    _, train_da = to_daily_matrix(dataset.train, "da_price")
    _, train_rt = to_daily_matrix(dataset.train, "rt_price")
    if train_actual.shape != (TRAIN_DAYS, DAYLIGHT_SLOTS_PER_DAY):
        raise OfficialRunError("training daily matrix shape mismatch")
    if test_actual.shape != (TEST_DAYS, DAYLIGHT_SLOTS_PER_DAY):
        raise OfficialRunError("test daily matrix shape mismatch")

    print("[2/6] fitting Conventional AR and MLR", flush=True)
    conventional_ar = fit_conventional_ar(history_actual, train_actual)
    conventional_mlr = fit_conventional_mlr(
        X_train, dataset.train["solar_power"].to_numpy(float)
    )
    _solver_record(conventional_ar)
    _solver_record(conventional_mlr)

    print("[3/6] fitting Proposed AR (12 target-hour MILPs)", flush=True)
    proposed_ar = fit_proposed_ar(
        history_actual, train_actual, train_da, train_rt, w1=W1, w2=W2
    )
    _solver_record(proposed_ar)

    print("[4/6] fitting Proposed pooled MLR MILP", flush=True)
    proposed_mlr = fit_proposed_mlr(
        X_train,
        dataset.train["solar_power"].to_numpy(float),
        dataset.train["da_price"].to_numpy(float),
        dataset.train["rt_price"].to_numpy(float),
        w1=W1,
        w2=W2,
    )
    _solver_record(proposed_mlr)

    raw_predictions = {
        "Conventional AR": predict_ar_rolling(
            conventional_ar.coefficients_by_hour, train_actual[-1], test_actual
        ).reshape(-1),
        "Conventional MLR": predict_linear_raw(X_test, conventional_mlr.coefficients),
        "Proposed AR": predict_ar_rolling(
            proposed_ar.coefficients_by_hour, train_actual[-1], test_actual
        ).reshape(-1),
        "Proposed MLR": predict_linear_raw(X_test, proposed_mlr.coefficients),
    }
    fits: dict[str, LinearFit | ARFit] = {
        "Conventional AR": conventional_ar,
        "Conventional MLR": conventional_mlr,
        "Proposed AR": proposed_ar,
        "Proposed MLR": proposed_mlr,
    }
    test_s = dataset.test["solar_power"].to_numpy(float)
    test_da = dataset.test["da_price"].to_numpy(float)
    test_rt = dataset.test["rt_price"].to_numpy(float)
    summaries: dict[str, dict] = {}
    prediction_frames = []
    for model in MODEL_ORDER:
        summary, arrays = evaluate_raw_predictions(
            test_s, raw_predictions[model], test_da, test_rt
        )
        summaries[model] = summary
        prediction_frames.append(_prediction_frame(model, dataset.test, raw_predictions[model], summary, arrays))
    predictions = pd.concat(prediction_frames, ignore_index=True)

    print("[5/6] staging predictions and independently recomputing metrics", flush=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".csv", encoding="utf-8", dir=output_dir, delete=False
    ) as tmp:
        predictions.to_csv(tmp.name, index=False)
        staged_predictions = Path(tmp.name)
    try:
        reloaded = pd.read_csv(staged_predictions)
        recomputed = independently_recompute_metrics(reloaded)
        verification = _assert_metric_agreement(summaries, recomputed)
    except Exception:
        staged_predictions.unlink(missing_ok=True)
        raise

    source_hashes = {
        str(path.name): _sha256(path)
        for path in (predictors_path, da_path, rt_path)
    }
    result_models = {}
    for model in MODEL_ORDER:
        fit = fits[model]
        coefficients = (
            fit.coefficients.tolist() if isinstance(fit, LinearFit)
            else fit.coefficients_by_hour.tolist()
        )
        result_models[model] = {
            "status": "OPTIMAL",
            "raw_prediction_source": "native_raw_unclipped",
            "official_prediction_view": OFFICIAL_PREDICTION_VIEW,
            "training_objective": (
                "mean_absolute_error" if model.startswith("Conventional")
                else OBJECTIVE_DEFINITION
            ),
            "coefficients": coefficients,
            "solver": _solver_record(fit),
            "metrics": summaries[model],
        }
        if result_models[model]["raw_prediction_source"] not in RAW_PREDICTION_SOURCES:
            raise OfficialRunError("invalid raw prediction provenance")

    result = {
        "title": "Operational corrected reproduction of the APEN optimization-driven forecasting approach",
        "official_version": OFFICIAL_VERSION,
        "status": "OPTIMAL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_protocol": {
            "legacy_status": LEGACY_STATUS,
            "official_execution_use": LEGACY_OFFICIAL_EXECUTION_USE,
            "superseded_by": LEGACY_SUPERSEDED_BY,
        },
        "dataset_policy": asdict(OFFICIAL_DATASET_POLICY),
        "source_hashes": source_hashes,
        "source_sequence_audit": dataset.source_sequence_audit,
        "alignment_audit": dataset.alignment_audit,
        "unit_and_price_audit": unit_audit,
        "preprocessing": {
            "accumulation_group_status": "STRUCTURALLY_IDENTIFIED_FROM_OFFICIAL_SOURCE_SEQUENCE",
            "differencing_order": "complete_source_sequence_before_timezone_conversion_and_local_09_to_20_filter",
            "negative_increment_policy": "preserve_source_difference_without_clipping",
            "timezone_policy": "step1_naive_as_utc_to_australia_sydney",
            "timezone_policy_status": "reproduction_assumption",
            "target_timezone": LOCAL_TIMEZONE,
            "day_grouping": "Sydney local date",
            "daylight_window": "local hours 09,...,20",
            "scaler": scaler.to_dict(),
            "mlr_design_columns": list(design_columns),
        },
        "objective": {
            "objective_definition": OBJECTIVE_DEFINITION,
            "W1": W1,
            "W2": W2,
            "oracle_denominator_source": "training_only",
        },
        "models": result_models,
        "official_prediction_rows": int(len(predictions)),
        "metric_verification_status": verification["status"],
        "superseded_invalid_result": {
            "status": "INVALID_TIME_WINDOW_SUPERSEDED",
            "reason": "UTC_00_TO_11_REUSED_INSTEAD_OF_CANONICAL_SYDNEY_LOCAL_09_TO_20",
            "official_use": "FORBIDDEN",
            "preserved_path": "invalid_utc_window",
        },
    }
    # Also catches NaN or infinity before writing the files.
    json.dumps(result, allow_nan=False)
    json.dumps(verification, allow_nan=False)

    print("[6/6] committing verified official artifacts", flush=True)
    os.replace(staged_predictions, output_dir / "operational_predictions.csv")
    dataset.train.to_csv(output_dir / "operational_preprocessed_train.csv", index=False)
    dataset.test.to_csv(output_dir / "operational_preprocessed_test.csv", index=False)
    _atomic_json(output_dir / "operational_results.json", result)
    _atomic_json(output_dir / "operational_metric_verification.json", verification)
    return result


def _failure_payload(exc: Exception) -> dict:
    payload = {
        "official_result_created": False,
        "status": "FAILED",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if isinstance(exc, SolverFailure):
        payload.update({
            "solver_model": exc.model_name,
            "solver_status": exc.status,
            "solver_message": exc.solver_message,
        })
    return payload


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictors", type=Path, default=root / "data" / "predictors15.csv")
    parser.add_argument("--da", type=Path, default=root / "data" / "da_lmp_prices.csv")
    parser.add_argument("--rt", type=Path, default=root / "data" / "rt_lmp_prices.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_official(args.predictors, args.da, args.rt, args.output_dir)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(args.output_dir / "operational_failure_diagnostics.json", _failure_payload(exc))
        print(f"official run failed: {exc}", flush=True)
        return 1
    print(json.dumps({
        "status": result["status"],
        "models": {name: data["metrics"] for name, data in result["models"].items()},
    }, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
