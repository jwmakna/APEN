"""Run the four models on the 300-day/100-day split."""

import argparse
import json
from pathlib import Path

import pandas as pd

from operational_config import PENALTY_RATE, TEST_DAYS, TRAIN_DAYS, W1, W2
from operational_evaluation import evaluate_raw_predictions
from operational_models import (
    ARFit,
    LinearFit,
    fit_conventional_ar,
    fit_conventional_mlr,
    fit_proposed_ar,
    fit_proposed_mlr,
    predict_ar_rolling,
    predict_linear_raw,
)
from operational_preprocessing import (
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


def _fit_summary(fit):
    if isinstance(fit, LinearFit):
        return {
            "formulation": fit.formulation,
            "objective_value": fit.objective_value,
            "mip_gap": fit.mip_gap,
            "binary_variable_count": fit.binary_variable_count,
        }
    return {
        "formulation": fit.formulation,
        "objective_values_by_hour": fit.objective_values_by_hour.tolist(),
        "mip_gaps_by_hour": list(fit.mip_gaps_by_hour),
        "binary_variable_counts_by_hour": list(fit.binary_variable_counts_by_hour),
    }


def _prediction_frame(model, test, arrays):
    return pd.DataFrame({
        "model": model,
        "timestamp_utc": test["timestamp"].map(lambda value: value.isoformat()),
        "timestamp_local": test["timestamp_local"].map(lambda value: value.isoformat()),
        "local_date": test["local_date"].map(lambda value: value.isoformat()),
        "local_hour": test["local_hour"].to_numpy(),
        "hour_idx": test["hour_idx"].to_numpy(),
        "actual": arrays["actual"],
        "raw": arrays["raw"],
        "projected": arrays["projected"],
        "DA": arrays["da"],
        "RT": arrays["rt"],
        "penalty": arrays["penalty"],
        "realized_profit": arrays["realized_profit"],
        "oracle_profit": arrays["oracle_profit"],
        "oracle_q": arrays["oracle_commitment"],
    })


def run_experiment(predictors_path, da_path, rt_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] preparing data")
    dataset = build_operational_dataset(predictors_path, da_path, rt_path)
    scaler = fit_train_scaler(dataset.train)
    X_train, design_columns = build_mlr_design(dataset.train, scaler)
    X_test, _ = build_mlr_design(dataset.test, scaler)

    _, history_actual = to_daily_matrix(dataset.history, "solar_power")
    _, train_actual = to_daily_matrix(dataset.train, "solar_power")
    _, test_actual = to_daily_matrix(dataset.test, "solar_power")
    _, train_da = to_daily_matrix(dataset.train, "da_price")
    _, train_rt = to_daily_matrix(dataset.train, "rt_price")

    print("[2/5] fitting conventional models")
    conventional_ar = fit_conventional_ar(history_actual, train_actual)
    conventional_mlr = fit_conventional_mlr(
        X_train, dataset.train["solar_power"].to_numpy()
    )

    print("[3/5] fitting proposed models")
    proposed_ar = fit_proposed_ar(
        history_actual, train_actual, train_da, train_rt, w1=W1, w2=W2
    )
    proposed_mlr = fit_proposed_mlr(
        X_train,
        dataset.train["solar_power"].to_numpy(),
        dataset.train["da_price"].to_numpy(),
        dataset.train["rt_price"].to_numpy(),
        w1=W1,
        w2=W2,
    )

    fits = {
        "Conventional AR": conventional_ar,
        "Conventional MLR": conventional_mlr,
        "Proposed AR": proposed_ar,
        "Proposed MLR": proposed_mlr,
    }
    raw_predictions = {
        "Conventional AR": predict_ar_rolling(
            conventional_ar.coefficients_by_hour, train_actual[-1], test_actual
        ).reshape(-1),
        "Conventional MLR": predict_linear_raw(
            X_test, conventional_mlr.coefficients
        ),
        "Proposed AR": predict_ar_rolling(
            proposed_ar.coefficients_by_hour, train_actual[-1], test_actual
        ).reshape(-1),
        "Proposed MLR": predict_linear_raw(X_test, proposed_mlr.coefficients),
    }

    print("[4/5] evaluating predictions")
    actual = dataset.test["solar_power"].to_numpy()
    day_ahead = dataset.test["da_price"].to_numpy()
    real_time = dataset.test["rt_price"].to_numpy()
    models = {}
    prediction_frames = []

    for model in MODEL_ORDER:
        metrics, arrays = evaluate_raw_predictions(
            actual, raw_predictions[model], day_ahead, real_time
        )
        fit = fits[model]
        coefficients = (
            fit.coefficients.tolist()
            if isinstance(fit, LinearFit)
            else fit.coefficients_by_hour.tolist()
        )
        models[model] = {
            "coefficients": coefficients,
            "fit": _fit_summary(fit),
            "metrics": metrics,
        }
        prediction_frames.append(_prediction_frame(model, dataset.test, arrays))

    results = {
        "data_split": {
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "train_rows": len(dataset.train),
            "test_rows": len(dataset.test),
        },
        "preprocessing": {
            "timezone": "Australia/Sydney",
            "local_hours": "09:00-20:00",
            "scaler": scaler.to_dict(),
            "mlr_design_columns": list(design_columns),
        },
        "objective_weights": {"W1": W1, "W2": W2},
        "penalty_rate": PENALTY_RATE,
        "models": models,
    }

    print("[5/5] saving results")
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(output_dir / "operational_predictions.csv", index=False)
    dataset.train.to_csv(output_dir / "operational_preprocessed_train.csv", index=False)
    dataset.test.to_csv(output_dir / "operational_preprocessed_test.csv", index=False)
    with (output_dir / "operational_results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)
        file.write("\n")
    return results


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictors", type=Path, default=root / "data" / "predictors15.csv")
    parser.add_argument("--da", type=Path, default=root / "data" / "da_lmp_prices.csv")
    parser.add_argument("--rt", type=Path, default=root / "data" / "rt_lmp_prices.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "reproduced_results")
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_experiment(args.predictors, args.da, args.rt, args.output_dir)
    metrics = {name: result["metrics"] for name, result in results["models"].items()}
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
