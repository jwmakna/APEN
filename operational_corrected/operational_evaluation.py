"""Prediction and profit metrics."""

import numpy as np

from operational_config import CAPACITY_MW, DURATION_HOURS, PENALTY_RATE


def _arrays(**values):
    arrays = {name: np.asarray(value, dtype=float) for name, value in values.items()}
    if len({array.shape for array in arrays.values()}) != 1:
        raise ValueError("all inputs must have the same shape")
    return arrays


def feasible_unit_projection(raw):
    return np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)


def imbalance_components(actual, commitment):
    data = _arrays(actual=actual, commitment=commitment)
    surplus = np.maximum(data["actual"] - data["commitment"], 0.0)
    shortage = np.maximum(data["commitment"] - data["actual"], 0.0)
    return surplus, shortage


def profit_per_observation(
    actual,
    commitment,
    day_ahead_price,
    real_time_price,
    penalty_rate=PENALTY_RATE,
):
    data = _arrays(
        actual=actual,
        commitment=commitment,
        day_ahead=day_ahead_price,
        real_time=real_time_price,
    )
    surplus, shortage = imbalance_components(data["actual"], data["commitment"])
    penalty = penalty_rate * data["day_ahead"]
    return CAPACITY_MW * DURATION_HOURS * (
        data["day_ahead"] * data["commitment"]
        + data["real_time"] * surplus
        - penalty * shortage
    )


def oracle_profit_per_observation(actual, day_ahead_price, real_time_price):
    actual = np.asarray(actual, dtype=float)
    candidates = np.stack([np.zeros_like(actual), actual, np.ones_like(actual)])
    profits = np.stack([
        profit_per_observation(actual, x, day_ahead_price, real_time_price)
        for x in candidates
    ])
    best = profits.argmax(axis=0)
    oracle_profit = np.take_along_axis(profits, best[None, :], axis=0)[0]
    oracle_commitment = np.take_along_axis(candidates, best[None, :], axis=0)[0]
    return oracle_profit, oracle_commitment


def normalized_economic_loss(actual, commitment, day_ahead_price, real_time_price):
    oracle, _ = oracle_profit_per_observation(
        actual, day_ahead_price, real_time_price
    )
    realized = profit_per_observation(
        actual, commitment, day_ahead_price, real_time_price
    )
    return float((oracle - realized).sum() / oracle.sum())


def rmse(actual, prediction):
    data = _arrays(actual=actual, prediction=prediction)
    return float(np.sqrt(np.mean((data["prediction"] - data["actual"]) ** 2)))


def nrmse_percent(actual, prediction):
    actual = np.asarray(actual, dtype=float)
    return 100.0 * rmse(actual, prediction) / actual.mean()


def evaluate_raw_predictions(actual, raw, day_ahead_price, real_time_price):
    data = _arrays(
        actual=actual,
        raw=raw,
        day_ahead=day_ahead_price,
        real_time=real_time_price,
    )
    projected = feasible_unit_projection(data["raw"])
    realized = profit_per_observation(
        data["actual"], projected, data["day_ahead"], data["real_time"]
    )
    oracle, oracle_commitment = oracle_profit_per_observation(
        data["actual"], data["day_ahead"], data["real_time"]
    )
    losses = oracle - realized
    change = np.abs(projected - data["raw"])

    summary = {
        "official_nrmse_percent": nrmse_percent(data["actual"], projected),
        "raw_nrmse_percent_diagnostic": nrmse_percent(data["actual"], data["raw"]),
        "absolute_gap": float(losses.sum()),
        "realized_profit": float(realized.sum()),
        "oracle_profit": float(oracle.sum()),
        "gap_percent": 100.0 * float(losses.sum() / oracle.sum()),
        "raw_prediction_min": float(data["raw"].min()),
        "raw_prediction_max": float(data["raw"].max()),
        "raw_boundary_violation_count": int(
            np.sum((data["raw"] < 0.0) | (data["raw"] > 1.0))
        ),
        "projection_changed_count": int(np.sum(change > 0.0)),
        "projection_max_absolute_change": float(change.max()),
        "evaluated_prediction_min": float(projected.min()),
        "evaluated_prediction_max": float(projected.max()),
    }
    arrays = {
        "actual": data["actual"],
        "raw": data["raw"],
        "projected": projected,
        "da": data["day_ahead"],
        "rt": data["real_time"],
        "penalty": PENALTY_RATE * data["day_ahead"],
        "realized_profit": realized,
        "oracle_profit": oracle,
        "oracle_commitment": oracle_commitment,
    }
    return summary, arrays
