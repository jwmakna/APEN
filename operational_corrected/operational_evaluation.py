"""Profit and prediction-error calculations."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from operational_config import (
    BOUNDARY_TOL,
    CAPACITY_MW,
    DENOMINATOR_TOL,
    DURATION_HOURS,
    PENALTY_RATE,
)


class EvaluationError(ValueError):
    pass


@dataclass(frozen=True)
class PredictionDiagnostics:
    raw_prediction_min: float
    raw_prediction_max: float
    raw_boundary_violation_count: int
    projection_changed_count: int
    projection_max_absolute_change: float
    evaluated_prediction_min: float
    evaluated_prediction_max: float


def _array(name: str, value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        raise EvaluationError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise EvaluationError(f"{name} contains NaN or infinity")
    return arr


def _same_shape(**arrays) -> dict[str, np.ndarray]:
    converted = {name: _array(name, value) for name, value in arrays.items()}
    shapes = {arr.shape for arr in converted.values()}
    if len(shapes) != 1:
        raise EvaluationError(f"shape mismatch: {[(k, v.shape) for k, v in converted.items()]}")
    return converted


def _positive_finite_scalar(name: str, value) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise EvaluationError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise EvaluationError(f"{name} must be finite and positive")
    return result


def feasible_unit_projection(raw) -> np.ndarray:
    arr = _array("raw", raw)
    return np.minimum(1.0, np.maximum(0.0, arr))


def prediction_diagnostics(raw, tol: float = BOUNDARY_TOL) -> PredictionDiagnostics:
    if not isinstance(tol, Real) or isinstance(tol, (bool, np.bool_)):
        raise EvaluationError("tol must be a finite nonnegative real scalar")
    tol = float(tol)
    if not np.isfinite(tol) or tol < 0:
        raise EvaluationError("tol must be a finite nonnegative real scalar")
    raw_arr = _array("raw", raw)
    before = raw_arr.copy()
    projected = feasible_unit_projection(raw_arr)
    if not np.array_equal(raw_arr, before):
        raise AssertionError("projection mutated raw prediction")
    diff = np.abs(projected - raw_arr)
    result = PredictionDiagnostics(
        raw_prediction_min=float(raw_arr.min()),
        raw_prediction_max=float(raw_arr.max()),
        raw_boundary_violation_count=int(np.sum((raw_arr < -tol) | (raw_arr > 1 + tol))),
        projection_changed_count=int(np.sum(diff > tol)),
        projection_max_absolute_change=float(diff.max()),
        evaluated_prediction_min=float(projected.min()),
        evaluated_prediction_max=float(projected.max()),
    )
    for field in (result.raw_prediction_min, result.raw_prediction_max):
        if not isinstance(field, float) or not np.isfinite(field):
            raise EvaluationError("raw min/max must be finite real scalars")
    if result.raw_prediction_min > result.raw_prediction_max:
        raise EvaluationError("raw_prediction_min exceeds raw_prediction_max")
    return result


def validate_official_prediction(q) -> np.ndarray:
    arr = _array("official prediction", q)
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise EvaluationError("official projected prediction must be exactly inside [0,1]")
    return arr


def imbalance_components(S, q) -> tuple[np.ndarray, np.ndarray]:
    a = _same_shape(S=S, q=q)
    surplus = np.maximum(a["S"] - a["q"], 0.0)
    shortage = np.maximum(a["q"] - a["S"], 0.0)
    if np.any((surplus > 0) & (shortage > 0)):
        raise AssertionError("surplus and shortage are simultaneously positive")
    return surplus, shortage


def profit_per_observation(
    S, q, DP, RP, *, penalty_rate: float = PENALTY_RATE,
    capacity_mw: float = CAPACITY_MW, duration_hours: float = DURATION_HOURS,
) -> np.ndarray:
    a = _same_shape(S=S, q=q, DP=DP, RP=RP)
    q_arr = validate_official_prediction(a["q"])
    if np.any(a["S"] < 0) or np.any(a["S"] > 1):
        raise EvaluationError("S must be in [0,1]")
    penalty_rate = _positive_finite_scalar("penalty_rate", penalty_rate)
    capacity_mw = _positive_finite_scalar("capacity_mw", capacity_mw)
    duration_hours = _positive_finite_scalar("duration_hours", duration_hours)
    surplus, shortage = imbalance_components(a["S"], q_arr)
    PC = penalty_rate * a["DP"]
    return capacity_mw * duration_hours * (
        a["DP"] * q_arr + a["RP"] * surplus - PC * shortage
    )


def oracle_profit_per_observation(
    S, DP, RP, *, penalty_rate: float = PENALTY_RATE,
    capacity_mw: float = CAPACITY_MW, duration_hours: float = DURATION_HOURS,
) -> tuple[np.ndarray, np.ndarray]:
    a = _same_shape(S=S, DP=DP, RP=RP)
    candidates = [np.zeros_like(a["S"]), a["S"].copy(), np.ones_like(a["S"])]
    profits = np.stack([
        profit_per_observation(
            a["S"], q, a["DP"], a["RP"], penalty_rate=penalty_rate,
            capacity_mw=capacity_mw, duration_hours=duration_hours,
        )
        for q in candidates
    ], axis=0)
    best_index = np.argmax(profits, axis=0)
    best_profit = np.take_along_axis(profits, np.expand_dims(best_index, 0), axis=0)[0]
    candidate_stack = np.stack(candidates, axis=0)
    best_q = np.take_along_axis(candidate_stack, np.expand_dims(best_index, 0), axis=0)[0]
    return best_profit, best_q


def normalized_economic_loss(
    S, q, DP, RP, *, penalty_rate: float = PENALTY_RATE,
    capacity_mw: float = CAPACITY_MW, duration_hours: float = DURATION_HOURS,
) -> float:
    oracle, _ = oracle_profit_per_observation(
        S, DP, RP, penalty_rate=penalty_rate,
        capacity_mw=capacity_mw, duration_hours=duration_hours,
    )
    realized = profit_per_observation(
        S, q, DP, RP, penalty_rate=penalty_rate,
        capacity_mw=capacity_mw, duration_hours=duration_hours,
    )
    denominator = float(np.sum(oracle))
    if not np.isfinite(denominator) or denominator <= DENOMINATOR_TOL:
        raise EvaluationError("oracle profit denominator is nonpositive or too small")
    loss_vector = oracle - realized
    if np.any(loss_vector < -BOUNDARY_TOL):
        raise EvaluationError("realized profit exceeds corrected oracle beyond tolerance")
    return float(np.sum(loss_vector) / denominator)


def rmse(S, q) -> float:
    a = _same_shape(S=S, q=q)
    return float(np.sqrt(np.mean((a["q"] - a["S"]) ** 2)))


def nrmse_percent(S, q) -> float:
    a = _same_shape(S=S, q=q)
    mean_actual = float(np.mean(a["S"]))
    if not np.isfinite(mean_actual) or mean_actual <= 0:
        raise EvaluationError("mean test actual must be positive")
    return 100.0 * rmse(a["S"], a["q"]) / mean_actual


def evaluate_raw_predictions(S, raw, DP, RP) -> tuple[dict, dict[str, np.ndarray]]:
    a = _same_shape(S=S, raw=raw, DP=DP, RP=RP)
    diagnostics = prediction_diagnostics(a["raw"])
    projected = feasible_unit_projection(a["raw"])
    validate_official_prediction(projected)
    realized = profit_per_observation(a["S"], projected, a["DP"], a["RP"])
    oracle, oracle_q = oracle_profit_per_observation(a["S"], a["DP"], a["RP"])
    losses = oracle - realized
    if np.any(losses < -BOUNDARY_TOL):
        raise EvaluationError("negative oracle loss beyond tolerance")
    denominator = float(np.sum(oracle))
    if denominator <= DENOMINATOR_TOL:
        raise EvaluationError("oracle profit denominator is nonpositive or too small")
    summary = {
        "official_nrmse_percent": nrmse_percent(a["S"], projected),
        "raw_nrmse_percent_diagnostic": nrmse_percent(a["S"], a["raw"]),
        "absolute_gap": float(np.sum(losses)),
        "realized_profit": float(np.sum(realized)),
        "oracle_profit": denominator,
        "gap_percent": 100.0 * float(np.sum(losses)) / denominator,
        **diagnostics.__dict__,
    }
    arrays = {
        "actual": a["S"].copy(),
        "raw": a["raw"].copy(),
        "projected": projected,
        "da": a["DP"].copy(),
        "rt": a["RP"].copy(),
        "penalty": PENALTY_RATE * a["DP"],
        "realized_profit": realized,
        "oracle_profit": oracle,
        "oracle_q": oracle_q,
    }
    return summary, arrays
