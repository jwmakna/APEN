"""AR and MLR models used in the experiment."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from operational_config import (
    CAPACITY_MW,
    DURATION_HOURS,
    PENALTY_RATE,
    W1,
    W2,
)
from operational_evaluation import (
    normalized_economic_loss,
    oracle_profit_per_observation,
)


class SolverFailure(RuntimeError):
    pass


@dataclass
class LinearFit:
    coefficients: np.ndarray
    fitted_raw: np.ndarray
    objective_value: float
    formulation: str
    mip_gap: float | None = None
    binary_variable_count: int = 0


@dataclass
class ARFit:
    coefficients_by_hour: np.ndarray
    fitted_raw: np.ndarray
    objective_values_by_hour: np.ndarray
    formulation: str
    mip_gaps_by_hour: tuple[float | None, ...] = ()
    binary_variable_counts_by_hour: tuple[int, ...] = ()


def _solve_bounded_lad(X, actual, model_name):
    X = np.asarray(X, dtype=float)
    actual = np.asarray(actual, dtype=float)
    n, p = X.shape

    X_sparse = sparse.csr_matrix(X)
    identity = sparse.eye(n, format="csr")
    zeros = sparse.csr_matrix((n, n))
    constraints = sparse.vstack([
        sparse.hstack([X_sparse, -identity]),
        sparse.hstack([-X_sparse, -identity]),
        sparse.hstack([X_sparse, zeros]),
        sparse.hstack([-X_sparse, zeros]),
    ], format="csr")
    limits = np.concatenate([actual, -actual, np.ones(n), np.zeros(n)])
    objective = np.concatenate([np.zeros(p), np.ones(n) / n])
    bounds = [(None, None)] * p + [(0.0, None)] * n

    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=limits,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise SolverFailure(f"{model_name}: {result.message}")

    coefficients = result.x[:p]
    fitted = X @ coefficients
    return LinearFit(
        coefficients=coefficients,
        fitted_raw=fitted,
        objective_value=float(np.mean(np.abs(actual - fitted))),
        formulation="bounded LAD",
    )


def fit_bounded_lad(X, actual, *, model_name="bounded_lad"):
    return _solve_bounded_lad(X, actual, model_name)


def combined_training_objective(
    actual,
    commitment,
    day_ahead_price,
    real_time_price,
    *,
    w1=W1,
    w2=W2,
):
    economic_loss = normalized_economic_loss(
        actual, commitment, day_ahead_price, real_time_price
    )
    mae = np.mean(np.abs(np.asarray(actual) - np.asarray(commitment)))
    return float(w1 * economic_loss + w2 * mae)


def fit_paper_proposed_milp(
    X,
    actual,
    day_ahead_price,
    real_time_price,
    *,
    w1=W1,
    w2=W2,
    model_name="paper_proposed_linear",
):
    X = np.asarray(X, dtype=float)
    actual = np.asarray(actual, dtype=float)
    day_ahead_price = np.asarray(day_ahead_price, dtype=float)
    real_time_price = np.asarray(real_time_price, dtype=float)

    if w1 == 0:
        fit = _solve_bounded_lad(X, actual, model_name)
        fit.objective_value *= w2
        fit.formulation = "bounded LAD (W1=0)"
        return fit

    n, p = X.shape
    oracle, _ = oracle_profit_per_observation(
        actual, day_ahead_price, real_time_price
    )
    denominator = oracle.sum()
    scale = CAPACITY_MW * DURATION_HOURS
    penalty = PENALTY_RATE * day_ahead_price

    surplus_cost = -w1 * scale * real_time_price / denominator + w2 / n
    shortage_cost = w1 * scale * penalty / denominator + w2 / n
    binary_rows = np.flatnonzero(surplus_cost + shortage_cost < 0)
    binary_count = len(binary_rows)

    beta = slice(0, p)
    x = slice(p, p + n)
    y_plus = slice(p + n, p + 2 * n)
    y_minus = slice(p + 2 * n, p + 3 * n)
    z = slice(p + 3 * n, p + 3 * n + binary_count)
    variable_count = p + 3 * n + binary_count

    objective = np.zeros(variable_count)
    objective[x] = -w1 * scale * day_ahead_price / denominator
    objective[y_plus] = surplus_cost
    objective[y_minus] = shortage_cost

    equations = sparse.lil_matrix((2 * n, variable_count))
    equations[:n, beta] = -X
    equations[:n, x] = sparse.eye(n)
    equations[n:, x] = sparse.eye(n)
    equations[n:, y_plus] = sparse.eye(n)
    equations[n:, y_minus] = -sparse.eye(n)
    right_hand_side = np.concatenate([np.zeros(n), actual])
    constraints = [
        LinearConstraint(equations.tocsr(), right_hand_side, right_hand_side)
    ]

    if binary_count:
        complementarity = sparse.lil_matrix((2 * binary_count, variable_count))
        complementarity[np.arange(binary_count), y_plus.start + binary_rows] = 1
        complementarity[np.arange(binary_count), z.start + np.arange(binary_count)] = 1
        complementarity[
            binary_count + np.arange(binary_count), y_minus.start + binary_rows
        ] = 1
        complementarity[
            binary_count + np.arange(binary_count), z.start + np.arange(binary_count)
        ] = -1
        constraints.append(
            LinearConstraint(
                complementarity.tocsr(),
                np.full(2 * binary_count, -np.inf),
                np.r_[np.ones(binary_count), np.zeros(binary_count)],
            )
        )

    lower = np.r_[np.full(p, -np.inf), np.zeros(3 * n + binary_count)]
    upper = np.r_[np.full(p, np.inf), np.ones(3 * n + binary_count)]
    integrality = np.zeros(variable_count, dtype=int)
    integrality[z] = 1

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"mip_rel_gap": 1e-9},
    )
    if not result.success:
        raise SolverFailure(f"{model_name}: {result.message}")

    coefficients = result.x[beta]
    fitted = X @ coefficients
    mip_gap = getattr(result, "mip_gap", None)
    return LinearFit(
        coefficients=coefficients,
        fitted_raw=fitted,
        objective_value=combined_training_objective(
            actual,
            fitted,
            day_ahead_price,
            real_time_price,
            w1=w1,
            w2=w2,
        ),
        formulation="sign-reduced complementarity MILP",
        mip_gap=None if mip_gap is None else float(mip_gap),
        binary_variable_count=int(binary_count),
    )


def build_ar_lag_design(history_daily, train_daily):
    history = np.asarray(history_daily, dtype=float)
    train = np.asarray(train_daily, dtype=float)
    previous_day = np.vstack([history[-1], train[:-1]])
    X = np.column_stack([np.ones(len(train)), previous_day[:, ::-1]])
    return X, train.copy()


def fit_baseline_ar(history_daily, train_daily):
    X, targets = build_ar_lag_design(history_daily, train_daily)
    fits = [
        _solve_bounded_lad(X, targets[:, hour], f"Baseline AR hour {hour}")
        for hour in range(12)
    ]
    return ARFit(
        coefficients_by_hour=np.vstack([fit.coefficients for fit in fits]),
        fitted_raw=np.column_stack([fit.fitted_raw for fit in fits]),
        objective_values_by_hour=np.array([fit.objective_value for fit in fits]),
        formulation="12 hourly bounded LAD models",
    )


def fit_paper_proposed_ar(
    history_daily,
    train_daily,
    train_da_daily,
    train_rt_daily,
    *,
    w1=W1,
    w2=W2,
):
    X, targets = build_ar_lag_design(history_daily, train_daily)
    fits = [
        fit_paper_proposed_milp(
            X,
            targets[:, hour],
            train_da_daily[:, hour],
            train_rt_daily[:, hour],
            w1=w1,
            w2=w2,
            model_name=f"Paper-proposed AR hour {hour}",
        )
        for hour in range(12)
    ]
    return ARFit(
        coefficients_by_hour=np.vstack([fit.coefficients for fit in fits]),
        fitted_raw=np.column_stack([fit.fitted_raw for fit in fits]),
        objective_values_by_hour=np.array([fit.objective_value for fit in fits]),
        formulation="12 hourly sign-reduced MILP models",
        mip_gaps_by_hour=tuple(fit.mip_gap for fit in fits),
        binary_variable_counts_by_hour=tuple(
            fit.binary_variable_count for fit in fits
        ),
    )


def predict_ar_rolling(coefficients_by_hour, previous_day_actual, test_actual):
    coefficients = np.asarray(coefficients_by_hour, dtype=float)
    previous_day = np.asarray(previous_day_actual, dtype=float)
    test_actual = np.asarray(test_actual, dtype=float)
    predictions = np.empty_like(test_actual)

    for day in range(len(test_actual)):
        features = np.r_[1.0, previous_day[::-1]]
        predictions[day] = coefficients @ features
        previous_day = test_actual[day]
    return predictions


def fit_baseline_mlr(X, actual):
    return _solve_bounded_lad(X, actual, "Baseline MLR")


def fit_paper_proposed_mlr(
    X, actual, day_ahead_price, real_time_price, *, w1=W1, w2=W2
):
    return fit_paper_proposed_milp(
        X,
        actual,
        day_ahead_price,
        real_time_price,
        w1=w1,
        w2=w2,
        model_name="Paper-proposed MLR",
    )


def predict_linear_raw(X, coefficients):
    return np.asarray(X, dtype=float) @ np.asarray(coefficients, dtype=float)
