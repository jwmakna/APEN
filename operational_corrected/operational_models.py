"""AR/MLR fitting routines used in the experiment."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from operational_config import (
    BOUNDARY_TOL,
    CAPACITY_MW,
    DURATION_HOURS,
    PENALTY_RATE,
    SOLVER_TOL,
    W1,
    W2,
)
from operational_evaluation import (
    normalized_economic_loss,
    oracle_profit_per_observation,
)


class ModelError(ValueError):
    pass


class SolverFailure(RuntimeError):
    def __init__(self, model_name: str, status, message: str):
        super().__init__(f"{model_name} solver failed: status={status}, message={message}")
        self.model_name = model_name
        self.status = status
        self.solver_message = message


@dataclass(frozen=True)
class LinearFit:
    coefficients: np.ndarray
    fitted_raw: np.ndarray
    objective_value: float
    solver_status: int
    solver_message: str
    certified_optimal: bool
    formulation: str
    mip_gap: float | None = None
    binary_variable_count: int = 0
    unreduced_binary_variable_count: int = 0


@dataclass(frozen=True)
class ARFit:
    coefficients_by_hour: np.ndarray
    fitted_raw: np.ndarray
    objective_values_by_hour: np.ndarray
    solver_statuses_by_hour: tuple[int, ...]
    solver_messages_by_hour: tuple[str, ...]
    certified_optimal: bool
    formulation: str
    mip_gaps_by_hour: tuple[float | None, ...] = ()
    binary_variable_counts_by_hour: tuple[int, ...] = ()
    unreduced_binary_variable_counts_by_hour: tuple[int, ...] = ()


def _validate_xy(X, actual) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if X.ndim != 2 or actual.ndim != 1:
        raise ModelError("X must be 2D and actual must be 1D")
    if X.shape[0] != actual.shape[0]:
        raise ModelError(f"row mismatch X={X.shape}, actual={actual.shape}")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ModelError("empty design")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(actual)):
        raise ModelError("X or actual contains NaN/infinity")
    if np.any(actual < 0) or np.any(actual > 1):
        raise ModelError("actual generation must be in [0,1]")
    return X, actual


def _validate_fitted_bounds(fitted: np.ndarray, name: str) -> None:
    if np.any(fitted < -SOLVER_TOL) or np.any(fitted > 1 + SOLVER_TOL):
        raise SolverFailure(name, "POSTCHECK", "training fitted prediction violates [0,1]")


def fit_bounded_lad(X, actual, *, model_name: str = "bounded_lad") -> LinearFit:
    X, actual = _validate_xy(X, actual)
    n, p = X.shape
    # Decision variables: regression coefficients and absolute residuals.
    c = np.concatenate([np.zeros(p), np.ones(n) / n])
    # Sparse matrices keep the pooled 3,600-row MLR problem small enough.
    X_sparse = sparse.csr_matrix(X)
    identity = sparse.eye(n, format="csr")
    zeros = sparse.csr_matrix((n, n))
    A_ub = sparse.vstack([
        sparse.hstack([X_sparse, -identity]),  # Xb-r <= y
        sparse.hstack([-X_sparse, -identity]), # -Xb-r <= -y
        sparse.hstack([X_sparse, zeros]),      # Xb <= 1
        sparse.hstack([-X_sparse, zeros]),     # -Xb <= 0
    ], format="csr")
    b_ub = np.concatenate([actual, -actual, np.ones(n), np.zeros(n)])
    bounds = [(None, None)] * p + [(0.0, None)] * n
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success or result.status != 0:
        raise SolverFailure(model_name, result.status, result.message)
    coefficients = np.asarray(result.x[:p], dtype=float)
    fitted = X @ coefficients
    _validate_fitted_bounds(fitted, model_name)
    objective = float(np.mean(np.abs(actual - fitted)))
    return LinearFit(
        coefficients=coefficients,
        fitted_raw=fitted,
        objective_value=objective,
        solver_status=int(result.status),
        solver_message=str(result.message),
        certified_optimal=True,
        formulation="bounded_lad_linear_program",
    )


def combined_training_objective(
    actual, commitment, DP, RP, *, w1: float = W1, w2: float = W2
) -> float:
    """Return W1 * economic loss + W2 * mean absolute error."""
    actual = np.asarray(actual, dtype=float)
    commitment = np.asarray(commitment, dtype=float)
    if commitment.shape != actual.shape:
        raise ModelError("actual and commitment shapes differ")
    bounded = np.clip(commitment, 0.0, 1.0)
    econ = normalized_economic_loss(actual, bounded, DP, RP)
    forecast = float(np.mean(np.abs(actual - commitment)))
    return float(w1 * econ + w2 * forecast)


def fit_proposed_milp(
    X, actual, DP, RP, *, w1: float = W1, w2: float = W2,
    model_name: str = "proposed_linear",
) -> LinearFit:
    X, actual = _validate_xy(X, actual)
    DP = np.asarray(DP, dtype=float)
    RP = np.asarray(RP, dtype=float)
    if DP.shape != actual.shape or RP.shape != actual.shape:
        raise ModelError("DP/RP must have the same 1D shape as actual")
    if not np.all(np.isfinite(DP)) or not np.all(np.isfinite(RP)):
        raise ModelError("DP/RP contains NaN/infinity")
    if not np.isfinite(w1) or not np.isfinite(w2) or w1 < 0 or w2 <= 0:
        raise ModelError("weights must satisfy finite w1>=0 and w2>0")

    if w1 == 0:
        # With no economic term, Proposed is exactly the bounded LAD model.
        lad = fit_bounded_lad(X, actual, model_name=f"{model_name}_w1_zero")
        return LinearFit(
            coefficients=lad.coefficients,
            fitted_raw=lad.fitted_raw,
            objective_value=float(w2 * lad.objective_value),
            solver_status=lad.solver_status,
            solver_message=lad.solver_message,
            certified_optimal=True,
            formulation="w1_zero_exact_bounded_lad_reduction",
        )

    n, p = X.shape
    oracle, _ = oracle_profit_per_observation(actual, DP, RP)
    oracle_denominator = float(np.sum(oracle))
    if not np.isfinite(oracle_denominator) or oracle_denominator <= 1e-9:
        raise ModelError("training oracle denominator is nonpositive or too small")

    # Paper notation: x = commitment, y_plus = surplus, y_minus = shortage.
    # A binary is needed only when increasing y_plus and y_minus together can
    # reduce the objective. This is an exact reduction of the full MILP.
    scale = CAPACITY_MW * DURATION_HOURS
    PC = PENALTY_RATE * DP
    surplus_cost = -w1 * scale * RP / oracle_denominator + w2 / n
    shortage_cost = +w1 * scale * PC / oracle_denominator + w2 / n
    binary_observations = np.flatnonzero((surplus_cost + shortage_cost) < 0.0)
    binary_count = int(len(binary_observations))

    # Variable blocks: beta, x, y_plus, y_minus, and required binaries.
    beta_slice = slice(0, p)
    x_slice = slice(p, p + n)
    y_plus_slice = slice(p + n, p + 2 * n)
    y_minus_slice = slice(p + 2 * n, p + 3 * n)
    binary_slice = slice(p + 3 * n, p + 3 * n + binary_count)
    total = p + 3 * n + binary_count

    c = np.zeros(total)
    c[x_slice] = -w1 * scale * DP / oracle_denominator
    c[y_plus_slice] = surplus_cost
    c[y_minus_slice] = shortage_cost

    Aeq = sparse.lil_matrix((2 * n, total), dtype=float)
    # x = X beta
    Aeq[:n, beta_slice] = -X
    Aeq[:n, x_slice] = sparse.eye(n)
    # x + y_plus - y_minus = S
    Aeq[n:, x_slice] = sparse.eye(n)
    Aeq[n:, y_plus_slice] = sparse.eye(n)
    Aeq[n:, y_minus_slice] = -sparse.eye(n)
    beq = np.concatenate([np.zeros(n), actual])

    constraints = [LinearConstraint(Aeq.tocsr(), beq, beq)]
    if binary_count:
        rows = np.concatenate([
            np.arange(binary_count),
            binary_count + np.arange(binary_count),
            np.arange(binary_count),
            binary_count + np.arange(binary_count),
        ])
        columns = np.concatenate([
            y_plus_slice.start + binary_observations,
            y_minus_slice.start + binary_observations,
            binary_slice.start + np.arange(binary_count),
            binary_slice.start + np.arange(binary_count),
        ])
        values = np.concatenate([
            np.ones(binary_count),
            np.ones(binary_count),
            np.ones(binary_count),
            -np.ones(binary_count),
        ])
        Aub = sparse.csr_matrix((values, (rows, columns)), shape=(2 * binary_count, total))
        # y_plus <= 1-z and y_minus <= z for the required rows.
        bub = np.concatenate([np.ones(binary_count), np.zeros(binary_count)])
        constraints.append(
            LinearConstraint(Aub, np.full(2 * binary_count, -np.inf), bub)
        )

    lower = np.concatenate([np.full(p, -np.inf), np.zeros(3 * n + binary_count)])
    upper = np.concatenate([np.full(p, np.inf), np.ones(3 * n + binary_count)])
    integrality = np.zeros(total, dtype=int)
    integrality[binary_slice] = 1
    result = milp(
        c=c,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=constraints,
        options={"disp": False, "mip_rel_gap": 1e-9},
    )
    if not result.success or result.status != 0:
        raise SolverFailure(model_name, result.status, result.message)

    coefficients = np.asarray(result.x[beta_slice], dtype=float)
    fitted = X @ coefficients
    _validate_fitted_bounds(fitted, model_name)
    fitted_projected = np.minimum(1.0, np.maximum(0.0, fitted))
    objective = combined_training_objective(
        actual, fitted_projected, DP, RP, w1=w1, w2=w2
    )
    surplus = np.asarray(result.x[y_plus_slice])
    shortage = np.asarray(result.x[y_minus_slice])
    if binary_count and np.any(
        (surplus[binary_observations] > BOUNDARY_TOL)
        & (shortage[binary_observations] > BOUNDARY_TOL)
    ):
        raise SolverFailure(model_name, "POSTCHECK", "binary-row surplus and shortage both positive")
    if np.max(np.abs(np.asarray(result.x[x_slice]) - fitted)) > 1e-6:
        raise SolverFailure(model_name, "POSTCHECK", "commitment != X beta")
    # result.fun omits the constant W1 * oracle/oracle = W1.
    if not np.isclose(float(result.fun) + w1, objective, rtol=1e-7, atol=1e-7):
        raise SolverFailure(model_name, "POSTCHECK", "reduced MILP objective mismatch")
    mip_gap = getattr(result, "mip_gap", None)
    if mip_gap is not None and not np.isfinite(mip_gap):
        mip_gap = None
    return LinearFit(
        coefficients=coefficients,
        fitted_raw=fitted,
        objective_value=objective,
        solver_status=int(result.status),
        solver_message=str(result.message),
        certified_optimal=True,
        formulation="exact_sign_reduced_complementarity_milp",
        mip_gap=None if mip_gap is None else float(mip_gap),
        binary_variable_count=binary_count,
        unreduced_binary_variable_count=n,
    )


def build_ar_lag_design(history_daily, train_daily) -> tuple[np.ndarray, np.ndarray]:
    history = np.asarray(history_daily, dtype=float)
    train = np.asarray(train_daily, dtype=float)
    if history.ndim != 2 or train.ndim != 2 or history.shape[1:] != (12,) or train.shape[1:] != (12,):
        raise ModelError("history and train must be 2D arrays with 12 slots")
    if history.shape[0] < 1 or train.shape[0] < 1:
        raise ModelError("history and train must be nonempty")
    if not np.all(np.isfinite(history)) or not np.all(np.isfinite(train)):
        raise ModelError("history/train contains NaN/infinity")
    previous = np.vstack([history[-1], train[:-1]])
    # Most recently available slot first: previous-day [11,10,...,0].
    X = np.column_stack([np.ones(len(train)), previous[:, ::-1]])
    return X, train.copy()


def fit_conventional_ar(history_daily, train_daily) -> ARFit:
    X, targets = build_ar_lag_design(history_daily, train_daily)
    fits = [fit_bounded_lad(X, targets[:, h], model_name=f"conventional_ar_h{h}") for h in range(12)]
    return ARFit(
        coefficients_by_hour=np.vstack([f.coefficients for f in fits]),
        fitted_raw=np.column_stack([f.fitted_raw for f in fits]),
        objective_values_by_hour=np.asarray([f.objective_value for f in fits]),
        solver_statuses_by_hour=tuple(f.solver_status for f in fits),
        solver_messages_by_hour=tuple(f.solver_message for f in fits),
        certified_optimal=all(f.certified_optimal and f.solver_status == 0 for f in fits),
        formulation="12_target_hour_bounded_lad_prev_day_block",
        mip_gaps_by_hour=tuple(f.mip_gap for f in fits),
        binary_variable_counts_by_hour=tuple(f.binary_variable_count for f in fits),
        unreduced_binary_variable_counts_by_hour=tuple(
            f.unreduced_binary_variable_count for f in fits
        ),
    )


def fit_proposed_ar(history_daily, train_daily, train_da_daily, train_rt_daily,
                    *, w1: float = W1, w2: float = W2) -> ARFit:
    X, targets = build_ar_lag_design(history_daily, train_daily)
    da = np.asarray(train_da_daily, dtype=float)
    rt = np.asarray(train_rt_daily, dtype=float)
    if da.shape != targets.shape or rt.shape != targets.shape:
        raise ModelError("AR price matrices must match train targets")
    fits = [
        fit_proposed_milp(
            X, targets[:, h], da[:, h], rt[:, h], w1=w1, w2=w2,
            model_name=f"proposed_ar_h{h}",
        )
        for h in range(12)
    ]
    return ARFit(
        coefficients_by_hour=np.vstack([f.coefficients for f in fits]),
        fitted_raw=np.column_stack([f.fitted_raw for f in fits]),
        objective_values_by_hour=np.asarray([f.objective_value for f in fits]),
        solver_statuses_by_hour=tuple(f.solver_status for f in fits),
        solver_messages_by_hour=tuple(f.solver_message for f in fits),
        certified_optimal=all(f.certified_optimal and f.solver_status == 0 for f in fits),
        formulation="12_target_hour_dimensionless_objective_exact_milp",
        mip_gaps_by_hour=tuple(f.mip_gap for f in fits),
        binary_variable_counts_by_hour=tuple(f.binary_variable_count for f in fits),
        unreduced_binary_variable_counts_by_hour=tuple(
            f.unreduced_binary_variable_count for f in fits
        ),
    )


def predict_ar_rolling(coefficients_by_hour, previous_day_actual, test_actual) -> np.ndarray:
    coef = np.asarray(coefficients_by_hour, dtype=float)
    previous = np.asarray(previous_day_actual, dtype=float)
    actual = np.asarray(test_actual, dtype=float)
    if coef.shape != (12, 13) or previous.shape != (12,) or actual.ndim != 2 or actual.shape[1] != 12:
        raise ModelError("invalid AR prediction shapes")
    if not np.all(np.isfinite(coef)) or not np.all(np.isfinite(previous)) or not np.all(np.isfinite(actual)):
        raise ModelError("AR prediction inputs contain NaN/infinity")
    raw = np.empty_like(actual, dtype=float)
    for day in range(len(actual)):
        feature = np.concatenate([[1.0], previous[::-1]])
        raw[day] = coef @ feature
        previous = actual[day].copy()  # observed only after that day
    return raw


def fit_conventional_mlr(X, actual) -> LinearFit:
    return fit_bounded_lad(X, actual, model_name="conventional_mlr")


def fit_proposed_mlr(X, actual, DP, RP, *, w1: float = W1, w2: float = W2) -> LinearFit:
    return fit_proposed_milp(
        X, actual, DP, RP, w1=w1, w2=w2, model_name="proposed_mlr"
    )


def predict_linear_raw(X, coefficients) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if X.ndim != 2 or coefficients.ndim != 1 or X.shape[1] != coefficients.shape[0]:
        raise ModelError("linear prediction shape mismatch")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(coefficients)):
        raise ModelError("linear prediction input contains NaN/infinity")
    raw = X @ coefficients
    if not np.all(np.isfinite(raw)):
        raise ModelError("raw linear prediction is non-finite")
    return raw
