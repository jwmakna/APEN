"""Read and align the solar, weather, and price data."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from operational_config import (
    DAYLIGHT_SLOTS_PER_DAY,
    HISTORY_START,
    LOCAL_HOUR_END,
    LOCAL_HOUR_START,
    LOCAL_TIMEZONE,
    TEST_DAYS,
    TEST_END,
    TEST_START,
    TRAIN_DAYS,
    TRAIN_END,
    TRAIN_START,
    ZONE_ID,
)


@dataclass
class ScalerMetadata:
    dSSRD_mean: float
    dSSRD_std: float
    dTSR_mean: float
    dTSR_std: float

    def to_dict(self):
        return asdict(self)


@dataclass
class OperationalDataset:
    history: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame


def _check_columns(frame, columns, name):
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def read_predictors_source(path):
    predictors = pd.read_csv(path)
    required = ["ZONEID", "TIMESTAMP", "VAR169", "VAR178", "POWER"]
    _check_columns(predictors, required, "predictors file")
    predictors["TIMESTAMP"] = pd.to_datetime(
        predictors["TIMESTAMP"], format="%Y%m%d %H:%M"
    ).dt.tz_localize("UTC")
    return predictors.sort_values(["ZONEID", "TIMESTAMP"]).reset_index(drop=True)


def deaccumulate_source(predictors):
    data = predictors.copy()
    data["forecast_run_date"] = (
        data["TIMESTAMP"] - pd.Timedelta(hours=1)
    ).dt.normalize()
    groups = data.groupby(["ZONEID", "forecast_run_date"], sort=False)
    data["forecast_step"] = groups.cumcount() + 1

    for source, output in (("VAR169", "dSSRD"), ("VAR178", "dTSR")):
        increments = groups[source].diff()
        first_step = data["forecast_step"].eq(1)
        increments.loc[first_step] = data.loc[first_step, source]
        data[output] = increments
    return data


def _read_price(path, value_column, output_column):
    prices = pd.read_csv(path, usecols=["TIMESTAMP", value_column])
    prices["timestamp"] = pd.to_datetime(prices.pop("TIMESTAMP")).dt.tz_localize("UTC")
    return prices.rename(columns={value_column: output_column})


def _select_period(frame, start, end):
    start = pd.Timestamp(start).date()
    end = pd.Timestamp(end).date()
    return frame[frame["local_date"].between(start, end)].copy()


def _check_split(frame, expected_days, name):
    expected_rows = expected_days * DAYLIGHT_SLOTS_PER_DAY
    if len(frame) != expected_rows or frame["local_date"].nunique() != expected_days:
        raise ValueError(
            f"{name} must contain {expected_days} complete days "
            f"({expected_rows} rows)"
        )
    if frame[["solar_power", "da_price", "rt_price", "dSSRD", "dTSR"]].isna().any().any():
        raise ValueError(f"{name} contains missing values")


def build_operational_dataset(
    predictors_path: str | Path,
    da_path: str | Path,
    rt_path: str | Path,
    zone_id: int = ZONE_ID,
):
    predictors = deaccumulate_source(read_predictors_source(predictors_path))
    data = predictors[predictors["ZONEID"].eq(zone_id)].copy()
    data = data.rename(columns={
        "TIMESTAMP": "timestamp",
        "POWER": "solar_power",
        "VAR169": "ssrd_accum",
        "VAR178": "tsr_accum",
    })

    data["timestamp_local"] = data["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)
    data["local_date"] = data["timestamp_local"].dt.date
    data["local_hour"] = data["timestamp_local"].dt.hour
    data["hour_idx"] = data["local_hour"] - LOCAL_HOUR_START
    daylight = data[
        data["local_hour"].between(LOCAL_HOUR_START, LOCAL_HOUR_END - 1)
    ].sort_values("timestamp")

    day_ahead = _read_price(da_path, "DA_LMP", "da_price")
    real_time = _read_price(rt_path, "RT_LMP", "rt_price")
    merged = daylight.merge(day_ahead, on="timestamp", how="left")
    merged = merged.merge(real_time, on="timestamp", how="left")

    columns = [
        "timestamp",
        "timestamp_local",
        "local_date",
        "local_hour",
        "hour_idx",
        "forecast_run_date",
        "forecast_step",
        "solar_power",
        "da_price",
        "rt_price",
        "ssrd_accum",
        "tsr_accum",
        "dSSRD",
        "dTSR",
    ]
    history_columns = [column for column in columns if column not in {"da_price", "rt_price"}]
    history = _select_period(daylight[history_columns], HISTORY_START, HISTORY_START)
    train = _select_period(merged[columns], TRAIN_START, TRAIN_END)
    test = _select_period(merged[columns], TEST_START, TEST_END)

    _check_split(train, TRAIN_DAYS, "training data")
    _check_split(test, TEST_DAYS, "test data")
    return OperationalDataset(history=history, train=train, test=test)


def fit_train_scaler(train):
    mean = train[["dSSRD", "dTSR"]].mean()
    std = train[["dSSRD", "dTSR"]].std(ddof=0)
    return ScalerMetadata(
        dSSRD_mean=float(mean["dSSRD"]),
        dSSRD_std=float(std["dSSRD"]),
        dTSR_mean=float(mean["dTSR"]),
        dTSR_std=float(std["dTSR"]),
    )


def build_mlr_design(frame, scaler):
    hours = frame["hour_idx"].to_numpy(dtype=int)
    z_ssrd = (frame["dSSRD"].to_numpy() - scaler.dSSRD_mean) / scaler.dSSRD_std
    z_tsr = (frame["dTSR"].to_numpy() - scaler.dTSR_mean) / scaler.dTSR_std
    hour_dummies = np.column_stack([hours == hour for hour in range(1, 12)])
    X = np.column_stack([np.ones(len(frame)), z_ssrd, z_tsr, hour_dummies])
    names = ("intercept", "z_dSSRD", "z_dTSR") + tuple(
        f"hour_{hour}" for hour in range(1, 12)
    )
    return X, names


def to_daily_matrix(frame, value_column):
    ordered = frame.sort_values(["local_date", "hour_idx"])
    dates = ordered["local_date"].drop_duplicates().to_numpy()
    matrix = ordered[value_column].to_numpy(dtype=float).reshape(-1, 12)
    return dates, matrix
