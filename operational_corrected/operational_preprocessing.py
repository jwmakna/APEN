"""Load and align the solar, weather, and price data."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from operational_config import (
    DAYLIGHT_SLOTS_PER_DAY,
    DAYLIGHT_LOCAL_HOURS,
    HISTORY_START,
    LOCAL_HOUR_END,
    LOCAL_HOUR_START,
    LOCAL_TIMEZONE,
    OFFICIAL_DATASET_POLICY,
    TEST_DAYS,
    TEST_END,
    TEST_OBSERVATIONS,
    TEST_START,
    TRAIN_DAYS,
    TRAIN_END,
    TRAIN_OBSERVATIONS,
    TRAIN_START,
    ZONE_ID,
    validate_official_dataset_policy,
)


RAW_REQUIRED_COLUMNS = (
    "ZONEID", "TIMESTAMP", "VAR169", "VAR178", "POWER",
)
ACCUMULATED_TO_INCREMENT = {"VAR169": "dSSRD", "VAR178": "dTSR"}
OFFICIAL_COLUMNS = (
    "timestamp", "timestamp_local", "local_date", "local_hour", "hour_idx",
    "forecast_run_date", "forecast_step",
    "solar_power", "da_price", "rt_price", "ssrd_accum", "tsr_accum",
    "dSSRD", "dTSR",
)
HISTORY_COLUMNS = tuple(column for column in OFFICIAL_COLUMNS if column not in ("da_price", "rt_price"))


class PreprocessingError(ValueError):
    pass


@dataclass(frozen=True)
class ScalerMetadata:
    dSSRD_mean: float
    dSSRD_std: float
    dTSR_mean: float
    dTSR_std: float
    ddof: int = 0
    fitted_on: str = "train_only"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OperationalDataset:
    history: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    source_sequence_audit: dict
    alignment_audit: dict


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise PreprocessingError(f"{name} missing columns: {missing}")


def _require_finite(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    values = frame[list(columns)].to_numpy(dtype=float)
    if values.size == 0:
        raise PreprocessingError(f"{name} is empty")
    if not np.all(np.isfinite(values)):
        raise PreprocessingError(f"{name} contains NaN or infinity")


def read_predictors_source(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    _require_columns(raw, RAW_REQUIRED_COLUMNS, "predictors source")
    raw = raw.copy()
    parsed = pd.to_datetime(raw["TIMESTAMP"], format="%Y%m%d %H:%M", errors="raise")
    if parsed.dt.tz is not None:
        raise PreprocessingError("predictors timestamps must be source-naive before UTC localization")
    raw["TIMESTAMP"] = parsed.dt.tz_localize("UTC")
    if raw[list(RAW_REQUIRED_COLUMNS)].isna().any().any():
        raise PreprocessingError("predictors source contains missing required values")
    if raw.duplicated(["ZONEID", "TIMESTAMP"]).any():
        raise PreprocessingError("duplicate (ZONEID, TIMESTAMP) rows")
    return raw


def validate_source_sequence(raw: pd.DataFrame) -> dict:
    """Check the 01,...,23,00 forecast blocks before filtering."""
    _require_columns(raw, RAW_REQUIRED_COLUMNS, "predictors source")
    if len(raw) == 0:
        raise PreprocessingError("predictors source is empty")

    zone_audit: dict[int, dict] = {}
    expected_cycle = np.asarray(list(range(1, 24)) + [0], dtype=int)
    for zone, zone_df in raw.groupby("ZONEID", sort=True):
        g = zone_df.sort_values("TIMESTAMP").reset_index(drop=True)
        elapsed = g["TIMESTAMP"].diff().dropna().dt.total_seconds().to_numpy() / 3600.0
        if elapsed.size and not np.all(elapsed == 1.0):
            raise PreprocessingError(f"zone {zone} is not an uninterrupted hourly sequence")
        run_date = (g["TIMESTAMP"] - pd.Timedelta(hours=1)).dt.normalize()
        sizes = g.groupby(run_date, sort=False).size()
        if not np.all(sizes.to_numpy() == 24):
            raise PreprocessingError(f"zone {zone} contains non-24-row accumulation groups")
        observed_cycle = g["TIMESTAMP"].dt.hour.to_numpy()
        tiled_cycle = np.tile(expected_cycle, len(sizes))
        if not np.array_equal(observed_cycle, tiled_cycle):
            raise PreprocessingError(f"zone {zone} does not follow 01,...,23,00 blocks")
        zone_audit[int(zone)] = {
            "rows": int(len(g)),
            "groups": int(len(sizes)),
            "start": g["TIMESTAMP"].iloc[0].isoformat(),
            "end": g["TIMESTAMP"].iloc[-1].isoformat(),
            "group_size": 24,
            "hour_cycle": "01,...,23,00",
        }

    return {
        "group_definition": "(ZONEID, normalize(TIMESTAMP - 1 hour))",
        "group_status": "STRUCTURALLY_IDENTIFIED_FROM_OFFICIAL_SOURCE_SEQUENCE",
        "zones": zone_audit,
    }


def deaccumulate_source(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Convert accumulated SSRD/TSR to hourly increments within each run."""
    audit = validate_source_sequence(raw)
    out = raw.sort_values(["ZONEID", "TIMESTAMP"]).copy()
    out["forecast_run_date"] = (out["TIMESTAMP"] - pd.Timedelta(hours=1)).dt.normalize()
    out["forecast_step"] = out.groupby(["ZONEID", "forecast_run_date"], sort=False).cumcount() + 1

    group_keys = [out["ZONEID"], out["forecast_run_date"]]
    for source_col, increment_col in ACCUMULATED_TO_INCREMENT.items():
        diff = out.groupby(group_keys, sort=False)[source_col].diff()
        first = out["forecast_step"].eq(1)
        diff.loc[first] = out.loc[first, source_col]
        if not np.all(np.isfinite(diff.to_numpy(dtype=float))):
            raise PreprocessingError(f"non-finite increment generated for {source_col}")
        out[increment_col] = diff.to_numpy(dtype=float)
    return out, audit


def _read_price(path: str | Path, value_column: str, output_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_columns(df, ("TIMESTAMP", value_column), f"price file {path}")
    df = df[["TIMESTAMP", value_column]].copy()
    parsed = pd.to_datetime(df["TIMESTAMP"], errors="raise")
    if parsed.dt.tz is not None:
        raise PreprocessingError(f"price timestamps in {path} must be source-naive")
    df["TIMESTAMP"] = parsed.dt.tz_localize("UTC")
    df = df.rename(columns={"TIMESTAMP": "timestamp", value_column: output_column})
    if df["timestamp"].duplicated().any():
        raise PreprocessingError(f"duplicate timestamp in {path}")
    _require_finite(df, (output_column,), str(path))
    return df


def _validate_fixed_split(
    frame: pd.DataFrame,
    name: str,
    days: int,
    observations: int,
    expected_start: str,
    expected_end: str,
) -> None:
    if len(frame) != observations:
        raise PreprocessingError(f"{name} rows must be {observations}, got {len(frame)}")
    if frame["local_date"].nunique() != days:
        raise PreprocessingError(f"{name} days must be {days}, got {frame['local_date'].nunique()}")
    if frame["local_date"].min().isoformat() != expected_start or frame["local_date"].max().isoformat() != expected_end:
        raise PreprocessingError(f"{name} local-date bounds do not match the canonical split")
    counts = frame.groupby("local_date", sort=False).size().to_numpy()
    if not np.all(counts == DAYLIGHT_SLOTS_PER_DAY):
        raise PreprocessingError(f"{name} must contain exactly 12 rows per day")
    if frame["timestamp"].duplicated().any():
        raise PreprocessingError(f"{name} contains duplicate timestamps")
    for _, day in frame.groupby("local_date", sort=False):
        if not np.array_equal(day["local_hour"].to_numpy(dtype=int), np.arange(9, 21)):
            raise PreprocessingError(f"{name} must contain local hours 09,...,20 in order")
        if not np.array_equal(day["hour_idx"].to_numpy(dtype=int), np.arange(12)):
            raise PreprocessingError(f"{name} must contain hour_idx 0,...,11 in order")
    expected_local = frame["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)
    if not np.array_equal(frame["timestamp_local"].astype(str), expected_local.astype(str)):
        raise PreprocessingError(f"{name} timestamp_local does not match UTC-to-Sydney conversion")
    _require_finite(
        frame,
        ("solar_power", "da_price", "rt_price", "dSSRD", "dTSR"),
        name,
    )


def _validate_history(frame: pd.DataFrame) -> None:
    if len(frame) != DAYLIGHT_SLOTS_PER_DAY or frame["local_date"].nunique() != 1:
        raise PreprocessingError("solar history must contain one complete 12-slot local day")
    if frame["local_date"].iloc[0].isoformat() != HISTORY_START:
        raise PreprocessingError(f"solar history must be local date {HISTORY_START}")
    if not np.array_equal(frame["local_hour"].to_numpy(dtype=int), np.arange(9, 21)):
        raise PreprocessingError("solar history must contain local hours 09,...,20")
    _require_finite(frame, ("solar_power", "dSSRD", "dTSR"), "solar history")


def build_operational_dataset(
    predictors_path: str | Path,
    da_path: str | Path,
    rt_path: str | Path,
    *,
    zone_id: int = ZONE_ID,
) -> OperationalDataset:
    if type(zone_id) is not int:
        raise TypeError("zone_id must be an exact int")
    validate_official_dataset_policy(OFFICIAL_DATASET_POLICY)

    raw = read_predictors_source(predictors_path)
    deaccumulated, audit = deaccumulate_source(raw)
    if zone_id not in set(deaccumulated["ZONEID"].astype(int)):
        raise PreprocessingError(f"zone {zone_id} is absent")

    zone = deaccumulated[deaccumulated["ZONEID"] == zone_id].copy()
    zone = zone.rename(
        columns={
            "TIMESTAMP": "timestamp",
            "POWER": "solar_power",
            "VAR169": "ssrd_accum",
            "VAR178": "tsr_accum",
        }
    )
    zone["timestamp_local"] = zone["timestamp"].dt.tz_convert(LOCAL_TIMEZONE)
    zone["local_date"] = zone["timestamp_local"].dt.date
    zone["local_hour"] = zone["timestamp_local"].dt.hour.astype(int)
    zone["hour_idx"] = zone["local_hour"] - LOCAL_HOUR_START
    daylight = zone[
        zone["local_hour"].ge(LOCAL_HOUR_START)
        & zone["local_hour"].lt(LOCAL_HOUR_END)
    ].sort_values("timestamp").copy()

    da = _read_price(da_path, "DA_LMP", "da_price")
    rt = _read_price(rt_path, "RT_LMP", "rt_price")
    # Select daylight hours by Sydney time, then join sources on UTC time.
    merged = daylight.merge(da, on="timestamp", how="left", validate="one_to_one")
    merged = merged.merge(rt, on="timestamp", how="left", validate="one_to_one")

    history_date = pd.Timestamp(HISTORY_START).date()
    train_start = pd.Timestamp(TRAIN_START).date()
    train_end = pd.Timestamp(TRAIN_END).date()
    test_start = pd.Timestamp(TEST_START).date()
    test_end = pd.Timestamp(TEST_END).date()

    # AR needs one complete POWER day before the training period.
    history = daylight[daylight["local_date"].eq(history_date)][list(HISTORY_COLUMNS)].copy()
    official = merged[list(OFFICIAL_COLUMNS)].copy()
    train = official[official["local_date"].between(train_start, train_end)].copy()
    test = official[official["local_date"].between(test_start, test_end)].copy()

    _validate_history(history)
    _validate_fixed_split(
        train, "train", TRAIN_DAYS, TRAIN_OBSERVATIONS, TRAIN_START, TRAIN_END
    )
    _validate_fixed_split(
        test, "test", TEST_DAYS, TEST_OBSERVATIONS, TEST_START, TEST_END
    )
    if not np.all((train["solar_power"] >= 0) & (train["solar_power"] <= 1)):
        raise PreprocessingError("train POWER is outside [0,1]")
    if not np.all((test["solar_power"] >= 0) & (test["solar_power"] <= 1)):
        raise PreprocessingError("test POWER is outside [0,1]")

    def _mapping(local_date: str, local_hour: int) -> dict:
        target_date = pd.Timestamp(local_date).date()
        row = daylight[
            daylight["local_date"].eq(target_date)
            & daylight["local_hour"].eq(local_hour)
        ]
        if len(row) != 1:
            raise PreprocessingError(f"missing unique mapping for {local_date} hour {local_hour}")
        item = row.iloc[0]
        return {
            "timestamp_utc": item["timestamp"].isoformat(),
            "timestamp_local": item["timestamp_local"].isoformat(),
            "local_date": item["local_date"].isoformat(),
            "local_hour": int(item["local_hour"]),
            "hour_idx": int(item["hour_idx"]),
        }

    history_join = merged[merged["local_date"].eq(history_date)]
    test_end_join = merged[merged["local_date"].eq(test_end)]
    alignment_audit = {
        "source_timestamp_interpretation": "NAIVE_AS_UTC_REPRODUCTION_ASSUMPTION",
        "target_timezone": LOCAL_TIMEZONE,
        "day_grouping": "SYDNEY_LOCAL_DATE",
        "daylight_local_hours": list(DAYLIGHT_LOCAL_HOURS),
        "price_join_key": "TIMEZONE_AWARE_UTC_TIMESTAMP",
        "history_local_date": HISTORY_START,
        "history_source_daylight_rows": int(len(history)),
        "history_complete_four_source_rows": int(
            history_join[["solar_power", "da_price", "rt_price", "dSSRD", "dTSR"]].notna().all(axis=1).sum()
        ),
        "test_end_local_date": TEST_END,
        "test_end_observed_rt_supported_rows": int(test_end_join["rt_price"].notna().sum()),
        "raw_da_first_observed_timestamp": da["timestamp"].min().isoformat(),
        "raw_da_last_observed_timestamp": da["timestamp"].max().isoformat(),
        "raw_rt_first_observed_timestamp": rt["timestamp"].min().isoformat(),
        "raw_rt_last_observed_timestamp": rt["timestamp"].max().isoformat(),
        "train_first_slot": _mapping(TRAIN_START, LOCAL_HOUR_START),
        "train_last_slot": _mapping(TRAIN_END, LOCAL_HOUR_END - 1),
        "test_first_slot": _mapping(TEST_START, LOCAL_HOUR_START),
        "test_last_slot": _mapping(TEST_END, LOCAL_HOUR_END - 1),
        "dst_before_example": _mapping("2014-04-05", LOCAL_HOUR_START),
        "dst_after_example": _mapping("2014-04-07", LOCAL_HOUR_START),
    }
    if alignment_audit["history_complete_four_source_rows"] != 10:
        raise PreprocessingError("history local day must have exactly 10 joined four-source rows")
    if alignment_audit["test_end_observed_rt_supported_rows"] != DAYLIGHT_SLOTS_PER_DAY:
        raise PreprocessingError("test end local day must have 12 observed RT-supported rows")
    return OperationalDataset(
        history=history,
        train=train,
        test=test,
        source_sequence_audit=audit,
        alignment_audit=alignment_audit,
    )


def fit_train_scaler(train: pd.DataFrame) -> ScalerMetadata:
    _require_columns(train, ("dSSRD", "dTSR"), "train")
    _require_finite(train, ("dSSRD", "dTSR"), "train")
    means = train[["dSSRD", "dTSR"]].mean(axis=0)
    stds = train[["dSSRD", "dTSR"]].std(axis=0, ddof=0)
    if not np.all(np.isfinite(stds.to_numpy())) or np.any(stds.to_numpy() <= 0):
        raise PreprocessingError("training standard deviation must be finite and positive")
    return ScalerMetadata(
        dSSRD_mean=float(means["dSSRD"]), dSSRD_std=float(stds["dSSRD"]),
        dTSR_mean=float(means["dTSR"]), dTSR_std=float(stds["dTSR"]),
    )


def build_mlr_design(frame: pd.DataFrame, scaler: ScalerMetadata) -> tuple[np.ndarray, tuple[str, ...]]:
    if type(scaler) is not ScalerMetadata:
        raise TypeError("scaler must be ScalerMetadata")
    _require_columns(frame, ("dSSRD", "dTSR", "hour_idx"), "MLR frame")
    _require_finite(frame, ("dSSRD", "dTSR", "hour_idx"), "MLR frame")
    hours = frame["hour_idx"].to_numpy(dtype=int)
    if not np.all((hours >= 0) & (hours < 12)):
        raise PreprocessingError("hour_idx must be in 0,...,11")
    z_ssrd = (frame["dSSRD"].to_numpy(dtype=float) - scaler.dSSRD_mean) / scaler.dSSRD_std
    z_tsr = (frame["dTSR"].to_numpy(dtype=float) - scaler.dTSR_mean) / scaler.dTSR_std
    dummies = np.column_stack([(hours == h).astype(float) for h in range(1, 12)])
    X = np.column_stack([np.ones(len(frame)), z_ssrd, z_tsr, dummies])
    names = ("intercept", "z_dSSRD", "z_dTSR") + tuple(f"hour_{h}" for h in range(1, 12))
    if X.shape[1] != 14 or len(names) != 14:
        raise AssertionError("pooled MLR design must contain 14 columns")
    if not np.all(np.isfinite(X)):
        raise PreprocessingError("MLR design contains NaN or infinity")
    return X, names


def to_daily_matrix(frame: pd.DataFrame, value_column: str) -> tuple[np.ndarray, np.ndarray]:
    _require_columns(frame, ("local_date", "hour_idx", value_column), "daily frame")
    dates = np.asarray(sorted(frame["local_date"].unique()))
    matrix = np.empty((len(dates), DAYLIGHT_SLOTS_PER_DAY), dtype=float)
    for i, date in enumerate(dates):
        day = frame[frame["local_date"] == date].sort_values("hour_idx")
        if not np.array_equal(day["hour_idx"].to_numpy(dtype=int), np.arange(12)):
            raise PreprocessingError(f"date {date} does not contain hour_idx 0,...,11")
        matrix[i] = day[value_column].to_numpy(dtype=float)
    if matrix.size == 0 or not np.all(np.isfinite(matrix)):
        raise PreprocessingError("daily matrix is empty or non-finite")
    return dates, matrix
