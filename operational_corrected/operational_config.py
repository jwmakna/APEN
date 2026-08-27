"""Experiment settings for the 300-day/100-day run."""
from __future__ import annotations

from dataclasses import asdict, dataclass


OFFICIAL_VERSION = "operational_corrected"
OBJECTIVE_DEFINITION = "corrected_dimensionless_training_objective_project_definition"
OFFICIAL_PREDICTION_VIEW = "feasible_unit_projection"

TRAIN_START = "2013-03-27"
TRAIN_END = "2014-01-20"
TRAIN_DAYS = 300
TRAIN_OBSERVATIONS = 3600

TEST_START = "2014-01-21"
TEST_END = "2014-04-30"
TEST_DAYS = 100
TEST_OBSERVATIONS = 1200

HISTORY_START = "2013-03-26"
DAYLIGHT_SLOTS_PER_DAY = 12
SOURCE_TIMEZONE_STATUS = "UNKNOWN_FROM_SOURCE_FILE"
ASSUMED_SOURCE_TIMEZONE = "UTC"
LOCAL_TIMEZONE = "Australia/Sydney"
LOCAL_HOUR_START = 9
LOCAL_HOUR_END = 21
DAYLIGHT_LOCAL_HOURS = tuple(range(LOCAL_HOUR_START, LOCAL_HOUR_END))
SYNTHETIC_RT_COUNT = 0
ZONE_ID = 1

CAPACITY_MW = 30.0
DURATION_HOURS = 1.0
PENALTY_RATE = 0.5
W1 = 1.0
W2 = 20.0

BOUNDARY_TOL = 1e-8
DENOMINATOR_TOL = 1e-9
SOLVER_TOL = 1e-7

RAW_PREDICTION_SOURCES = (
    "native_raw_unclipped",
    "reconstructed_pre_native_projection",
)

LEGACY_STATUS = "UNAPPROVED_SUPERSEDED"
LEGACY_OFFICIAL_EXECUTION_USE = "FORBIDDEN"
LEGACY_SUPERSEDED_BY = OFFICIAL_VERSION


@dataclass(frozen=True)
class DatasetPolicy:
    train_start: str = TRAIN_START
    train_end: str = TRAIN_END
    train_days: int = TRAIN_DAYS
    train_observations: int = TRAIN_OBSERVATIONS
    test_start: str = TEST_START
    test_end: str = TEST_END
    test_days: int = TEST_DAYS
    test_observations: int = TEST_OBSERVATIONS
    daylight_slots_per_day: int = DAYLIGHT_SLOTS_PER_DAY
    local_hour_start: int = LOCAL_HOUR_START
    local_hour_end_exclusive: int = LOCAL_HOUR_END
    synthetic_rt_count: int = SYNTHETIC_RT_COUNT
    requires_observed_rt: bool = True
    source_timezone_status: str = SOURCE_TIMEZONE_STATUS
    timezone_policy: str = "step1_naive_as_utc_to_australia_sydney"
    timezone_policy_status: str = "reproduction_assumption"


OFFICIAL_DATASET_POLICY = DatasetPolicy()


def validate_official_dataset_policy(policy: DatasetPolicy) -> None:
    """Check that the fixed data split was not changed accidentally."""
    if type(policy) is not DatasetPolicy:
        raise TypeError("policy must be exactly DatasetPolicy")
    expected = asdict(OFFICIAL_DATASET_POLICY)
    actual = asdict(policy)
    for field, expected_value in expected.items():
        actual_value = actual[field]
        if type(actual_value) is not type(expected_value):
            raise TypeError(
                f"{field} type must be {type(expected_value).__name__}, "
                f"got {type(actual_value).__name__}"
            )
        if actual_value != expected_value:
            raise ValueError(f"{field} must equal {expected_value!r}, got {actual_value!r}")
