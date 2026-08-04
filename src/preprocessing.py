"""
Preprocessing and time-aware feature engineering for KW pipe-break classification.

Phase 2 scope
-------------
- Consume historical snapshots from Phase 1 (breaks-only physical attributes).
- Build model features available at snapshot date t:
    material, diameter_mm, install_year, age_years,
    prior_break_count, years_since_last_break
- Enforce temporal leakage protections (no post-break / repair fields).
- Provide a strict time-based train/test split (no random splitting).

Length join and training/API/UI wiring are intentionally out of scope here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from src.config import (
        TEMPORAL_SPLIT_DATE,
        processed_snapshots_path,
        raw_breaks_path,
    )
    from src.labeling import generate_snapshot_dataset, load_break_events
    from src.schema import (
        CATEGORICAL_COLUMNS,
        FEATURE_COLUMNS,
        LEAKAGE_FORBIDDEN_COLUMNS,
        META_COLUMNS,
        NUMERIC_COLUMNS,
        TARGET_COLUMN,
    )
except ModuleNotFoundError:
    from config import (
        TEMPORAL_SPLIT_DATE,
        processed_snapshots_path,
        raw_breaks_path,
    )
    from labeling import generate_snapshot_dataset, load_break_events
    from schema import (
        CATEGORICAL_COLUMNS,
        FEATURE_COLUMNS,
        LEAKAGE_FORBIDDEN_COLUMNS,
        META_COLUMNS,
        NUMERIC_COLUMNS,
        TARGET_COLUMN,
    )

# Public aliases kept for later training-phase imports.
DEFAULT_TARGET_COLUMN = TARGET_COLUMN
DATA_COLUMNS = FEATURE_COLUMNS

# Explicit denylist mirrored at the preprocessing boundary.
PREDICTION_EXCLUDED_COLUMNS = list(LEAKAGE_FORBIDDEN_COLUMNS) + [
    "asset_id",
    "snapshot_date",
    "street",
    "snapshot_origin",
    "horizon_years",
    "length_m",  # deferred; must not sneak in until a later phase
]


def resolve_dataset_path(csv_path=None) -> Path:
    dataset_path = Path(csv_path) if csv_path is not None else Path(processed_snapshots_path)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parent.parent / dataset_path
    return dataset_path


def _normalize_material(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.upper()
    cleaned = cleaned.replace(
        {
            "": "UNKNOWN",
            "NAN": "UNKNOWN",
            "NONE": "UNKNOWN",
            "NULL": "UNKNOWN",
            "XXX": "UNKNOWN",
        }
    )
    return cleaned.fillna("UNKNOWN")


def assert_no_temporal_leakage(df: pd.DataFrame) -> None:
    """Fail fast if post-break / forbidden columns are present in a modeling frame."""
    lower_map = {c.lower(): c for c in df.columns}
    leaked = [lower_map[c.lower()] for c in LEAKAGE_FORBIDDEN_COLUMNS if c.lower() in lower_map]
    if leaked:
        raise ValueError(
            "Temporal leakage risk: forbidden post-break columns present in frame: "
            f"{leaked}"
        )

    unexpected = [c for c in df.columns if c not in set(FEATURE_COLUMNS + [TARGET_COLUMN] + META_COLUMNS)]
    # Allow only known contract columns through this gate.
    if unexpected:
        raise ValueError(
            "Unexpected columns outside the Phase-2 data contract: "
            f"{unexpected}"
        )


def _years_since_last_break_for_row(
    snapshot_date: pd.Timestamp,
    prior_dates: list[pd.Timestamp],
) -> float:
    """Return years since the latest break strictly before snapshot_date, else NaN."""
    prior = [d for d in prior_dates if d < snapshot_date]
    if not prior:
        return np.nan
    last_break = max(prior)
    return float((snapshot_date - last_break).days) / 365.25


def add_years_since_last_break(
    snapshots: pd.DataFrame,
    events: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Attach years_since_last_break using only breaks with incident_date < snapshot_date.

    This recomputes the feature from raw events so it cannot accidentally use the
    label-defining future break.
    """
    if events is None:
        events = load_break_events(raw_breaks_path)

    breaks_by_asset = {
        str(asset_id): sorted(group["incident_date"].tolist())
        for asset_id, group in events.groupby("asset_id", sort=False)
    }

    out = snapshots.copy()
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"]).dt.normalize()
    out["asset_id"] = out["asset_id"].astype(str)

    values = []
    for asset_id, snapshot_date in zip(out["asset_id"], out["snapshot_date"]):
        values.append(
            _years_since_last_break_for_row(
                snapshot_date,
                breaks_by_asset.get(str(asset_id), []),
            )
        )
    out["years_since_last_break"] = values
    return out


def validate_time_aware_features(df: pd.DataFrame) -> None:
    """Sanity-check consistency of prior_break_count vs years_since_last_break."""
    if "prior_break_count" not in df.columns or "years_since_last_break" not in df.columns:
        raise ValueError("Missing time-aware features required for Phase 2.")

    zero_prior = df["prior_break_count"].fillna(0).astype(int) == 0
    if zero_prior.any():
        bad = df.loc[zero_prior, "years_since_last_break"].notna().any()
        if bad:
            raise ValueError(
                "Leakage/consistency error: years_since_last_break is set while "
                "prior_break_count == 0."
            )

    positive_prior = df["prior_break_count"].fillna(0).astype(int) > 0
    if positive_prior.any():
        missing = df.loc[positive_prior, "years_since_last_break"].isna().any()
        if missing:
            raise ValueError(
                "Consistency error: prior_break_count > 0 but years_since_last_break "
                "is missing."
            )


def load_snapshot_data(csv_path=None, regenerate_if_missing: bool = True) -> pd.DataFrame:
    """Load Phase-1 snapshots, optionally regenerating them if the file is absent."""
    dataset_path = resolve_dataset_path(csv_path)
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
    elif regenerate_if_missing:
        df = generate_snapshot_dataset()
    else:
        raise FileNotFoundError(f"Snapshot dataset not found: {dataset_path}")

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"]).dt.normalize()
    df["asset_id"] = df["asset_id"].astype(str)
    return df


def engineer_pipe_features(snapshots: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build the cleaned modeling table with Phase-2 features only.

    Returns a frame containing META_COLUMNS + FEATURE_COLUMNS + TARGET_COLUMN.
    """
    if snapshots is None:
        snapshots = load_snapshot_data()

    df = add_years_since_last_break(snapshots)

    # Physical / categorical cleaning
    df["material"] = _normalize_material(df["material"])
    df["diameter_mm"] = pd.to_numeric(df["diameter_mm"], errors="coerce")
    df["install_year"] = pd.to_numeric(df["install_year"], errors="coerce")
    df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
    df["prior_break_count"] = pd.to_numeric(df["prior_break_count"], errors="coerce").fillna(0).astype(int)
    df["years_since_last_break"] = pd.to_numeric(df["years_since_last_break"], errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype("Int64")

    # Drop rows that cannot support the physical feature set.
    required = ["material", "diameter_mm", "install_year", "age_years", TARGET_COLUMN, "snapshot_date"]
    df = df.dropna(subset=required).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Recompute age from install_year at snapshot_date to keep the feature
    # strictly time-consistent (install_year is known at t; age is derived).
    df["age_years"] = df["snapshot_date"].dt.year - df["install_year"]

    keep_cols = [c for c in META_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns]
    modeled = df[keep_cols].copy()

    assert_no_temporal_leakage(modeled)
    validate_time_aware_features(modeled)

    # Final feature-set check: only approved modeling columns will be exposed via X.
    missing_features = [c for c in FEATURE_COLUMNS if c not in modeled.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    return modeled.sort_values(["snapshot_date", "asset_id"]).reset_index(drop=True)


def temporal_train_test_split(
    df: pd.DataFrame,
    cutoff: Optional[str] = None,
):
    """
    Strict time-based split on snapshot_date.

    Train: snapshot_date < cutoff
    Test:  snapshot_date >= cutoff

    Both partitions must contain at least one positive and one negative label.
    """
    split_date = pd.Timestamp(cutoff or TEMPORAL_SPLIT_DATE)
    if "snapshot_date" not in df.columns:
        raise ValueError("temporal_train_test_split requires snapshot_date.")
    if TARGET_COLUMN not in df.columns:
        raise ValueError("temporal_train_test_split requires the target column.")

    train_df = df[df["snapshot_date"] < split_date].copy()
    test_df = df[df["snapshot_date"] >= split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Temporal split at {split_date.date()} produced an empty train or test set "
            f"(train={len(train_df)}, test={len(test_df)})."
        )

    for name, part in (("train", train_df), ("test", test_df)):
        classes = set(part[TARGET_COLUMN].dropna().astype(int).unique())
        if not {0, 1}.issubset(classes):
            raise ValueError(
                f"Temporal split at {split_date.date()} produced a single-class {name} "
                f"set (classes={sorted(classes)}, n={len(part)}). "
                "Adjust TEMPORAL_SPLIT_DATE or negative snapshot density."
            )

    return train_df, test_df


def _split_xy(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    # Defense in depth: never return meta/target leakage through X.
    assert_no_temporal_leakage(pd.concat([X, y.rename(TARGET_COLUMN)], axis=1))
    return X, y


def prepare_pipe_break_data(
    csv_path=None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    cutoff: Optional[str] = None,
):
    """
    End-to-end Phase-2 preparation:

    returns X_train, X_test, y_train, y_test, cleaned_df
    using a strict temporal cutoff (never random splitting).
    """
    if target_column != TARGET_COLUMN:
        raise ValueError(
            f"Unsupported target_column={target_column!r}; "
            f"Phase 2 classification target is {TARGET_COLUMN!r}."
        )

    cleaned_df = engineer_pipe_features(load_snapshot_data(csv_path))
    train_df, test_df = temporal_train_test_split(cleaned_df, cutoff=cutoff)

    X_train, y_train = _split_xy(train_df)
    X_test, y_test = _split_xy(test_df)
    return X_train, X_test, y_train, y_test, cleaned_df


# Backward-compatible name used by the legacy training entrypoint.
# Phase 3 will switch train.py fully to the pipe-break API.
def clean_airplane_data(*args, **kwargs):
    raise RuntimeError(
        "Aircraft preprocessing has been removed. "
        "Use prepare_pipe_break_data() / engineer_pipe_features() for the "
        "KW water-main classification pipeline."
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, cleaned = prepare_pipe_break_data()
    print("Feature columns:", FEATURE_COLUMNS)
    print("Temporal cutoff:", TEMPORAL_SPLIT_DATE)
    print(f"cleaned rows: {len(cleaned)}")
    print(f"train: {len(X_train)} | test: {len(X_test)}")
    print("train label balance:\n", y_train.value_counts(normalize=True).round(3))
    print("test label balance:\n", y_test.value_counts(normalize=True).round(3))
    print("X_train dtypes:\n", X_train.dtypes)
    print(
        "years_since_last_break null rate (train):",
        float(X_train["years_since_last_break"].isna().mean()),
    )
