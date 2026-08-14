"""
Preprocessing and time-aware feature engineering for KW pipe-break classification.

Phase 2 scope
-------------
- Consume historical snapshots from Phase 1 (breaks-only physical attributes).
- Build model features available at snapshot date t:
    material, diameter_mm, install_year, age_years,
    prior_break_count, years_since_last_break
- Exclude snapshots with missing core physical attributes.
- Save excluded rows in data/processed/rejected_rows.csv for traceability.
- Enforce temporal leakage protections (no post-break / repair fields).
- Provide a strict time-based train/test split (no random splitting).

Horizon labels (1 / 2 / 5 years) are produced in labeling.generate_snapshot_dataset
from raw break events — never recomputed from snapshot rows.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from src.config import (
        HORIZON_YEARS,
        TEMPORAL_SPLIT_DATE,
        processed_snapshots_path,
        raw_breaks_path,
    )
    from src.labeling import generate_snapshot_dataset, load_break_events
    from src.schema import (
        FEATURE_COLUMNS,
        LEAKAGE_FORBIDDEN_COLUMNS,
        META_COLUMNS,
        TARGET_COLUMN,
    )
except ModuleNotFoundError:
    from config import (
        HORIZON_YEARS,
        TEMPORAL_SPLIT_DATE,
        processed_snapshots_path,
        raw_breaks_path,
    )
    from labeling import generate_snapshot_dataset, load_break_events
    from schema import (
        FEATURE_COLUMNS,
        LEAKAGE_FORBIDDEN_COLUMNS,
        META_COLUMNS,
        TARGET_COLUMN,
    )


SUPPORTED_HORIZON_YEARS = frozenset({1, 2, 5})
DEFAULT_TARGET_COLUMN = TARGET_COLUMN
DATA_COLUMNS = FEATURE_COLUMNS

PREDICTION_EXCLUDED_COLUMNS = list(LEAKAGE_FORBIDDEN_COLUMNS) + [
    "asset_id",
    "snapshot_date",
    "street",
    "snapshot_origin",
    "horizon_years",
    "length_m",
    "break_date",
    "incident_date",
]

# A snapshot cannot represent a usable pipe for the model if all of these
# physical attributes are missing. Do not include target or temporal features here.
CORE_PHYSICAL_COLUMNS = ["material", "diameter_mm", "install_year"]


def resolve_dataset_path(csv_path=None) -> Path:
    dataset_path = (
        Path(csv_path) if csv_path is not None else Path(processed_snapshots_path)
    )
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parent.parent / dataset_path
    return dataset_path


def _rejected_rows_path() -> Path:
    """Return the audit file path next to the processed snapshot dataset."""
    return resolve_dataset_path().parent / "rejected_rows.csv"


def remove_incomplete_snapshots(
    df: pd.DataFrame,
    required_columns: Optional[list[str]] = None,
    rejected_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Exclude rows where every core physical pipe attribute is missing.

    The excluded source rows are saved to a CSV with the missing columns and
    rejection metadata. This preserves traceability without adding extra folders.
    """
    required_columns = required_columns or CORE_PHYSICAL_COLUMNS
    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"Columns required for snapshot quality validation are missing: "
            f"{missing_columns}"
        )

    out = df.replace(r"^\s*$", pd.NA, regex=True).copy()
    invalid_mask = out[required_columns].isna().all(axis=1)

    rejected_rows = out.loc[invalid_mask].copy()
    clean_df = out.loc[~invalid_mask].copy()

    if not rejected_rows.empty:
        rejected_rows["missing_columns"] = rejected_rows[required_columns].apply(
            lambda row: ",".join(row.index[row.isna()].tolist()), axis=1
        )
        rejected_rows["rejection_reason"] = "all_core_physical_features_missing"
        rejected_rows["rejected_at_utc"] = datetime.now(timezone.utc).isoformat()

        output_path = rejected_path or _rejected_rows_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rejected_rows.to_csv(output_path, index=False)

    print(
        f"[Data quality] Input: {len(out)} | "
        f"Kept: {len(clean_df)} | Rejected: {len(rejected_rows)}"
    )
    return clean_df


def _normalize_material(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip().str.upper()
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
    leaked = [
        lower_map[c.lower()]
        for c in LEAKAGE_FORBIDDEN_COLUMNS
        if c.lower() in lower_map
    ]
    if leaked:
        raise ValueError(
            "Temporal leakage risk: forbidden post-break columns present in frame: "
            f"{leaked}"
        )

    allowed_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN] + META_COLUMNS)
    unexpected = [c for c in df.columns if c not in allowed_columns]
    if unexpected:
        raise ValueError(
            "Unexpected columns outside the Phase-2 data contract: "
            f"{unexpected}"
        )


def _years_since_last_break_for_row(
    snapshot_date: pd.Timestamp,
    prior_dates: list[pd.Timestamp],
) -> float:
    prior = [date for date in prior_dates if date < snapshot_date]
    if not prior:
        return np.nan
    return float((snapshot_date - max(prior)).days) / 365.25


def add_years_since_last_break(
    snapshots: pd.DataFrame,
    events: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Attach years_since_last_break using only breaks before snapshot_date."""
    if events is None:
        events = load_break_events(raw_breaks_path)

    events = events.copy()
    if "incident_date" not in events.columns:
        raise ValueError("load_break_events must return an 'incident_date' column.")
    if "asset_id" not in events.columns:
        raise ValueError("load_break_events must return an 'asset_id' column.")

    events["incident_date"] = pd.to_datetime(
        events["incident_date"], errors="coerce"
    ).dt.normalize()
    events["asset_id"] = events["asset_id"].astype(str)

    breaks_by_asset = {
        str(asset_id): sorted(group["incident_date"].dropna().tolist())
        for asset_id, group in events.groupby("asset_id", sort=False)
    }

    out = snapshots.copy()
    out["snapshot_date"] = pd.to_datetime(
        out["snapshot_date"], errors="coerce"
    ).dt.normalize()
    out["asset_id"] = out["asset_id"].astype(str)

    out["years_since_last_break"] = [
        _years_since_last_break_for_row(
            snapshot_date,
            breaks_by_asset.get(str(asset_id), []),
        )
        for asset_id, snapshot_date in zip(out["asset_id"], out["snapshot_date"])
    ]
    return out


def validate_time_aware_features(df: pd.DataFrame) -> None:
    """Sanity-check consistency of prior_break_count vs years_since_last_break."""
    if "prior_break_count" not in df.columns or "years_since_last_break" not in df.columns:
        raise ValueError("Missing time-aware features required for Phase 2.")

    zero_prior = df["prior_break_count"].fillna(0).astype(int) == 0
    if zero_prior.any() and df.loc[zero_prior, "years_since_last_break"].notna().any():
        raise ValueError(
            "Leakage/consistency error: years_since_last_break is set while "
            "prior_break_count == 0."
        )

    positive_prior = df["prior_break_count"].fillna(0).astype(int) > 0
    if positive_prior.any() and df.loc[
        positive_prior, "years_since_last_break"
    ].isna().any():
        raise ValueError(
            "Consistency error: prior_break_count > 0 but years_since_last_break "
            "is missing."
        )


def _validate_horizon_years(horizon_years: int) -> int:
    horizon = int(horizon_years)
    if horizon not in SUPPORTED_HORIZON_YEARS:
        raise ValueError(
            f"horizon_years must be one of {sorted(SUPPORTED_HORIZON_YEARS)}; "
            f"got {horizon_years!r}."
        )
    return horizon


def _snapshot_horizon_matches(df: pd.DataFrame, horizon_years: int) -> bool:
    if "horizon_years" not in df.columns or df.empty:
        return False
    values = pd.to_numeric(df["horizon_years"], errors="coerce")
    return not values.isna().any() and bool(
        (values.astype(int) == int(horizon_years)).all()
    )


def load_snapshot_data(
    csv_path=None,
    regenerate_if_missing: bool = True,
    horizon_years: int = HORIZON_YEARS,
) -> pd.DataFrame:
    """Load Phase-1 snapshots for the requested horizon."""
    horizon_years = _validate_horizon_years(horizon_years)
    dataset_path = resolve_dataset_path(csv_path)

    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        df["snapshot_date"] = pd.to_datetime(
            df["snapshot_date"], errors="coerce"
        ).dt.normalize()
        df["asset_id"] = df["asset_id"].astype(str)
        if _snapshot_horizon_matches(df, horizon_years):
            return df
        if not regenerate_if_missing:
            raise ValueError(
                f"Cached snapshots at {dataset_path} do not match "
                f"horizon_years={horizon_years}."
            )
    elif not regenerate_if_missing:
        raise FileNotFoundError(f"Snapshot dataset not found: {dataset_path}")

    df = generate_snapshot_dataset(horizon_years=horizon_years)
    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"], errors="coerce"
    ).dt.normalize()
    df["asset_id"] = df["asset_id"].astype(str)

    if not dataset_path.exists():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dataset_path, index=False)

    return df


def engineer_pipe_features(
    snapshots: Optional[pd.DataFrame] = None,
    horizon_years: int = HORIZON_YEARS,
) -> pd.DataFrame:
    """Build the cleaned modeling table with Phase-2 features only."""
    horizon_years = _validate_horizon_years(horizon_years)

    if snapshots is None:
        snapshots = load_snapshot_data(horizon_years=horizon_years)
    elif not _snapshot_horizon_matches(snapshots, horizon_years):
        raise ValueError(
            f"Provided snapshots do not match horizon_years={horizon_years}. "
            "Regenerate via labeling.generate_snapshot_dataset(horizon_years=...)."
        )

    if TARGET_COLUMN not in snapshots.columns:
        raise ValueError(
            f"Snapshots must already contain {TARGET_COLUMN!r} from labeling; "
            "horizon labels are not recomputed in preprocessing."
        )

    # Keep an audit trail before imputing material or filtering numerical values.
    df = remove_incomplete_snapshots(snapshots)
    df = add_years_since_last_break(df)
    df["horizon_years"] = int(horizon_years)

    df["material"] = _normalize_material(df["material"])
    df["diameter_mm"] = pd.to_numeric(df["diameter_mm"], errors="coerce")
    df["install_year"] = pd.to_numeric(df["install_year"], errors="coerce")
    df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
    df["prior_break_count"] = pd.to_numeric(
        df["prior_break_count"], errors="coerce"
    ).fillna(0).astype(int)
    df["years_since_last_break"] = pd.to_numeric(
        df["years_since_last_break"], errors="coerce"
    )
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="coerce"
    ).astype("Int64")

    # The target and time reference are indispensable. Diameter/install year can
    # still be missing individually only if the pipeline later adds imputation.
    required = [TARGET_COLUMN, "snapshot_date", "asset_id"]
    df = df.dropna(subset=required).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    df["age_years"] = df["snapshot_date"].dt.year - df["install_year"]

    keep_cols = [
        column
        for column in META_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
        if column in df.columns
    ]
    modeled = df[keep_cols].copy()

    assert_no_temporal_leakage(modeled)
    validate_time_aware_features(modeled)

    missing_features = [column for column in FEATURE_COLUMNS if column not in modeled.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {missing_features}")

    return modeled.sort_values(["snapshot_date", "asset_id"]).reset_index(drop=True)


def temporal_train_test_split(
    df: pd.DataFrame,
    cutoff: Optional[str] = None,
):
    """Strict time-based split on snapshot_date."""
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
    assert_no_temporal_leakage(pd.concat([X, y.rename(TARGET_COLUMN)], axis=1))
    return X, y


def prepare_pipe_break_data(
    csv_path=None,
    target_column: str = DEFAULT_TARGET_COLUMN,
    cutoff: Optional[str] = None,
    horizon_years: int = HORIZON_YEARS,
):
    """Prepare time-split train/test data for Phase-2 classification."""
    if target_column != TARGET_COLUMN:
        raise ValueError(
            f"Unsupported target_column={target_column!r}; "
            f"Phase 2 classification target is {TARGET_COLUMN!r}."
        )

    horizon_years = _validate_horizon_years(horizon_years)
    cleaned_df = engineer_pipe_features(
        load_snapshot_data(csv_path, horizon_years=horizon_years),
        horizon_years=horizon_years,
    )
    train_df, test_df = temporal_train_test_split(cleaned_df, cutoff=cutoff)

    X_train, y_train = _split_xy(train_df)
    X_test, y_test = _split_xy(test_df)
    return X_train, X_test, y_train, y_test, cleaned_df


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
