"""
Historical Snapshot / Windowing label generation for KW water-main breaks.

Problem framing
---------------
Binary classification:
    break_within_horizon = 1  iff the pipe breaks in (t, t + H years]
    break_within_horizon = 0  otherwise

Important limitation (Positive-Unlabeled / selection bias)
---------------------------------------------------------
The Kitchener open dataset only contains water mains that have broken at least
once. Label 0 therefore means "no recorded break inside the forward window for
a pipe that eventually appears in the break history", NOT "confirmed permanently
healthy / never-break risk is zero".

Negative-class strategy (chosen)
--------------------------------
For each break at time T on a pipe:
  * Positive snapshot at t = T - H (features as-of t; label 1)
  * Negative snapshots at t = T - H - k for k in
    NEGATIVE_OFFSETS_BEFORE_POSITIVE_YEARS (default 1..H),
    kept only if (t, t+H] contains no break.
    k = H recovers the original T - 2H negative from Phase 1.
    k in {1..H-1} densifies healthy windows so later temporal holdouts
    still contain negatives (fixes one-class test sets).

Features use only information available strictly before/at snapshot date t:
  material, diameter_mm, install_year, age_years, prior_break_count.
Post-break fields (nature, cause, repair flags, ...) are excluded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from src.config import (
        HORIZON_YEARS,
        NEGATIVE_OFFSETS_BEFORE_POSITIVE_YEARS,
        OBSERVATION_END,
        OBSERVATION_START,
        processed_snapshots_path,
        raw_breaks_path,
    )
    from src.schema import (
        LEAKAGE_FORBIDDEN_COLUMNS,
        RAW_REQUIRED_COLUMNS,
        SNAPSHOT_FEATURE_COLUMNS,
        SNAPSHOT_OUTPUT_COLUMNS,
        TARGET_COLUMN,
    )
except ModuleNotFoundError:
    from config import (
        HORIZON_YEARS,
        NEGATIVE_OFFSETS_BEFORE_POSITIVE_YEARS,
        OBSERVATION_END,
        OBSERVATION_START,
        processed_snapshots_path,
        raw_breaks_path,
    )
    from schema import (
        LEAKAGE_FORBIDDEN_COLUMNS,
        RAW_REQUIRED_COLUMNS,
        SNAPSHOT_FEATURE_COLUMNS,
        SNAPSHOT_OUTPUT_COLUMNS,
        TARGET_COLUMN,
    )


def compute_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_incident_dates(series: pd.Series) -> pd.Series:
    """Parse mixed KW timestamp formats into timezone-naive timestamps."""
    cleaned = series.astype(str).str.split("+").str[0].str.strip()
    parsed = pd.to_datetime(cleaned, errors="coerce", utc=False)
    # Normalize to midnight for date-level window arithmetic.
    return parsed.dt.normalize()


def load_break_events(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load and lightly normalize the raw KW break-incident extract."""
    path = Path(csv_path) if csv_path is not None else Path(raw_breaks_path)
    df = pd.read_csv(path)

    # ArcGIS CSV may include a UTF-8 BOM on the first column name.
    df.columns = [c.encode("utf-8").decode("utf-8-sig") if isinstance(c, str) else c for c in df.columns]
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Raw breaks file missing required columns: {missing}")

    events = pd.DataFrame(
        {
            "asset_id": df["ASSETID"].astype(str).str.strip(),
            "incident_date": _parse_incident_dates(df["INCIDENT_DATE"]),
            "material": df["ASSET_MATERIAL"].astype(str).str.strip().replace({"": np.nan, "nan": np.nan}),
            "diameter_mm": pd.to_numeric(df["ASSET_SIZE"], errors="coerce"),
            "install_year": pd.to_numeric(df["ASSET_YEAR_INSTALLED"], errors="coerce"),
            "street": df["STREET"].astype(str).str.strip() if "STREET" in df.columns else "",
        }
    )

    events = events.dropna(subset=["asset_id", "incident_date"]).copy()
    events = events[events["asset_id"].ne("") & events["asset_id"].ne("nan")]

    # Collapse duplicate same-day incidents per asset (same physical break day).
    events = (
        events.sort_values(["asset_id", "incident_date"])
        .drop_duplicates(subset=["asset_id", "incident_date"], keep="first")
        .reset_index(drop=True)
    )
    return events


def _years_delta(start: pd.Timestamp, years: float) -> pd.Timestamp:
    """Approximate calendar shift used for the H-year horizon windows."""
    return start + pd.DateOffset(years=years)


def _age_years(snapshot_date: pd.Timestamp, install_year: float) -> float:
    if pd.isna(install_year):
        return np.nan
    return float(snapshot_date.year) - float(install_year)


def _prior_break_count(break_dates: Iterable[pd.Timestamp], snapshot_date: pd.Timestamp) -> int:
    return int(sum(1 for d in break_dates if d < snapshot_date))


def _has_break_in_window(
    break_dates: Iterable[pd.Timestamp],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> bool:
    """True if any break occurs in (window_start, window_end]."""
    for d in break_dates:
        if window_start < d <= window_end:
            return True
    return False


def _is_label_window_observable(
    snapshot_date: pd.Timestamp,
    horizon_years: int,
    observation_start: pd.Timestamp,
    observation_end: pd.Timestamp,
) -> bool:
    """Require the full forward horizon to lie inside the recording window."""
    window_end = _years_delta(snapshot_date, horizon_years)
    return snapshot_date >= observation_start and window_end <= observation_end


def build_snapshots_for_asset(
    asset_events: pd.DataFrame,
    horizon_years: int = HORIZON_YEARS,
    observation_start: Optional[pd.Timestamp] = None,
    observation_end: Optional[pd.Timestamp] = None,
    negative_offsets_before_positive: Iterable[int] = NEGATIVE_OFFSETS_BEFORE_POSITIVE_YEARS,
) -> list[dict]:
    """
    Build positive/negative historical snapshots for one asset.

    For each unique break date T:
      - positive at t = T - H, label 1 (by construction if T is a break)
      - negatives at t = T - H - k for each k in negative_offsets_before_positive,
        kept only if (t, t+H] has no break (k=H == classic T-2H)
    """
    if asset_events.empty:
        return []

    observation_start = pd.Timestamp(observation_start or OBSERVATION_START)
    observation_end = pd.Timestamp(observation_end or OBSERVATION_END)

    break_dates = sorted(asset_events["incident_date"].tolist())
    # Physical attributes are treated as time-invariant on the breaks extract.
    # (Length join / richer inventory attributes are deferred.)
    material = asset_events["material"].dropna().iloc[0] if asset_events["material"].notna().any() else np.nan
    diameter = asset_events["diameter_mm"].dropna().iloc[0] if asset_events["diameter_mm"].notna().any() else np.nan
    install_year = asset_events["install_year"].dropna().iloc[0] if asset_events["install_year"].notna().any() else np.nan
    street = asset_events["street"].dropna().iloc[0] if "street" in asset_events and asset_events["street"].notna().any() else ""
    asset_id = str(asset_events["asset_id"].iloc[0])

    snapshots: list[dict] = []
    seen_keys: set[tuple[str, pd.Timestamp, int]] = set()

    for break_date in break_dates:
        candidates = [
            # Positive: prediction date H years before the break.
            (_years_delta(break_date, -horizon_years), 1, "positive_T_minus_H"),
        ]
        for offset_years in negative_offsets_before_positive:
            # t = T - H - k; when k=H this is the original T-2H negative.
            snapshot_date = _years_delta(break_date, -(horizon_years + int(offset_years)))
            candidates.append(
                (
                    snapshot_date,
                    0,
                    f"negative_T_minus_H_minus_{int(offset_years)}y",
                )
            )

        for snapshot_date, desired_label, origin in candidates:
            if not _is_label_window_observable(
                snapshot_date, horizon_years, observation_start, observation_end
            ):
                continue

            window_end = _years_delta(snapshot_date, horizon_years)
            broke = _has_break_in_window(break_dates, snapshot_date, window_end)
            label = 1 if broke else 0

            # Enforce strategy: keep positives for T-H; keep negatives only when healthy.
            if desired_label == 1 and label != 1:
                continue
            if desired_label == 0 and label != 0:
                continue

            key = (asset_id, snapshot_date, label)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            snapshots.append(
                {
                    "asset_id": asset_id,
                    "snapshot_date": snapshot_date,
                    "material": material,
                    "diameter_mm": diameter,
                    "install_year": install_year,
                    "age_years": _age_years(snapshot_date, install_year),
                    "prior_break_count": _prior_break_count(break_dates, snapshot_date),
                    TARGET_COLUMN: label,
                    "horizon_years": horizon_years,
                    "street": street,
                    "snapshot_origin": origin,
                }
            )

    return snapshots


def generate_snapshot_dataset(
    events: Optional[pd.DataFrame] = None,
    csv_path: Optional[Path] = None,
    horizon_years: int = HORIZON_YEARS,
    observation_start: Optional[str] = None,
    observation_end: Optional[str] = None,
) -> pd.DataFrame:
    """Generate the full snapshot table used for later training phases."""
    if events is None:
        events = load_break_events(csv_path)

    obs_start = pd.Timestamp(observation_start or OBSERVATION_START)
    obs_end = pd.Timestamp(observation_end or OBSERVATION_END)

    rows: list[dict] = []
    for asset_id, asset_events in events.groupby("asset_id", sort=False):
        rows.extend(
            build_snapshots_for_asset(
                asset_events,
                horizon_years=horizon_years,
                observation_start=obs_start,
                observation_end=obs_end,
            )
        )

    if not rows:
        return pd.DataFrame(columns=SNAPSHOT_OUTPUT_COLUMNS + ["snapshot_origin"])

    snapshots = pd.DataFrame(rows)
    snapshots = snapshots.sort_values(["snapshot_date", "asset_id"]).reset_index(drop=True)

    # Soft contract check: modeling features must not include leakage columns.
    forbidden_present = [c for c in LEAKAGE_FORBIDDEN_COLUMNS if c.lower() in {x.lower() for x in snapshots.columns}]
    if forbidden_present:
        raise RuntimeError(f"Leakage columns present in snapshot table: {forbidden_present}")

    missing_features = [c for c in SNAPSHOT_FEATURE_COLUMNS if c not in snapshots.columns]
    if missing_features:
        raise RuntimeError(f"Snapshot table missing feature columns: {missing_features}")

    return snapshots


def save_snapshot_dataset(
    snapshots: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    path = Path(output_path) if output_path is not None else Path(processed_snapshots_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshots.to_csv(path, index=False)
    return path


def _years_until_next_break_for_row(
    snapshot_date: pd.Timestamp,
    break_dates: list[pd.Timestamp],
) -> float:
    future = [d for d in break_dates if d > snapshot_date]
    if not future:
        return np.nan
    next_break = min(future)
    return float((next_break - snapshot_date).days) / 365.25


def add_years_until_next_break(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach years_until_break using only breaks with incident_date > snapshot_date.

    For each row, finds the next recorded break after snapshot_date for the same
    asset_id and computes years_until_break = (next_break - snapshot_date).days / 365.25.
    Rows for assets with no future recorded break are right-censored and get NaN
    (this is a regression target, not a classification label: censored rows must
    be excluded from regression training, see REGRESSION_TARGET_COLUMN).
    """
    events = events.copy()
    if "incident_date" not in events.columns:
        raise ValueError("events must contain an 'incident_date' column.")
    if "asset_id" not in events.columns:
        raise ValueError("events must contain an 'asset_id' column.")

    events["incident_date"] = pd.to_datetime(events["incident_date"], errors="coerce").dt.normalize()
    events["asset_id"] = events["asset_id"].astype(str)

    breaks_by_asset = {
        str(asset_id): sorted(group["incident_date"].dropna().tolist())
        for asset_id, group in events.groupby("asset_id", sort=False)
    }

    out = snapshots.copy()
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce").dt.normalize()
    out["asset_id"] = out["asset_id"].astype(str)

    values = []
    for asset_id, snapshot_date in zip(out["asset_id"], out["snapshot_date"]):
        values.append(
            _years_until_next_break_for_row(
                snapshot_date,
                breaks_by_asset.get(str(asset_id), []),
            )
        )
    out["years_until_break"] = values
    return out


def summarize_snapshots(snapshots: pd.DataFrame) -> dict:
    n = len(snapshots)
    n_pos = int((snapshots[TARGET_COLUMN] == 1).sum()) if n else 0
    n_neg = int((snapshots[TARGET_COLUMN] == 0).sum()) if n else 0
    return {
        "n_snapshots": n,
        "n_assets": int(snapshots["asset_id"].nunique()) if n else 0,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_rate": (n_pos / n) if n else 0.0,
        "horizon_years": int(snapshots["horizon_years"].iloc[0]) if n else HORIZON_YEARS,
        "snapshot_date_min": str(snapshots["snapshot_date"].min().date()) if n else None,
        "snapshot_date_max": str(snapshots["snapshot_date"].max().date()) if n else None,
    }


if __name__ == "__main__":
    raw_path = Path(raw_breaks_path)
    print(f"Loading breaks from: {raw_path}")
    print(f"SHA256: {compute_file_sha256(raw_path)}")

    events = load_break_events(raw_path)
    print(f"Break events (deduped by asset/day): {len(events)}")
    print(f"Distinct assets: {events['asset_id'].nunique()}")

    snapshots = generate_snapshot_dataset(events)
    out = save_snapshot_dataset(snapshots)
    summary = summarize_snapshots(snapshots)

    print(f"Wrote snapshots -> {out}")
    for key, value in summary.items():
        print(f"  {key}: {value}")
