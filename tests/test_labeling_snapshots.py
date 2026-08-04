"""
Phase 1 validation: historical snapshot / negative-class generation.

Includes the concrete worked example for ASSETID 33550 (Ross Ave, CI, 150 mm, 1959)
with horizon H = 5 years.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import HORIZON_YEARS, dataset_checksums_path, raw_breaks_path
from src.labeling import (
    build_snapshots_for_asset,
    compute_file_sha256,
    generate_snapshot_dataset,
    load_break_events,
    summarize_snapshots,
)
from src.schema import LEAKAGE_FORBIDDEN_COLUMNS, SNAPSHOT_FEATURE_COLUMNS, TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_raw_dataset_is_pinned_by_sha256():
    assert raw_breaks_path.exists(), f"Missing pinned dataset: {raw_breaks_path}"
    assert dataset_checksums_path.exists(), f"Missing checksums file: {dataset_checksums_path}"

    expected = None
    for line in dataset_checksums_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(maxsplit=1)
        if rel.endswith("Water_Main_Breaks.csv"):
            expected = digest
            break

    assert expected is not None, "SHA256SUMS does not list Water_Main_Breaks.csv"
    assert compute_file_sha256(raw_breaks_path) == expected


def test_load_break_events_schema_and_dedup():
    events = load_break_events()
    assert not events.empty
    assert {"asset_id", "incident_date", "material", "diameter_mm", "install_year"}.issubset(
        events.columns
    )
    # Same asset + same calendar day should appear at most once.
    dupes = events.duplicated(subset=["asset_id", "incident_date"]).sum()
    assert dupes == 0


def test_generate_snapshot_dataset_label_balance_and_features():
    snapshots = generate_snapshot_dataset()
    summary = summarize_snapshots(snapshots)

    assert summary["n_snapshots"] > 0
    assert summary["n_positive"] > 0
    assert summary["n_negative"] > 0
    assert summary["horizon_years"] == HORIZON_YEARS
    assert set(SNAPSHOT_FEATURE_COLUMNS).issubset(snapshots.columns)
    assert TARGET_COLUMN in snapshots.columns
    assert set(snapshots[TARGET_COLUMN].unique()).issubset({0, 1})

    # Leakage columns from the break/repair record must not appear as features.
    lower_cols = {c.lower() for c in snapshots.columns}
    leaked = [c for c in LEAKAGE_FORBIDDEN_COLUMNS if c.lower() in lower_cols]
    assert leaked == []


def test_worked_example_asset_33550_h5():
    """
    Worked example agreed in the migration plan:

    Pipe ASSETID 33550 — Ross Ave, CI, 150 mm, installed 1959
    First break T = 1999-01-19, H = 5 years

      t = 1994-01-19 (T - H)  -> age 35, prior 0, label 1
      t = 1989-01-19 (T - 2H) -> age 30, prior 0, label 0
    """
    events = load_break_events()
    asset_events = events[events["asset_id"] == "33550"].copy()
    assert not asset_events.empty, "ASSETID 33550 missing from pinned KW extract"

    # Physical attributes used in the plan.
    assert asset_events["material"].dropna().iloc[0] == "CI"
    assert float(asset_events["diameter_mm"].dropna().iloc[0]) == 150.0
    assert int(asset_events["install_year"].dropna().iloc[0]) == 1959

    first_break = asset_events["incident_date"].min()
    assert first_break == pd.Timestamp("1999-01-19")

    snapshots = build_snapshots_for_asset(asset_events, horizon_years=5)
    by_date = {pd.Timestamp(s["snapshot_date"]): s for s in snapshots}

    pos = by_date[pd.Timestamp("1994-01-19")]
    neg = by_date[pd.Timestamp("1989-01-19")]

    assert pos[TARGET_COLUMN] == 1
    assert pos["age_years"] == 35
    assert pos["prior_break_count"] == 0
    assert pos["material"] == "CI"
    assert pos["diameter_mm"] == 150.0

    assert neg[TARGET_COLUMN] == 0
    assert neg["age_years"] == 30
    assert neg["prior_break_count"] == 0
    assert neg["material"] == "CI"
    assert neg["diameter_mm"] == 150.0


def test_invalid_negative_window_is_rejected_for_later_break():
    """
    For break T = 2010-07-07 on ASSETID 33550, the naive negative
    t = 2000-07-07 must be discarded because 2004-10-15 falls in (t, t+5y].
    """
    events = load_break_events()
    asset_events = events[events["asset_id"] == "33550"].copy()
    snapshots = build_snapshots_for_asset(asset_events, horizon_years=5)
    snapshot_dates = {pd.Timestamp(s["snapshot_date"]).normalize() for s in snapshots}

    assert pd.Timestamp("2005-07-07") in snapshot_dates  # valid positive for T=2010-07-07
    assert pd.Timestamp("2000-07-07") not in snapshot_dates  # invalid negative
