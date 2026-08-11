"""Phase 6 validation: years_until_break regression labeling.

Mirrors the style of tests/test_labeling_snapshots.py and
tests/test_preprocessing.py (add_years_since_last_break tests), but for the
forward-looking years_until_break regression target.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.labeling import (
    add_years_until_next_break,
    build_snapshots_for_asset,
    load_break_events,
)


def _events(rows: list[tuple[str, pd.Timestamp]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["asset_id", "incident_date"])


def test_years_until_break_is_computed_from_next_future_break():
    """A snapshot with a recorded future break gets a finite years_until_break."""
    snapshots = pd.DataFrame(
        {
            "asset_id": ["A1"],
            "snapshot_date": [pd.Timestamp("2000-01-01")],
        }
    )
    events = _events(
        [
            ("A1", pd.Timestamp("1995-01-01")),  # past break, irrelevant
            ("A1", pd.Timestamp("2005-01-01")),  # nearest future break
            ("A1", pd.Timestamp("2010-01-01")),  # later future break, ignored
        ]
    )

    out = add_years_until_next_break(snapshots, events)

    assert "years_until_break" in out.columns
    expected = (pd.Timestamp("2005-01-01") - pd.Timestamp("2000-01-01")).days / 365.25
    assert out.loc[0, "years_until_break"] == pytest.approx(expected, rel=1e-6)


def test_years_until_break_is_nan_when_censored():
    """No recorded break after snapshot_date => censored target => NaN."""
    snapshots = pd.DataFrame(
        {
            "asset_id": ["A2"],
            "snapshot_date": [pd.Timestamp("2020-01-01")],
        }
    )
    events = _events(
        [
            ("A2", pd.Timestamp("2010-01-01")),  # only a past break
        ]
    )

    out = add_years_until_next_break(snapshots, events)
    assert pd.isna(out.loc[0, "years_until_break"])


def test_years_until_break_is_nan_when_asset_has_no_events_at_all():
    snapshots = pd.DataFrame(
        {
            "asset_id": ["UNKNOWN_ASSET"],
            "snapshot_date": [pd.Timestamp("2000-01-01")],
        }
    )
    events = _events([("OTHER_ASSET", pd.Timestamp("2005-01-01"))])

    out = add_years_until_next_break(snapshots, events)
    assert pd.isna(out.loc[0, "years_until_break"])


def test_years_until_break_mixed_censoring_across_rows():
    snapshots = pd.DataFrame(
        {
            "asset_id": ["A1", "A1", "A2"],
            "snapshot_date": [
                pd.Timestamp("1994-01-19"),
                pd.Timestamp("2005-07-07"),
                pd.Timestamp("2020-01-01"),
            ],
        }
    )
    events = _events(
        [
            ("A1", pd.Timestamp("1999-01-19")),
            ("A1", pd.Timestamp("2004-10-15")),
            ("A1", pd.Timestamp("2010-07-07")),
            ("A2", pd.Timestamp("2010-01-01")),  # only in the past for the 2020 snapshot
        ]
    )

    out = add_years_until_next_break(snapshots, events).reset_index(drop=True)

    expected_row0 = (pd.Timestamp("1999-01-19") - pd.Timestamp("1994-01-19")).days / 365.25
    expected_row1 = (pd.Timestamp("2010-07-07") - pd.Timestamp("2005-07-07")).days / 365.25

    assert out.loc[0, "years_until_break"] == pytest.approx(expected_row0, rel=1e-6)
    assert out.loc[1, "years_until_break"] == pytest.approx(expected_row1, rel=1e-6)
    assert pd.isna(out.loc[2, "years_until_break"])


def test_years_until_break_worked_example_asset_33550():
    """
    Reuses the pinned worked example (ASSETID 33550, first break 1999-01-19).

    The classification-positive snapshot at t = 1994-01-19 (T - 5y) must have
    years_until_break equal to the gap to that same first break, since it is
    the next break strictly after t.
    """
    events = load_break_events()
    asset_events = events[events["asset_id"] == "33550"].copy()
    assert not asset_events.empty, "ASSETID 33550 missing from pinned KW extract"

    first_break = asset_events["incident_date"].min()
    assert first_break == pd.Timestamp("1999-01-19")

    snapshots = pd.DataFrame(build_snapshots_for_asset(asset_events, horizon_years=5))
    out = add_years_until_next_break(snapshots, events)

    row = out[out["snapshot_date"] == pd.Timestamp("1994-01-19")].iloc[0]
    expected = (first_break - pd.Timestamp("1994-01-19")).days / 365.25
    assert row["years_until_break"] == pytest.approx(expected, rel=1e-6)


def test_years_until_break_does_not_alter_classification_columns():
    """The regression target is additive: it must not touch break_within_horizon."""
    events = load_break_events()
    asset_events = events[events["asset_id"] == "33550"].copy()
    snapshots = pd.DataFrame(build_snapshots_for_asset(asset_events, horizon_years=5))

    out = add_years_until_next_break(snapshots, events)

    assert "break_within_horizon" in out.columns
    pd.testing.assert_series_equal(
        out["break_within_horizon"], snapshots["break_within_horizon"], check_names=False
    )
