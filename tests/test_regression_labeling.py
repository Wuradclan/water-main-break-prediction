"""Tests for years_until_break regression labeling (non-censored vs censored)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labeling import add_years_until_next_break
from src.schema import REGRESSION_TARGET_COLUMN


def test_years_until_break_with_future_rupture():
    snapshots = pd.DataFrame(
        {
            "asset_id": ["A1", "A1"],
            "snapshot_date": [pd.Timestamp("2010-01-01"), pd.Timestamp("2012-01-01")],
        }
    )
    events = pd.DataFrame(
        {
            "asset_id": ["A1", "A1"],
            "incident_date": [pd.Timestamp("2011-01-01"), pd.Timestamp("2015-07-01")],
        }
    )

    out = add_years_until_next_break(snapshots, events)
    assert REGRESSION_TARGET_COLUMN in out.columns

    # First snapshot: next break is 2011-01-01 (~1 year)
    expected_0 = (pd.Timestamp("2011-01-01") - pd.Timestamp("2010-01-01")).days / 365.25
    assert out.loc[0, REGRESSION_TARGET_COLUMN] == pytest.approx(expected_0)

    # Second snapshot: next break is 2015-07-01
    expected_1 = (pd.Timestamp("2015-07-01") - pd.Timestamp("2012-01-01")).days / 365.25
    assert out.loc[1, REGRESSION_TARGET_COLUMN] == pytest.approx(expected_1)


def test_years_until_break_censored_is_nan():
    snapshots = pd.DataFrame(
        {
            "asset_id": ["B1"],
            "snapshot_date": [pd.Timestamp("2020-01-01")],
        }
    )
    events = pd.DataFrame(
        {
            "asset_id": ["B1"],
            # Only a past break — no future rupture after snapshot_date
            "incident_date": [pd.Timestamp("2015-01-01")],
        }
    )

    out = add_years_until_next_break(snapshots, events)
    assert pd.isna(out.loc[0, REGRESSION_TARGET_COLUMN])


def test_years_until_break_unknown_asset_is_nan():
    snapshots = pd.DataFrame(
        {
            "asset_id": ["UNKNOWN"],
            "snapshot_date": [pd.Timestamp("2010-01-01")],
        }
    )
    events = pd.DataFrame(
        {
            "asset_id": ["OTHER"],
            "incident_date": [pd.Timestamp("2012-01-01")],
        }
    )
    out = add_years_until_next_break(snapshots, events)
    assert np.isnan(out.loc[0, REGRESSION_TARGET_COLUMN])
