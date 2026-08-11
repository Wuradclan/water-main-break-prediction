"""Unit checks for regression training helpers (metrics + empty-target guard)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.schema import REGRESSION_TARGET_COLUMN
from src.train import calculate_regression_metrics, prepare_regression_data


def test_regression_metrics_rmse_equals_sqrt_mse():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 2.5, 2.5, 5.0])
    metrics = calculate_regression_metrics(y_true, y_pred)

    assert {"mse", "rmse", "mae", "r2"} <= set(metrics)
    assert metrics["rmse"] == pytest.approx(np.sqrt(metrics["mse"]))
    assert metrics["mse"] >= 0.0
    assert metrics["mae"] >= 0.0


def test_regression_metrics_with_prefix():
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.0, 1.0])
    metrics = calculate_regression_metrics(y_true, y_pred, prefix="test_")
    assert metrics["test_mse"] == pytest.approx(0.0)
    assert metrics["test_rmse"] == pytest.approx(0.0)
    assert metrics["test_r2"] == pytest.approx(1.0)


def test_prepare_regression_data_fails_when_all_censored(monkeypatch):
    """Training must fail cleanly if no row has a valid years_until_break."""

    def _fake_load_snapshot_data(**kwargs):
        return pd.DataFrame(
            {
                "asset_id": ["X1", "X2"],
                "snapshot_date": [pd.Timestamp("2010-01-01"), pd.Timestamp("2011-01-01")],
                "material": ["CI", "DI"],
                "diameter_mm": [150.0, 200.0],
                "install_year": [1950.0, 1960.0],
                "age_years": [60.0, 51.0],
                "prior_break_count": [0, 0],
                "break_within_horizon": [0, 1],
                "horizon_years": [5, 5],
                "street": ["A", "B"],
                "snapshot_origin": ["neg", "pos"],
            }
        )

    def _fake_engineer(snapshots, horizon_years=5):
        out = snapshots.copy()
        out["years_since_last_break"] = np.nan
        return out

    def _fake_events():
        return pd.DataFrame(
            {
                "asset_id": ["X1", "X2"],
                "incident_date": [pd.Timestamp("2000-01-01"), pd.Timestamp("2001-01-01")],
            }
        )

    def _fake_add_years(snapshots, events):
        out = snapshots.copy()
        out[REGRESSION_TARGET_COLUMN] = np.nan
        return out

    monkeypatch.setattr("src.train.load_snapshot_data", _fake_load_snapshot_data)
    monkeypatch.setattr("src.train.engineer_pipe_features", _fake_engineer)
    monkeypatch.setattr("src.train.load_break_events", _fake_events)
    monkeypatch.setattr("src.train.add_years_until_next_break", _fake_add_years)

    with pytest.raises(ValueError, match="No valid regression targets"):
        prepare_regression_data(horizon_years=5)
