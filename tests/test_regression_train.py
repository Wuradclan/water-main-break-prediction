"""Phase 6 validation: years_until_break regression training helpers.

Mirrors the style of tests/test_train_metrics.py, extended for the
regression task (--task regression) added to src/train.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.train as train_module
from src.train import (
    REGRESSION_MODEL_TYPES,
    calculate_regression_metrics,
    get_regression_models,
    prepare_regression_data,
)


def test_calculate_regression_metrics_matches_manual_formulas():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.5, 2.5, 2.5, 4.5, 4.0])

    metrics = calculate_regression_metrics(y_true, y_pred)

    assert {"mse", "rmse", "mae", "r2"} <= set(metrics)

    manual_mse = float(np.mean((y_true - y_pred) ** 2))
    manual_mae = float(np.mean(np.abs(y_true - y_pred)))
    assert metrics["mse"] == pytest.approx(manual_mse, rel=1e-9)
    assert metrics["mae"] == pytest.approx(manual_mae, rel=1e-9)
    assert metrics["rmse"] == pytest.approx(np.sqrt(manual_mse), rel=1e-9)


def test_rmse_equals_sqrt_mse_for_perfect_and_imperfect_fits():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])

    perfect = calculate_regression_metrics(y_true, y_true)
    assert perfect["mse"] == pytest.approx(0.0, abs=1e-9)
    assert perfect["rmse"] == pytest.approx(np.sqrt(perfect["mse"]), rel=1e-9)
    assert perfect["r2"] == pytest.approx(1.0)

    noisy_pred = y_true + np.array([1.0, -2.0, 3.0, -1.0])
    noisy = calculate_regression_metrics(y_true, noisy_pred)
    assert noisy["rmse"] == pytest.approx(np.sqrt(noisy["mse"]), rel=1e-9)


def test_get_regression_models_exposes_all_required_model_types():
    feature_columns = [
        "material",
        "diameter_mm",
        "install_year",
        "age_years",
        "prior_break_count",
        "years_since_last_break",
    ]
    models = get_regression_models(feature_columns)

    assert set(REGRESSION_MODEL_TYPES) <= set(models)
    for model_type in REGRESSION_MODEL_TYPES:
        pipeline = models[model_type]
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps


def test_regression_pipelines_fit_and_predict_on_synthetic_data():
    rng = np.random.default_rng(42)
    n = 40
    df = pd.DataFrame(
        {
            "material": rng.choice(["CI", "PVC", "DI"], size=n),
            "diameter_mm": rng.uniform(100, 300, size=n),
            "install_year": rng.integers(1950, 2000, size=n).astype(float),
            "age_years": rng.uniform(5, 60, size=n),
            "prior_break_count": rng.integers(0, 5, size=n).astype(float),
            "years_since_last_break": rng.uniform(0, 10, size=n),
        }
    )
    y = pd.Series(rng.uniform(1, 20, size=n))

    models = get_regression_models(list(df.columns))
    for model_type in REGRESSION_MODEL_TYPES:
        pipeline = models[model_type]
        pipeline.fit(df, y)
        preds = pipeline.predict(df)
        assert len(preds) == n
        assert np.all(np.isfinite(preds))


def test_prepare_regression_data_raises_when_all_rows_censored(monkeypatch):
    """Training must fail cleanly (ValueError) if no row has a valid target."""

    def _all_censored(snapshots: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        out = snapshots.copy()
        out["years_until_break"] = np.nan
        return out

    monkeypatch.setattr(train_module, "add_years_until_next_break", _all_censored)

    with pytest.raises(ValueError, match="censored"):
        train_module.prepare_regression_data()


def test_prepare_regression_data_only_keeps_uncensored_rows():
    X_train, X_test, y_train, y_test, with_target = prepare_regression_data()

    assert with_target["years_until_break"].notna().all()
    assert y_train.notna().all()
    assert y_test.notna().all()
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert not X_train.empty
    assert not X_test.empty
