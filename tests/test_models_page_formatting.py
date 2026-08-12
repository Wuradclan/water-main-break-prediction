"""
Tests for app/formatting.py — the pandas-NaN-safe display helpers used by
app/pages/2_Modeles.py to render classification and regression runs side by
side.

GET /models returns `null` for metrics that don't apply to a given run's
task (e.g. rmse_test on a classification run). Once the API payload goes
through pandas.DataFrame(models), those `null` values become NaN, and the
page must not let a naive `f"{value:.3f}"` turn that into a literal "nan" /
"nan / nan" in the UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.formatting import format_f1_pair, format_metric


def test_format_metric_handles_none_and_nan():
    assert format_metric(None) == "—"
    assert format_metric(np.nan) == "—"
    assert format_metric(pd.NA) == "—"


def test_format_metric_formats_finite_numbers():
    assert format_metric(0.82412) == "0.824"
    assert format_metric(0) == "0.000"
    assert format_metric(1.5, decimals=1) == "1.5"


def test_format_metric_rejects_non_numeric_strings():
    assert format_metric("n/a") == "—"


def test_format_f1_pair_handles_none_and_nan_on_either_side():
    assert format_f1_pair(None, 0.5) == "—"
    assert format_f1_pair(0.5, None) == "—"
    assert format_f1_pair(np.nan, 0.5) == "—"
    assert format_f1_pair(0.5, np.nan) == "—"
    assert format_f1_pair(np.nan, np.nan) == "—"
    assert format_f1_pair(pd.NA, pd.NA) == "—"


def test_format_f1_pair_formats_both_finite_values():
    assert format_f1_pair(0.7, 0.65) == "0.700 / 0.650"


def test_mixed_classification_and_regression_dataframe_never_shows_nan():
    """
    Reproduces the exact scenario from the bug report: a models list mixing
    classification runs (PR-AUC/F1/ROC-AUC, no RMSE/MAE/R2) and regression
    runs (RMSE/MAE/R2, no PR-AUC/F1/ROC-AUC) — as returned by GET /models —
    turned into a DataFrame, where the API's `None` becomes pandas NaN.
    """
    models = [
        {
            "run_id": "run_a",
            "task": "classification",
            "model_type": "xgboost",
            "pr_auc_test": 0.82,
            "f1_train": 0.70,
            "f1_test": 0.65,
            "roc_auc_test": 0.88,
            "rmse_test": None,
            "mae_test": None,
            "r2_test": None,
        },
        {
            "run_id": "run_b",
            "task": "regression",
            "model_type": "xgb_reg",
            "pr_auc_test": None,
            "f1_train": None,
            "f1_test": None,
            "roc_auc_test": None,
            "rmse_test": 1.1,
            "mae_test": 0.9,
            "r2_test": 0.4,
        },
    ]

    df = pd.DataFrame(models)

    # Sanity check that pandas really did turn the API's None into NaN —
    # this is the precondition that caused the original "nan" rendering bug.
    assert df["rmse_test"].isna().iloc[0]
    assert df["pr_auc_test"].isna().iloc[1]

    df["PR-AUC test"] = df["pr_auc_test"].apply(format_metric)
    df["F1 train/test"] = df.apply(
        lambda r: format_f1_pair(r.get("f1_train"), r.get("f1_test")), axis=1
    )
    df["ROC-AUC test"] = df["roc_auc_test"].apply(format_metric)
    df["RMSE test"] = df["rmse_test"].apply(format_metric)
    df["MAE test"] = df["mae_test"].apply(format_metric)
    df["R² test"] = df["r2_test"].apply(format_metric)

    by_run = df.set_index("run_id")

    # Classification run: real numbers for its own metrics, "—" for regression ones.
    assert by_run.loc["run_a", "PR-AUC test"] == "0.820"
    assert by_run.loc["run_a", "F1 train/test"] == "0.700 / 0.650"
    assert by_run.loc["run_a", "ROC-AUC test"] == "0.880"
    assert by_run.loc["run_a", "RMSE test"] == "—"
    assert by_run.loc["run_a", "MAE test"] == "—"
    assert by_run.loc["run_a", "R² test"] == "—"

    # Regression run: real numbers for its own metrics, "—" for classification ones.
    assert by_run.loc["run_b", "PR-AUC test"] == "—"
    assert by_run.loc["run_b", "F1 train/test"] == "—"
    assert by_run.loc["run_b", "ROC-AUC test"] == "—"
    assert by_run.loc["run_b", "RMSE test"] == "1.100"
    assert by_run.loc["run_b", "MAE test"] == "0.900"
    assert by_run.loc["run_b", "R² test"] == "0.400"

    # No column, for any run, should ever contain the literal string "nan".
    formatted_columns = [
        "PR-AUC test", "F1 train/test", "ROC-AUC test", "RMSE test", "MAE test", "R² test",
    ]
    for column in formatted_columns:
        for value in df[column]:
            assert "nan" not in value.lower()
