"""
Tests for the confusion-matrix evaluation endpoints in api/main.py:

    GET /models/{run_id}/evaluation
    GET /models/{run_id}/confusion-matrix

Both endpoints must read exclusively from the requested MLflow run_id (never
from a shared reports/ file or the in-memory champion), be a no-op for
regression runs, and gracefully report a missing artifact instead of
raising a 500.
"""

from __future__ import annotations

import numpy as np
import mlflow
import pytest
from fastapi import HTTPException

import api.main as api_main
from api.main import get_confusion_matrix_image, get_model_evaluation
from src.train import log_confusion_matrix_artifacts


def _log_classification_run(reports_dir, horizon_years: int = 5) -> str:
    """Start+log a classification run with real confusion-matrix artifacts."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])  # tn=2, fp=1, fn=1, tp=2

    with mlflow.start_run(run_name="run_classif_eval") as run:
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("horizon_years", horizon_years)
        mlflow.log_metric("pr_auc_test", 0.82)
        mlflow.log_metric("f1_test", 0.65)
        mlflow.log_metric("roc_auc_test", 0.88)
        mlflow.log_metric("recall_at_k_test", 0.6)
        mlflow.log_metric("brier_test", 0.12)
        confusion_metrics = log_confusion_matrix_artifacts(y_true, y_pred, reports_dir=reports_dir)
        mlflow.log_metrics(confusion_metrics)
        run_id = run.info.run_id
    return run_id


def _log_classification_run_without_artifact() -> str:
    """A classification run predating this feature: metrics but no PNG/CSV."""
    with mlflow.start_run(run_name="run_classif_no_artifact") as run:
        mlflow.log_param("model_type", "logistic")
        mlflow.log_param("horizon_years", 5)
        mlflow.log_metric("pr_auc_test", 0.70)
        mlflow.log_metric("f1_test", 0.55)
        run_id = run.info.run_id
    return run_id


def _log_regression_run() -> str:
    with mlflow.start_run(run_name="run_reg_eval") as run:
        mlflow.log_param("model_type", "xgb_reg")
        mlflow.log_param("task", "regression")
        mlflow.log_metric("rmse_test", 1.1)
        mlflow.log_metric("mae_test", 0.9)
        mlflow.log_metric("r2_test", 0.4)
        run_id = run.info.run_id
    return run_id


@pytest.fixture()
def mlflow_store(tmp_path, monkeypatch):
    """Throwaway local MLflow file store, mirroring test_api_training.py."""
    tracking_uri = f"file://{tmp_path}/mlruns_eval"
    monkeypatch.setattr(api_main, "resolve_tracking_uri", lambda *a, **kw: tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(api_main.MLFLOW_EXPERIMENT_NAME)
    return tmp_path


# ---------------------------------------------------------------------------
# GET /models/{run_id}/evaluation
# ---------------------------------------------------------------------------


def test_evaluation_returns_confusion_counts_for_classification_run(mlflow_store, tmp_path):
    run_id = _log_classification_run(reports_dir=tmp_path / "reports")

    body = get_model_evaluation(run_id)

    assert body["run_id"] == run_id
    assert body["task"] == "classification"
    assert body["threshold"] == pytest.approx(0.50)
    assert body["confusion_matrix"] == {
        "true_negatives": 2,
        "false_positives": 1,
        "false_negatives": 1,
        "true_positives": 2,
    }
    assert body["artifact_available"] is True
    assert body["metrics"]["pr_auc_test"] == pytest.approx(0.82)
    assert body["metrics"]["f1_test"] == pytest.approx(0.65)
    assert body["metrics"]["roc_auc_test"] == pytest.approx(0.88)
    assert body["metrics"]["recall_at_k"] == pytest.approx(0.6)
    assert body["metrics"]["brier_test"] == pytest.approx(0.12)


def test_evaluation_horizon_is_always_five_for_classification(mlflow_store, tmp_path):
    """Spec constraint: this feature is fixed to the 5-year horizon, so the
    reported horizon must be 5 regardless of what an (unsupported/legacy)
    run's own horizon_years param says."""
    run_id = _log_classification_run(reports_dir=tmp_path / "reports", horizon_years=1)

    body = get_model_evaluation(run_id)

    assert body["horizon_years"] == 5


def test_evaluation_reports_artifact_unavailable_for_older_run(mlflow_store):
    run_id = _log_classification_run_without_artifact()

    body = get_model_evaluation(run_id)

    assert body["task"] == "classification"
    assert body["artifact_available"] is False
    # Counts weren't logged either: every entry falls back to None -> "—" in the UI.
    assert body["confusion_matrix"] == {
        "true_negatives": None,
        "false_positives": None,
        "false_negatives": None,
        "true_positives": None,
    }


def test_evaluation_hides_confusion_matrix_for_regression_run(mlflow_store):
    run_id = _log_regression_run()

    body = get_model_evaluation(run_id)

    assert body["task"] == "regression"
    assert body["confusion_matrix"] is None
    assert body["artifact_available"] is False
    assert body["horizon_years"] is None
    assert body["threshold"] is None


def test_evaluation_unknown_run_raises_404(mlflow_store):
    with pytest.raises(HTTPException) as exc_info:
        get_model_evaluation("does-not-exist")
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# GET /models/{run_id}/confusion-matrix
# ---------------------------------------------------------------------------


def test_confusion_matrix_image_returns_png_for_classification_run(mlflow_store, tmp_path):
    run_id = _log_classification_run(reports_dir=tmp_path / "reports")

    response = get_confusion_matrix_image(run_id)

    assert response.media_type == "image/png"
    assert response.body[:8] == b"\x89PNG\r\n\x1a\n"  # PNG file signature
    assert len(response.body) > 0


def test_confusion_matrix_image_404_when_artifact_missing(mlflow_store):
    run_id = _log_classification_run_without_artifact()

    with pytest.raises(HTTPException) as exc_info:
        get_confusion_matrix_image(run_id)
    assert exc_info.value.status_code == 404


def test_confusion_matrix_image_404_for_regression_run(mlflow_store):
    run_id = _log_regression_run()

    with pytest.raises(HTTPException) as exc_info:
        get_confusion_matrix_image(run_id)
    assert exc_info.value.status_code == 404


def test_confusion_matrix_image_404_for_unknown_run(mlflow_store):
    with pytest.raises(HTTPException) as exc_info:
        get_confusion_matrix_image("does-not-exist")
    assert exc_info.value.status_code == 404
