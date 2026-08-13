"""Unit tests for the fixed 5-year-horizon confusion matrix in src/train.py.

Covers:
- compute_confusion_counts on a small, known sequence (one sample per cell).
- CSV/PNG artifact creation with the right columns/labels.
- End-to-end wiring through train_evaluate_and_log: the confusion matrix is
  computed exclusively on the test set, using the calibrated model's
  probabilities thresholded at CLASSIFICATION_THRESHOLD, and both the four
  counts (as MLflow metrics) and the CSV/PNG artifacts are logged.
"""

from __future__ import annotations

import mlflow
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.train import (
    CONFUSION_MATRIX_CLASS_LABELS,
    compute_confusion_counts,
    log_confusion_matrix_artifacts,
    save_confusion_matrix_csv,
    save_confusion_matrix_png,
    train_evaluate_and_log,
)


def test_compute_confusion_counts_on_known_sequence():
    # One sample landing in each of the four cells: (true=0,pred=0),
    # (true=0,pred=1), (true=1,pred=0), (true=1,pred=1).
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]

    counts = compute_confusion_counts(y_true, y_pred)

    assert counts["true_negatives"] == 1
    assert counts["false_positives"] == 1
    assert counts["false_negatives"] == 1
    assert counts["true_positives"] == 1


def test_compute_confusion_counts_all_correct_has_no_off_diagonal():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1]

    counts = compute_confusion_counts(y_true, y_pred)

    assert counts["true_negatives"] == 2
    assert counts["true_positives"] == 3
    assert counts["false_positives"] == 0
    assert counts["false_negatives"] == 0


def test_compute_confusion_counts_is_stable_when_a_class_is_absent():
    # labels=[0, 1] must keep the matrix 2x2 even if the slice has no
    # positives at all (e.g. an all-negative batch).
    y_true = [0, 0, 0]
    y_pred = [0, 1, 0]

    counts = compute_confusion_counts(y_true, y_pred)

    assert counts["true_negatives"] == 2
    assert counts["false_positives"] == 1
    assert counts["false_negatives"] == 0
    assert counts["true_positives"] == 0


def test_save_confusion_matrix_csv_has_required_columns(tmp_path):
    counts = {
        "true_negatives": 10,
        "false_positives": 2,
        "false_negatives": 3,
        "true_positives": 5,
    }
    csv_path = save_confusion_matrix_csv(counts, threshold=0.50, reports_dir=tmp_path)

    assert csv_path.name == "confusion_matrix_test.csv"
    df = pd.read_csv(csv_path)
    expected_columns = {
        "horizon_years",
        "threshold",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
    }
    assert expected_columns <= set(df.columns)
    assert df.loc[0, "horizon_years"] == 5
    assert df.loc[0, "threshold"] == pytest.approx(0.50)
    assert df.loc[0, "true_negatives"] == 10
    assert df.loc[0, "false_positives"] == 2
    assert df.loc[0, "false_negatives"] == 3
    assert df.loc[0, "true_positives"] == 5


def test_save_confusion_matrix_png_is_created(tmp_path):
    counts = {"true_negatives": 4, "false_positives": 1, "false_negatives": 1, "true_positives": 4}
    png_path = save_confusion_matrix_png(counts, threshold=0.50, reports_dir=tmp_path)

    assert png_path.name == "confusion_matrix_test.png"
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_confusion_matrix_business_labels_are_french_and_horizon_specific():
    assert CONFUSION_MATRIX_CLASS_LABELS == ("Pas de bris ≤ 5 ans", "Bris ≤ 5 ans")


def test_log_confusion_matrix_artifacts_logs_csv_and_png(tmp_path, monkeypatch):
    tracking_uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("confusion_matrix_artifact_test")

    reports_dir = tmp_path / "reports"
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]

    with mlflow.start_run(run_name="test_confusion_matrix_artifacts") as run:
        metrics = log_confusion_matrix_artifacts(y_true, y_pred, threshold=0.50, reports_dir=reports_dir)
        run_id = run.info.run_id

    assert metrics == {
        "test_true_negatives": 1.0,
        "test_false_positives": 1.0,
        "test_false_negatives": 1.0,
        "test_true_positives": 1.0,
    }

    client = mlflow.tracking.MlflowClient()
    artifact_names = {a.path for a in client.list_artifacts(run_id)}
    assert "confusion_matrix_test.csv" in artifact_names
    assert "confusion_matrix_test.png" in artifact_names


def _synthetic_frame(n_samples, weights, n_features=4, random_state=0):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        weights=weights,
        random_state=random_state,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


def test_train_evaluate_and_log_reports_confusion_matrix_metrics(tmp_path):
    tracking_uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("confusion_matrix_train_pipeline_test")

    X_train, y_train = _synthetic_frame(n_samples=300, weights=[0.75, 0.25], random_state=1)
    X_test, y_test = _synthetic_frame(n_samples=120, weights=[0.75, 0.25], random_state=2)

    pipeline = Pipeline([("model", LogisticRegression(max_iter=1000))])
    model_path = tmp_path / "model.pkl"

    with mlflow.start_run(run_name="test_confusion_matrix_in_pipeline") as run:
        metrics = train_evaluate_and_log(
            pipeline,
            X_train,
            y_train,
            X_test,
            y_test,
            model_path,
            "logistic",
            params={"model_type": "logistic"},
        )
        run_id = run.info.run_id

    for key in (
        "test_true_negatives",
        "test_false_positives",
        "test_false_negatives",
        "test_true_positives",
    ):
        assert key in metrics
        assert metrics[key] >= 0

    # The four counts must sum to exactly len(y_test): computed on the test
    # set only, never mixed in with train-set predictions.
    total = (
        metrics["test_true_negatives"]
        + metrics["test_false_positives"]
        + metrics["test_false_negatives"]
        + metrics["test_true_positives"]
    )
    assert total == len(y_test)

    logged_run = mlflow.get_run(run_id)
    assert logged_run.data.metrics.get("test_true_positives") == metrics["test_true_positives"]

    client = mlflow.tracking.MlflowClient()
    artifact_names = {a.path for a in client.list_artifacts(run_id)}
    assert "confusion_matrix_test.csv" in artifact_names
    assert "confusion_matrix_test.png" in artifact_names
