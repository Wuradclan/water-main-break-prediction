"""Unit tests for probability calibration in src/train.py.

Covers:
- choose_calibration_method: sigmoid on small/imbalanced data, isotonic once
  there is enough (balanced) data.
- train_evaluate_and_log: the model actually saved/served is the calibrated
  CalibratedClassifierCV wrapper (not the raw, uncalibrated pipeline), and
  Brier scores are reported alongside the usual classification metrics.
"""

from __future__ import annotations

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.train import choose_calibration_method, train_evaluate_and_log


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


def test_choose_calibration_method_prefers_sigmoid_on_small_imbalanced_data():
    _, y = _synthetic_frame(n_samples=300, weights=[0.9, 0.1])
    assert choose_calibration_method(y) == "sigmoid"


def test_choose_calibration_method_allows_isotonic_on_large_balanced_data():
    _, y = _synthetic_frame(n_samples=4000, weights=[0.5, 0.5])
    assert choose_calibration_method(y) == "isotonic"


def test_choose_calibration_method_handles_empty_input():
    assert choose_calibration_method(np.array([])) == "sigmoid"


@pytest.fixture()
def mlflow_tmp_tracking(tmp_path, monkeypatch):
    tracking_uri = f"file://{tmp_path}/mlruns"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("calibration_test_experiment")
    return tracking_uri


def test_train_evaluate_and_log_saves_calibrated_model_with_brier_scores(tmp_path, mlflow_tmp_tracking):
    X_train, y_train = _synthetic_frame(n_samples=400, weights=[0.75, 0.25], random_state=1)
    X_test, y_test = _synthetic_frame(n_samples=150, weights=[0.75, 0.25], random_state=2)

    pipeline = Pipeline([("model", LogisticRegression(max_iter=1000))])
    model_path = tmp_path / "model.pkl"

    with mlflow.start_run(run_name="test_calibrated_logistic"):
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

    # Brier score is reported (calibration quality evaluation) and bounded.
    assert "brier_train" in metrics
    assert "brier_test" in metrics
    assert 0.0 <= metrics["brier_train"] <= 1.0
    assert 0.0 <= metrics["brier_test"] <= 1.0

    # The persisted artifact — what /predict loads and calls predict_proba on
    # — is the calibrated wrapper, not the bare, uncalibrated pipeline.
    saved_model = joblib.load(model_path)
    assert isinstance(saved_model, CalibratedClassifierCV)

    # The original (unfitted) `pipeline` object passed in must never be
    # mutated into "the" served model: CalibratedClassifierCV clones it.
    assert saved_model is not pipeline


def test_train_evaluate_and_log_calibrated_probabilities_differ_from_raw_pipeline(
    tmp_path, mlflow_tmp_tracking
):
    """Regression test for requirement #10: the probability shown to users
    must come from the calibrated model, not from an uncalibrated fit of the
    same estimator on the same data."""
    X_train, y_train = _synthetic_frame(n_samples=400, weights=[0.8, 0.2], random_state=3)
    X_test, y_test = _synthetic_frame(n_samples=150, weights=[0.8, 0.2], random_state=4)

    pipeline = Pipeline([("model", LogisticRegression(max_iter=1000))])
    model_path = tmp_path / "model.pkl"

    with mlflow.start_run(run_name="test_calibrated_vs_raw"):
        train_evaluate_and_log(
            pipeline,
            X_train,
            y_train,
            X_test,
            y_test,
            model_path,
            "logistic",
            params={"model_type": "logistic"},
        )

    calibrated_model = joblib.load(model_path)

    # Fit an independent, uncalibrated copy of the same estimator on the same
    # training data, purely as a reference point to prove the two are not
    # numerically identical (calibration actually re-maps the probabilities).
    raw_reference = Pipeline([("model", LogisticRegression(max_iter=1000))])
    raw_reference.fit(X_train, y_train)

    calibrated_proba = calibrated_model.predict_proba(X_test)[:, 1]
    raw_proba = raw_reference.predict_proba(X_test)[:, 1]

    assert not np.allclose(calibrated_proba, raw_proba)


def test_calibration_method_is_logged_as_a_param(tmp_path, mlflow_tmp_tracking):
    X_train, y_train = _synthetic_frame(n_samples=300, weights=[0.9, 0.1], random_state=5)
    X_test, y_test = _synthetic_frame(n_samples=100, weights=[0.9, 0.1], random_state=6)

    pipeline = Pipeline([("model", LogisticRegression(max_iter=1000))])
    model_path = tmp_path / "model.pkl"

    with mlflow.start_run(run_name="test_calibration_method_logged") as run:
        train_evaluate_and_log(
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

    logged_run = mlflow.get_run(run_id)
    assert logged_run.data.params.get("calibration_method") == "sigmoid"
