"""
Tests for the training/model-management endpoints restored in api/main.py
after the feature/training-ui-regression merge conflict:

    GET    /models
    DELETE /models/{run_id}
    POST   /start-training  (now task-aware: classification | regression)
    GET    /training-status/{job_id}

These tests avoid any real `docker exec` calls (the trainer container is not
running in CI) by monkeypatching either the background thread target
(`_run_training_job`) or `subprocess.Popen` directly, and use a throwaway
local MLflow file store for the /models list/delete lifecycle.
"""

from __future__ import annotations

import mlflow
import pytest
from fastapi import HTTPException

import api.main as api_main
from api.main import (
    CLASSIFICATION_TRAINING_MODELS,
    REGRESSION_TRAINING_MODELS,
    delete_model,
    list_models,
    start_training,
    training_status,
)
from src.model_gate import ChampionSelection, RegressorChampionSelection


@pytest.fixture(autouse=True)
def _clean_jobs_status():
    """Isolate each test's view of the in-memory jobs registry."""
    api_main.jobs_status.clear()
    yield
    api_main.jobs_status.clear()


def _noop_run_training_job(job_id, cmd, task):
    """Stand-in for the background thread target: never touches docker/mlflow."""
    return None


def _fake_classification_selection(run_id: str = "run123") -> ChampionSelection:
    return ChampionSelection(
        run_id=run_id,
        run_name="run_test",
        model_type="xgboost",
        model_uri=f"runs:/{run_id}/model",
        horizon_years=5,
        pr_auc_test=0.9,
        f1_train=0.8,
        f1_test=0.75,
        overfit_f1_gap=0.05,
        roc_auc_test=0.85,
        recall_at_k_test=0.6,
        passes_overfit_gate=True,
        selection_mode="champion",
        overfit_threshold=0.30,
        n_candidates=3,
        n_rejected_overfit=0,
    )


def _fake_regression_selection(run_id: str = "runreg1") -> RegressorChampionSelection:
    return RegressorChampionSelection(
        run_id=run_id,
        run_name="run_reg_test",
        model_type="xgb_reg",
        model_uri=f"runs:/{run_id}/model",
        rmse_test=1.2,
        mae_test=0.8,
        r2_test=0.6,
    )


class _FakeProcess:
    def __init__(self, returncode: int = 0):
        self.returncode = returncode

    def wait(self) -> int:
        return self.returncode


# ---------------------------------------------------------------------------
# POST /start-training — validation
# ---------------------------------------------------------------------------


def test_allowed_training_models_match_spec():
    assert CLASSIFICATION_TRAINING_MODELS == {
        "logistic",
        "ridge",
        "lasso",
        "random_forest",
        "extra_trees",
        "xgboost",
        "knn",
        "svc",
        "mlp",
        "stacking",
        "h2o",
    }
    assert REGRESSION_TRAINING_MODELS == {"linear", "ridge_reg", "lasso_reg", "rf_reg", "xgb_reg"}


def test_start_training_rejects_invalid_task(monkeypatch):
    monkeypatch.setattr(api_main, "_run_training_job", _noop_run_training_job)
    with pytest.raises(HTTPException) as exc_info:
        start_training(task="anomaly_detection", model_type="xgboost")
    assert exc_info.value.status_code == 422


def test_start_training_rejects_model_type_for_wrong_task(monkeypatch):
    monkeypatch.setattr(api_main, "_run_training_job", _noop_run_training_job)
    with pytest.raises(HTTPException) as exc_info:
        start_training(task="regression", model_type="xgboost")  # classification-only model
    assert exc_info.value.status_code == 422

    with pytest.raises(HTTPException) as exc_info:
        start_training(task="classification", model_type="xgb_reg")  # regression-only model
    assert exc_info.value.status_code == 422


def test_start_training_rejects_bad_horizon_for_classification(monkeypatch):
    monkeypatch.setattr(api_main, "_run_training_job", _noop_run_training_job)
    with pytest.raises(HTTPException) as exc_info:
        start_training(task="classification", model_type="xgboost", horizon_years=3)
    assert exc_info.value.status_code == 422


def test_start_training_rejects_concurrent_jobs(monkeypatch):
    monkeypatch.setattr(api_main, "_run_training_job", _noop_run_training_job)
    api_main.jobs_status["existing"] = {"status": "running"}
    with pytest.raises(HTTPException) as exc_info:
        start_training(task="classification", model_type="xgboost")
    assert exc_info.value.status_code == 409


# ---------------------------------------------------------------------------
# POST /start-training — docker command construction
# ---------------------------------------------------------------------------


def test_start_training_classification_builds_expected_docker_command(monkeypatch):
    monkeypatch.setattr(api_main, "_run_training_job", _noop_run_training_job)
    response = start_training(
        task="classification",
        model_type="xgboost",
        tune=True,
        n_trials=15,
        horizon_years=5,
    )
    assert response["task"] == "classification"
    job_id = response["job_id"]
    assert api_main.jobs_status[job_id]["command"] == [
        "docker",
        "exec",
        api_main.TRAINER_CONTAINER,
        "python",
        "-m",
        "src.train",
        "--task",
        "classification",
        "--model_type",
        "xgboost",
        "--horizon_years",
        "5",
        "--tune",
        "--n_trials",
        "15",
    ]


def test_start_training_regression_builds_expected_docker_command_without_horizon(monkeypatch):
    monkeypatch.setattr(api_main, "_run_training_job", _noop_run_training_job)
    response = start_training(task="regression", model_type="xgb_reg", tune=True, n_trials=20)
    assert response["task"] == "regression"
    job_id = response["job_id"]
    cmd = api_main.jobs_status[job_id]["command"]
    assert "--horizon_years" not in cmd
    assert cmd == [
        "docker",
        "exec",
        api_main.TRAINER_CONTAINER,
        "python",
        "-m",
        "src.train",
        "--task",
        "regression",
        "--model_type",
        "xgb_reg",
        "--tune",
        "--n_trials",
        "20",
    ]
    assert api_main.jobs_status[job_id]["horizon_years"] is None


def test_start_training_h2o_ignores_tune_flag(monkeypatch):
    monkeypatch.setattr(api_main, "_run_training_job", _noop_run_training_job)
    response = start_training(task="classification", model_type="h2o", tune=True, n_trials=10)
    job_id = response["job_id"]
    cmd = api_main.jobs_status[job_id]["command"]
    assert "--tune" not in cmd
    assert "--n_trials" not in cmd
    assert api_main.jobs_status[job_id]["tune"] is False


# ---------------------------------------------------------------------------
# GET /training-status/{job_id}
# ---------------------------------------------------------------------------


def test_training_status_unknown_job_raises_404():
    with pytest.raises(HTTPException) as exc_info:
        training_status("does-not-exist")
    assert exc_info.value.status_code == 404


def test_training_status_reports_classification_champion(monkeypatch):
    fake_selection = _fake_classification_selection()
    monkeypatch.setattr(api_main, "select_champion_run", lambda **kwargs: fake_selection)
    monkeypatch.setattr(api_main.subprocess, "Popen", lambda *a, **kw: _FakeProcess(0))

    job_id = "job-classif"
    api_main.jobs_status[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "task": "classification",
        "model_type": "xgboost",
        "tune": False,
        "n_trials": None,
        "horizon_years": 5,
        "started_at": None,
        "finished_at": None,
        "return_code": None,
        "result": None,
        "command": ["echo", "ok"],
    }
    api_main._run_training_job(job_id, ["echo", "ok"], "classification")

    body = training_status(job_id)
    assert body["task"] == "classification"
    assert body["status"] == "completed"
    assert body["result"]["run_id"] == "run123"
    assert body["result"]["task"] == "classification"


def test_training_status_reports_regression_champion(monkeypatch):
    fake_selection = _fake_regression_selection()
    monkeypatch.setattr(api_main, "select_champion_regressor", lambda **kwargs: fake_selection)
    monkeypatch.setattr(api_main.subprocess, "Popen", lambda *a, **kw: _FakeProcess(0))

    job_id = "job-reg"
    api_main.jobs_status[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "task": "regression",
        "model_type": "xgb_reg",
        "tune": False,
        "n_trials": None,
        "horizon_years": None,
        "started_at": None,
        "finished_at": None,
        "return_code": None,
        "result": None,
        "command": ["echo", "ok"],
    }
    api_main._run_training_job(job_id, ["echo", "ok"], "regression")

    body = training_status(job_id)
    assert body["task"] == "regression"
    assert body["status"] == "completed"
    assert body["result"]["run_id"] == "runreg1"
    assert body["result"]["task"] == "regression"
    assert body["result"]["rmse_test"] == pytest.approx(1.2)


def test_training_status_marks_job_failed_on_nonzero_return_code(monkeypatch):
    monkeypatch.setattr(api_main.subprocess, "Popen", lambda *a, **kw: _FakeProcess(1))

    job_id = "job-fail"
    api_main.jobs_status[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "task": "classification",
        "model_type": "xgboost",
        "tune": False,
        "n_trials": None,
        "horizon_years": 5,
        "started_at": None,
        "finished_at": None,
        "return_code": None,
        "result": None,
        "command": ["echo", "fail"],
    }
    api_main._run_training_job(job_id, ["echo", "fail"], "classification")

    body = training_status(job_id)
    assert body["status"] == "failed"
    assert body["return_code"] == 1
    assert body["result"] is None


# ---------------------------------------------------------------------------
# GET /models, DELETE /models/{run_id}
# ---------------------------------------------------------------------------


def test_models_list_and_delete_lifecycle(tmp_path, monkeypatch):
    tracking_uri = f"file://{tmp_path}/mlruns"
    monkeypatch.setattr(api_main, "resolve_tracking_uri", lambda *a, **kw: tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(api_main.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="run_01_logistic") as run_a:
        mlflow.log_param("model_type", "logistic")
        mlflow.log_param("horizon_years", 5)
        mlflow.log_metric("pr_auc_test", 0.7)
        mlflow.log_metric("f1_train", 0.6)
        mlflow.log_metric("f1_test", 0.55)
        run_a_id = run_a.info.run_id

    with mlflow.start_run(run_name="run_reg_linear") as run_b:
        mlflow.log_param("model_type", "linear")
        mlflow.log_param("task", "regression")
        mlflow.log_metric("rmse_test", 1.2)
        mlflow.log_metric("mae_test", 0.9)
        mlflow.log_metric("r2_test", 0.5)
        run_b_id = run_b.info.run_id

    # Simulate that run_a is the classification champion currently loaded in memory.
    monkeypatch.setattr(
        api_main,
        "champion_info",
        ChampionSelection(
            run_id=run_a_id,
            run_name="run_01_logistic",
            model_type="logistic",
            model_uri=f"runs:/{run_a_id}/model",
            horizon_years=5,
            pr_auc_test=0.7,
            f1_train=0.6,
            f1_test=0.55,
            overfit_f1_gap=0.05,
            roc_auc_test=None,
            recall_at_k_test=None,
            passes_overfit_gate=True,
            selection_mode="champion",
            overfit_threshold=0.30,
            n_candidates=1,
            n_rejected_overfit=0,
        ),
    )

    body = list_models()
    assert body["status"] == "success"
    by_id = {m["run_id"]: m for m in body["models"]}
    assert run_a_id in by_id
    assert run_b_id in by_id
    assert by_id[run_a_id]["is_current_champion"] is True
    assert by_id[run_a_id]["task"] == "classification"
    assert by_id[run_b_id]["task"] == "regression"
    assert by_id[run_b_id]["is_current_champion"] is False

    # Deleting the active champion without force is rejected.
    with pytest.raises(HTTPException) as exc_info:
        delete_model(run_a_id, force=False)
    assert exc_info.value.status_code == 409

    # Deleting a non-champion run succeeds immediately.
    result = delete_model(run_b_id, force=False)
    assert result["status"] == "success"
    assert result["was_champion"] is False

    remaining = {m["run_id"] for m in list_models(include_deleted=False)["models"]}
    assert run_b_id not in remaining
    assert run_a_id in remaining

    everything = {m["run_id"] for m in list_models(include_deleted=True)["models"]}
    assert run_b_id in everything

    # Deleting the champion is allowed once force=True is passed.
    forced = delete_model(run_a_id, force=True)
    assert forced["status"] == "success"
    assert forced["was_champion"] is True


def test_list_models_returns_error_when_experiment_missing(tmp_path, monkeypatch):
    tracking_uri = f"file://{tmp_path}/empty_mlruns"
    monkeypatch.setattr(api_main, "resolve_tracking_uri", lambda *a, **kw: tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    body = list_models()
    assert body["status"] == "error"
    assert body["models"] == []
