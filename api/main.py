"""
FastAPI inference service for KW water-main break risk classification.

Phase 5:
- Request schema = pipe physical / time-aware features (not aircraft)
- Champion selection via src.model_gate (PR-AUC + F1 overfit gate)
- Response = class label + break probability

Phase 6 (UI entraînement) :
- Jobs d'entraînement asynchrones via docker exec + thread daemon
- Endpoints /start-training et /training-status/{job_id}, pour la
  classification (break_within_horizon) ET la régression (years_until_break)
- Endpoints /models et /models/{run_id} pour lister/supprimer les runs MLflow
"""

from __future__ import annotations

import math
import subprocess
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import mlflow
import mlflow.h2o
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from mlflow.entities import ViewType
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.model_gate import (
    ChampionSelection,
    RegressorChampionSelection,
    explain_regression_selection,
    explain_selection,
    resolve_tracking_uri,
    select_champion_regressor,
    select_champion_run,
)
from src.schema import FEATURE_COLUMNS, INFERENCE_INPUT_COLUMNS, TARGET_COLUMN

try:
    from src.config import CLASSIFICATION_THRESHOLD, MLFLOW_EXPERIMENT_NAME
except ModuleNotFoundError:
    from config import CLASSIFICATION_THRESHOLD, MLFLOW_EXPERIMENT_NAME


app = FastAPI(
    title="KW Water Main Break Risk API",
    description=(
        "Binary classification API: will this pipe break within the next H years? "
        "Champion model is selected by the industrial Model Gate (PR-AUC / F1 overfit)."
    ),
    version="0.5.0",
)

best_model = None
champion_info: Optional[ChampionSelection] = None
model_name_info = "Aucun modèle chargé"

# Regression model (years_until_break), used only when the classifier predicts
# label=1. Loaded independently so classification keeps working even if no
# regression run is available yet.
regressor_model = None
regressor_champion_info: Optional[RegressorChampionSelection] = None
regressor_name_info = "Aucun régresseur chargé"

# ---------------------------------------------------------------------------
# Jobs d'entraînement asynchrones (un seul à la fois, classification OU
# régression) via `docker exec` dans le conteneur trainer.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINER_CONTAINER = "bris-aqueduc-trainer"
# /app/training_jobs quand l'API tourne dans son conteneur Docker (volume
# monté sur tout le repo, cf docker-compose.yml), sinon un dossier local pour
# l'exécution/tests en dehors de Docker.
TRAINING_JOBS_DIR = (
    Path("/app/training_jobs") if Path("/.dockerenv").exists() else PROJECT_ROOT / "training_jobs"
)
TRAINING_JOBS_DIR.mkdir(parents=True, exist_ok=True)

CLASSIFICATION_TRAINING_MODELS = {
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
REGRESSION_TRAINING_MODELS = {
    "linear",
    "ridge_reg",
    "lasso_reg",
    "rf_reg",
    "xgb_reg",
}

# job_id -> métadonnées (status, timestamps, result, etc.)
jobs_status: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tail_log_file(log_path: Path, n_lines: int = 250) -> str:
    """Retourne les ~n_lines dernières lignes d'un fichier de log."""
    if not log_path.exists():
        return ""
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        return "".join(lines[-n_lines:])
    except OSError as exc:
        return f"[ERREUR] Impossible de lire le log : {exc}\n"


def _run_training_job(job_id: str, cmd: list[str], task: str) -> None:
    """Exécute la commande d'entraînement et met à jour jobs_status."""
    log_path = TRAINING_JOBS_DIR / f"{job_id}.log"
    with _jobs_lock:
        jobs_status[job_id]["status"] = "running"
        jobs_status[job_id]["started_at"] = _utc_now_iso()

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(f"$ {' '.join(cmd)}\n")
            log_file.flush()
            process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            return_code = process.wait()
    except FileNotFoundError as exc:
        with _jobs_lock:
            jobs_status[job_id]["status"] = "failed"
            jobs_status[job_id]["finished_at"] = _utc_now_iso()
            jobs_status[job_id]["return_code"] = -1
            jobs_status[job_id]["result"] = None
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(
                f"\n[ERREUR] Commande Docker introuvable dans le conteneur API : {exc}\n"
            )
        return
    except Exception as exc:
        with _jobs_lock:
            jobs_status[job_id]["status"] = "failed"
            jobs_status[job_id]["finished_at"] = _utc_now_iso()
            jobs_status[job_id]["return_code"] = -1
            jobs_status[job_id]["result"] = None
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n[ERREUR] Échec du lancement de l'entraînement : {exc}\n")
        return

    finished_at = _utc_now_iso()
    status = "completed" if return_code == 0 else "failed"
    result = None

    if status == "completed":
        try:
            if task == "regression":
                selection = select_champion_regressor()
            else:
                selection = select_champion_run()
            result = selection.to_dict()
            result["task"] = task
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"\n[Champion] {selection.summary()}\n")
        except Exception as exc:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(
                    f"\n[AVERTISSEMENT] Entraînement OK mais sélection champion échouée : {exc}\n"
                )

    with _jobs_lock:
        jobs_status[job_id]["status"] = status
        jobs_status[job_id]["finished_at"] = finished_at
        jobs_status[job_id]["return_code"] = return_code
        jobs_status[job_id]["result"] = result


class PipeBreakRequest(BaseModel):
    """Kitchener pipe features required for inference (length deferred)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "material": "CI",
                "diameter_mm": 150.0,
                "install_year": 1959.0,
                "age_years": 46.0,
                "prior_break_count": 3.0,
                "years_since_last_break": 0.75,
            }
        },
    )

    material: str = Field(..., description="Pipe material code, e.g. CI, DI, PVC")
    diameter_mm: float = Field(..., gt=0, description="Nominal diameter in millimeters")
    install_year: float = Field(..., ge=1800, le=2100, description="Year the pipe was installed")
    age_years: float = Field(..., ge=0, description="Pipe age in years at prediction time t")
    prior_break_count: float = Field(
        ...,
        ge=0,
        description="Number of recorded breaks strictly before prediction time t",
    )
    years_since_last_break: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Years since the latest prior break (null/omitted when prior_break_count == 0)"
        ),
    )

    @field_validator("material")
    @classmethod
    def normalize_material(cls, value: str) -> str:
        cleaned = str(value).strip().upper()
        if not cleaned or cleaned in {"NAN", "NONE", "NULL", "XXX"}:
            return "UNKNOWN"
        return cleaned


class PipeBreakPredictionResponse(BaseModel):
    break_within_horizon: int = Field(
        ..., description="Predicted class: 1=break within H years, 0=no"
    )
    probability: float = Field(..., ge=0.0, le=1.0, description="P(break_within_horizon = 1)")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Decision threshold used")
    model_name: str
    model_type: str
    run_id: str
    pr_auc_test: float
    overfit_f1_gap: float
    selection_mode: str
    estimated_years_until_break: Optional[float] = Field(
        default=None,
        description=(
            "Estimation du nombre d'années avant la rupture, calculée par le "
            "modèle de régression champion lorsque break_within_horizon == 1 "
            "et qu'un régresseur est chargé."
        ),
    )


def _features_frame(payload: PipeBreakRequest) -> pd.DataFrame:
    row = payload.model_dump()
    df = pd.DataFrame([row], columns=INFERENCE_INPUT_COLUMNS)
    for col in FEATURE_COLUMNS:
        if col == "material":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df[FEATURE_COLUMNS]


def _predict_proba_positive(model, frame: pd.DataFrame) -> float:
    # 1. Priorité absolue à predict_proba si le modèle natif le supporte (Sklearn / XGBoost)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)
        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        return float(proba[0])

    # 2. Si le modèle utilise decision_function (ex: SVC)
    if hasattr(model, "decision_function"):
        import math
        score = float(model.decision_function(frame)[0])
        return 1.0 / (1.0 + math.exp(-score))

    # 3. Fallback pour les prédictions standard (H2O DataFrame ou pyfunc)
    predictions = model.predict(frame)
    
    if isinstance(predictions, pd.DataFrame):
        if "p1" in predictions.columns:
            return float(predictions["p1"].iloc[0])
        return float(predictions.iloc[0, 0])
        
    if isinstance(predictions, pd.Series):
        return float(predictions.iloc[0])
        
    return float(predictions[0])


def load_best_model_from_mlflow() -> None:
    """Select the gate champion and load its native model pipeline from MLflow."""
    global best_model, champion_info, model_name_info

    try:
        tracking_uri = resolve_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        print(f"📡 MLflow tracking URI: {tracking_uri}")

        selection = select_champion_run(tracking_uri=tracking_uri)
        print(explain_selection(selection))

        model_type = selection.model_type.lower()
        if "h2o" in model_type:
            model = mlflow.h2o.load_model(selection.model_uri)
        else:
            try:
                model = mlflow.sklearn.load_model(selection.model_uri)
            except Exception:
                model = mlflow.pyfunc.load_model(selection.model_uri)

        best_model = model
        champion_info = selection
        model_name_info = selection.summary()
        print(f"✅ Champion loaded: {model_name_info}")
    except Exception as exc:
        best_model = None
        champion_info = None
        model_name_info = "Aucun modèle chargé"
        print(f"❌ Failed to load champion model: {exc}")
        traceback.print_exc()


def load_best_regressor_from_mlflow() -> None:
    """Select the regression gate champion and load its native model from MLflow."""
    global regressor_model, regressor_champion_info, regressor_name_info

    try:
        tracking_uri = resolve_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)

        selection = select_champion_regressor(tracking_uri=tracking_uri)
        print(explain_regression_selection(selection))

        try:
            model = mlflow.sklearn.load_model(selection.model_uri)
        except Exception:
            model = mlflow.pyfunc.load_model(selection.model_uri)

        regressor_model = model
        regressor_champion_info = selection
        regressor_name_info = selection.summary()
        print(f"✅ Regression champion loaded: {regressor_name_info}")
    except Exception as exc:
        regressor_model = None
        regressor_champion_info = None
        regressor_name_info = "Aucun régresseur chargé"
        print(f"⚠️  No regression champion loaded (this is OK if no --task regression run exists yet): {exc}")


load_best_model_from_mlflow()
load_best_regressor_from_mlflow()


@app.get("/health")
def health():
    return {
        "status": "ok" if best_model is not None else "degraded",
        "model_loaded": best_model is not None,
        "regressor_loaded": regressor_model is not None,
        "tracking_uri": resolve_tracking_uri(),
        "target": TARGET_COLUMN,
    }


@app.post("/predict", response_model=PipeBreakPredictionResponse)
async def predict(
    payload: PipeBreakRequest,
    threshold: float = Query(CLASSIFICATION_THRESHOLD, ge=0.0, le=1.0),
):
    if best_model is None or champion_info is None:
        raise HTTPException(
            status_code=503,
            detail="No champion model loaded from MLflow. Train a model and/or call /reload-model.",
        )

    # Pre-existing compatibility note: when predict() is invoked directly (e.g.
    # unit tests calling the coroutine without going through FastAPI's request
    # handling), `threshold` still holds its Query(...) sentinel instead of the
    # resolved default. Fall back to 0.5 in that case; real HTTP calls are
    # unaffected since FastAPI already resolves the float before this point.
    if not isinstance(threshold, (int, float)):
        threshold = CLASSIFICATION_THRESHOLD

    try:
        if payload.prior_break_count == 0 and payload.years_since_last_break is not None:
            raise HTTPException(
                status_code=422,
                detail="years_since_last_break must be null/omitted when prior_break_count == 0.",
            )
        if payload.prior_break_count > 0 and payload.years_since_last_break is None:
            raise HTTPException(
                status_code=422,
                detail="years_since_last_break is required when prior_break_count > 0.",
            )

        frame = _features_frame(payload)
        probability = _predict_proba_positive(best_model, frame)
        label = int(probability >= threshold)

        estimated_years_until_break: Optional[float] = None
        if label == 1 and regressor_model is not None:
            try:
                raw_prediction = regressor_model.predict(frame)
                estimated_years_until_break = max(0.0, float(np.asarray(raw_prediction, dtype=float).ravel()[0]))
            except Exception as exc:
                print(f"⚠️  Regression estimate failed: {exc}")
                estimated_years_until_break = None

        return PipeBreakPredictionResponse(
            break_within_horizon=label,
            probability=probability,
            threshold=threshold,
            model_name=model_name_info,
            model_type=champion_info.model_type,
            run_id=champion_info.run_id,
            pr_auc_test=champion_info.pr_auc_test,
            overfit_f1_gap=champion_info.overfit_f1_gap,
            selection_mode=champion_info.selection_mode,
            estimated_years_until_break=estimated_years_until_break,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.get("/model-info")
def get_model_info():
    if best_model is None or champion_info is None:
        return {"status": "error", "model_name": "Aucun modèle"}
    return {
        "status": "success",
        "model_name": model_name_info,
        "champion": champion_info.to_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "tracking_uri": resolve_tracking_uri(),
    }


@app.get("/regressor-info")
def get_regressor_info():
    """Champion info for the years_until_break regressor (analogous to /model-info)."""
    if regressor_model is None or regressor_champion_info is None:
        return {"status": "error", "model_name": "Aucun régresseur"}
    return {
        "status": "success",
        "model_name": regressor_name_info,
        "champion": regressor_champion_info.to_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "tracking_uri": resolve_tracking_uri(),
    }


# Design choice: /reload-model reloads BOTH the classifier and the regressor
# (they share the same MLflow tracking store and are refreshed together from
# Streamlit's single "Recharger" button). A dedicated /reload-regressor is
# also exposed for callers who only want to refresh the regressor without
# touching the classification champion (e.g. after training only a new
# --task regression run).
@app.post("/reload-model")
def reload_model():
    try:
        load_best_model_from_mlflow()
        load_best_regressor_from_mlflow()
        if best_model is None or champion_info is None:
            return {
                "status": "error",
                "message": "Failed to load champion. Check MLflow tracking URI and runs.",
            }
        response = {
            "status": "success",
            "message": f"Loaded champion '{model_name_info}'.",
            "champion": champion_info.to_dict(),
        }
        if regressor_model is not None and regressor_champion_info is not None:
            response["regressor_champion"] = regressor_champion_info.to_dict()
        return response
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/reload-regressor")
def reload_regressor():
    try:
        load_best_regressor_from_mlflow()
        if regressor_model is None or regressor_champion_info is None:
            return {
                "status": "error",
                "message": "Failed to load regression champion. Train with --task regression first.",
            }
        return {
            "status": "success",
            "message": f"Loaded regression champion '{regressor_name_info}'.",
            "champion": regressor_champion_info.to_dict(),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/start-training")
def start_training(
    task: str = Query(
        "classification",
        description="Tâche à entraîner : 'classification' (break_within_horizon) ou 'regression' (years_until_break)",
    ),
    model_type: str = Query(
        ...,
        description="Type de modèle (dépend de task ; ex: xgboost pour classification, xgb_reg pour regression)",
    ),
    tune: bool = Query(False, description="Activer Optuna"),
    n_trials: int = Query(15, ge=1, le=200, description="Nombre d'essais Optuna"),
    horizon_years: int = Query(
        5, description="Horizon de prédiction (1, 2 ou 5 ans) ; utilisé uniquement pour task=classification"
    ),
):
    """
    Lance un entraînement en arrière-plan dans le conteneur trainer.
    Répond immédiatement avec un job_id (ne bloque pas).
    """
    # Pre-existing compatibility note (see predict()): when this endpoint is
    # invoked directly (e.g. unit tests calling the function without going
    # through FastAPI's request handling), optional Query(...) parameters
    # that weren't explicitly passed still hold their sentinel FieldInfo
    # instead of the resolved default. Fall back to the documented defaults
    # in that case; real HTTP calls are unaffected since FastAPI already
    # resolves these values before this point.
    if not isinstance(task, str):
        task = "classification"
    if not isinstance(tune, bool):
        tune = False
    if not isinstance(n_trials, int):
        n_trials = 15
    if not isinstance(horizon_years, int):
        horizon_years = 5

    if task not in {"classification", "regression"}:
        raise HTTPException(
            status_code=422,
            detail=f"task invalide : {task!r}. Choix : ['classification', 'regression'].",
        )

    allowed_models = (
        CLASSIFICATION_TRAINING_MODELS if task == "classification" else REGRESSION_TRAINING_MODELS
    )
    if model_type not in allowed_models:
        raise HTTPException(
            status_code=422,
            detail=f"model_type invalide pour task={task!r} : {model_type!r}. Choix : {sorted(allowed_models)}",
        )
    if task == "classification" and horizon_years not in {1, 2, 5}:
        raise HTTPException(
            status_code=422,
            detail="horizon_years doit être 1, 2 ou 5.",
        )

    # h2o (AutoML) ne supporte pas Optuna ; tous les modèles de régression le supportent.
    tune_supported = model_type != "h2o"
    effective_tune = bool(tune and tune_supported)

    with _jobs_lock:
        running = [
            jid
            for jid, meta in jobs_status.items()
            if meta.get("status") == "running"
        ]
        if running:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Un entraînement est déjà en cours (job_id={running[0]}). "
                    "Attends sa fin avant d'en lancer un autre."
                ),
            )

        job_id = str(uuid.uuid4())[:8]
        cmd = [
            "docker",
            "exec",
            TRAINER_CONTAINER,
            "python",
            "-m",
            "src.train",
            "--task",
            task,
            "--model_type",
            model_type,
        ]
        if task == "classification":
            cmd.extend(["--horizon_years", str(horizon_years)])
        if effective_tune:
            cmd.extend(["--tune", "--n_trials", str(n_trials)])

        jobs_status[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "task": task,
            "model_type": model_type,
            "tune": effective_tune,
            "n_trials": n_trials if effective_tune else None,
            "horizon_years": horizon_years if task == "classification" else None,
            "started_at": None,
            "finished_at": None,
            "return_code": None,
            "result": None,
            "command": cmd,
        }

    thread = threading.Thread(
        target=_run_training_job,
        args=(job_id, cmd, task),
        daemon=True,
        name=f"training-{job_id}",
    )
    thread.start()

    return {"job_id": job_id, "status": "pending", "task": task}


@app.get("/training-status/{job_id}")
def training_status(job_id: str):
    """Retourne l'état d'un job d'entraînement et la queue des logs."""
    with _jobs_lock:
        meta = jobs_status.get(job_id)
        if meta is None:
            raise HTTPException(
                status_code=404,
                detail=f"Aucun job d'entraînement trouvé pour job_id={job_id!r}.",
            )
        snapshot = dict(meta)

    log_tail = _tail_log_file(TRAINING_JOBS_DIR / f"{job_id}.log", n_lines=250)

    return {
        "job_id": job_id,
        "status": snapshot.get("status"),
        "task": snapshot.get("task"),
        "model_type": snapshot.get("model_type"),
        "tune": snapshot.get("tune"),
        "n_trials": snapshot.get("n_trials"),
        "horizon_years": snapshot.get("horizon_years"),
        "started_at": snapshot.get("started_at"),
        "finished_at": snapshot.get("finished_at"),
        "return_code": snapshot.get("return_code"),
        "log_tail": log_tail,
        "result": snapshot.get("result"),
    }


def json_safe_float(value):
    """
    Coerce an MLflow metric value into a strict-JSON-compliant float.

    mlflow.search_runs() returns a DataFrame where metrics missing for a
    given run (e.g. rmse_test on a classification run, pr_auc_test on a
    regression run) show up as NaN. Python's json module (which FastAPI's
    default encoder uses under the hood) refuses to serialize NaN/Infinity
    ("Out of range float values are not JSON compliant"), so every metric
    must be sanitized before it reaches the response body.
    """
    if value is None:
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if math.isfinite(numeric_value) else None


def _safe_str(value, default: str = "—") -> str:
    """String-ify a pandas cell, falling back to `default` for NaN/NA/None."""
    return str(value) if pd.notna(value) else default


@app.get("/models")
def list_models(
    include_deleted: bool = Query(False, description="Inclure les runs déjà supprimés (soft delete)"),
):
    """
    Liste tous les runs MLflow de premier niveau (hors essais Optuna 'trial_*'),
    avec leurs métriques principales et un indicateur si c'est le champion actif.

    Couvre à la fois les runs de classification (break_within_horizon) et de
    régression (years_until_break, params.task == 'regression'). Les deux
    familles de runs ont des métriques disjointes (PR-AUC/F1/ROC-AUC vs
    RMSE/MAE/R2) : les métriques non applicables à un run sont renvoyées à
    `null` plutôt qu'omises, pour que la page Streamlit affiche « — ».
    """
    try:
        tracking_uri = resolve_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)

        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            return {"status": "error", "message": "Expérience MLflow introuvable.", "models": []}

        view_type = ViewType.ALL if include_deleted else ViewType.ACTIVE_ONLY
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            run_view_type=view_type,
            max_results=300,
            order_by=["start_time DESC"],
        )

        if runs.empty:
            return {"status": "success", "models": []}

        # Exclure les essais Optuna imbriqués pour garder une liste lisible
        if "tags.mlflow.parentRunId" in runs.columns:
            runs = runs[runs["tags.mlflow.parentRunId"].isnull()]
        if "tags.mlflow.runName" in runs.columns:
            runs = runs[~runs["tags.mlflow.runName"].astype(str).str.startswith("trial_")]

        current_champion_id = champion_info.run_id if champion_info is not None else None
        current_regressor_champion_id = (
            regressor_champion_info.run_id if regressor_champion_info is not None else None
        )

        models = []
        for _, row in runs.iterrows():
            run_id = row.get("run_id")
            raw_task = row.get("params.task")
            task = str(raw_task) if pd.notna(raw_task) else "classification"

            raw_horizon = row.get("params.horizon_years")
            horizon_years = None
            if pd.notna(raw_horizon):
                try:
                    horizon_years = int(float(raw_horizon))
                except (TypeError, ValueError):
                    horizon_years = None

            raw_start_time = row.get("start_time")
            start_time = str(raw_start_time) if pd.notna(raw_start_time) else None

            models.append(
                {
                    "run_id": run_id,
                    "run_name": _safe_str(row.get("tags.mlflow.runName")),
                    "task": task,
                    "model_type": _safe_str(row.get("params.model_type")),
                    "horizon_years": horizon_years,
                    "pr_auc_test": json_safe_float(row.get("metrics.pr_auc_test")),
                    "f1_train": json_safe_float(row.get("metrics.f1_train")),
                    "f1_test": json_safe_float(row.get("metrics.f1_test")),
                    "roc_auc_test": json_safe_float(row.get("metrics.roc_auc_test")),
                    "rmse_test": json_safe_float(row.get("metrics.rmse_test")),
                    "mae_test": json_safe_float(row.get("metrics.mae_test")),
                    "r2_test": json_safe_float(row.get("metrics.r2_test")),
                    "start_time": start_time,
                    "status": _safe_str(row.get("status")),
                    "is_current_champion": run_id == current_champion_id
                    or run_id == current_regressor_champion_id,
                }
            )

        return {"status": "success", "models": models}

    except Exception as exc:
        return {"status": "error", "message": str(exc), "models": []}


@app.delete("/models/{run_id}")
def delete_model(
    run_id: str,
    force: bool = Query(False, description="Confirme la suppression même si c'est le champion actif"),
):
    """
    Supprime (soft delete MLflow) un run par son run_id.

    Protège contre la suppression accidentelle du champion actuellement chargé
    en mémoire (classification ou régression) : il faut passer force=true
    explicitement pour le supprimer.

    Note : MLflow marque le run comme 'deleted' (lifecycle_stage) mais les
    artefacts restent sur disque jusqu'à l'exécution de `mlflow gc` côté serveur.
    """
    is_classification_champion = champion_info is not None and run_id == champion_info.run_id
    is_regression_champion = (
        regressor_champion_info is not None and run_id == regressor_champion_info.run_id
    )

    if (is_classification_champion or is_regression_champion) and not force:
        raise HTTPException(
            status_code=409,
            detail=(
                "Ce run est un champion actuellement chargé en mémoire par l'API. "
                "Relance la requête avec ?force=true pour confirmer la suppression."
            ),
        )

    try:
        tracking_uri = resolve_tracking_uri()
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        client.delete_run(run_id)

        return {
            "status": "success",
            "message": f"Run {run_id} supprimé (soft delete MLflow).",
            "was_champion": is_classification_champion or is_regression_champion,
        }

    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Échec de la suppression : {exc}")