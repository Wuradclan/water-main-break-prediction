"""
FastAPI inference service for KW water-main break risk classification.

Phase 5:
- Request schema = pipe physical / time-aware features (not aircraft)
- Champion selection via src.model_gate (PR-AUC + F1 overfit gate)
- Response = class label + break probability

Phase 6 (UI entraînement) :
- Jobs d'entraînement asynchrones via docker exec + thread daemon
- Endpoints /start-training et /training-status/{job_id}
"""

from __future__ import annotations

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
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.model_gate import (
    ChampionSelection,
    explain_selection,
    resolve_tracking_uri,
    select_champion_run,
)
from src.schema import FEATURE_COLUMNS, INFERENCE_INPUT_COLUMNS, TARGET_COLUMN


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

# ---------------------------------------------------------------------------
# Jobs d'entraînement asynchrones (un seul à la fois)
# ---------------------------------------------------------------------------
TRAINER_CONTAINER = "bris-aqueduc-trainer"
TRAINING_JOBS_DIR = Path("/app/training_jobs")
TRAINING_JOBS_DIR.mkdir(parents=True, exist_ok=True)

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


def _run_training_job(job_id: str, cmd: list[str]) -> None:
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
            selection = select_champion_run()
            result = selection.to_dict()
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

load_best_model_from_mlflow()


@app.get("/health")
def health():
    return {
        "status": "ok" if best_model is not None else "degraded",
        "model_loaded": best_model is not None,
        "tracking_uri": resolve_tracking_uri(),
        "target": TARGET_COLUMN,
    }


@app.post("/predict", response_model=PipeBreakPredictionResponse)
async def predict(
    payload: PipeBreakRequest,
    threshold: float = Query(0.5, ge=0.0, le=1.0),
):
    if best_model is None or champion_info is None:
        raise HTTPException(
            status_code=503,
            detail="No champion model loaded from MLflow. Train a model and/or call /reload-model.",
        )

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


@app.post("/reload-model")
def reload_model():
    try:
        load_best_model_from_mlflow()
        if best_model is None or champion_info is None:
            return {
                "status": "error",
                "message": "Failed to load champion. Check MLflow tracking URI and runs.",
            }
        return {
            "status": "success",
            "message": f"Loaded champion '{model_name_info}'.",
            "champion": champion_info.to_dict(),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/start-training")
def start_training(
    model_type: str = Query(
        ...,
        description="Type de modèle (logistic, xgboost, h2o, ...)",
    ),
    tune: bool = Query(False, description="Activer Optuna"),
    n_trials: int = Query(15, ge=1, le=200, description="Nombre d'essais Optuna"),
    horizon_years: int = Query(5, description="Horizon de prédiction (1, 2 ou 5 ans)"),
):
    """
    Lance un entraînement en arrière-plan dans le conteneur trainer.
    Répond immédiatement avec un job_id (ne bloque pas).
    """
    allowed_models = {
        "logistic",
        "ridge",
        "lasso",
        "random_forest",
        "xgboost",
        "extra_trees",
        "knn",
        "svc",
        "mlp",
        "stacking",
        "h2o",
    }
    if model_type not in allowed_models:
        raise HTTPException(
            status_code=422,
            detail=f"model_type invalide : {model_type!r}. Choix : {sorted(allowed_models)}",
        )
    if horizon_years not in {1, 2, 5}:
        raise HTTPException(
            status_code=422,
            detail="horizon_years doit être 1, 2 ou 5.",
        )

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
            "--model_type",
            model_type,
            "--horizon_years",
            str(horizon_years),
        ]
        if tune and model_type != "h2o":
            cmd.extend(["--tune", "--n_trials", str(n_trials)])

        jobs_status[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "model_type": model_type,
            "tune": tune and model_type != "h2o",
            "n_trials": n_trials if (tune and model_type != "h2o") else None,
            "horizon_years": horizon_years,
            "started_at": None,
            "finished_at": None,
            "return_code": None,
            "result": None,
            "command": cmd,
        }

    thread = threading.Thread(
        target=_run_training_job,
        args=(job_id, cmd),
        daemon=True,
        name=f"training-{job_id}",
    )
    thread.start()

    return {"job_id": job_id, "status": "pending"}


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