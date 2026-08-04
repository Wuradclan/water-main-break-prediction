"""
FastAPI inference service for KW water-main break risk classification.

Phase 5:
- Request schema = pipe physical / time-aware features (not aircraft)
- Champion selection via src.model_gate (PR-AUC + F1 overfit gate)
- Response = class label + break probability
"""

from __future__ import annotations

import traceback
from typing import Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
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

# Globals populated at startup / reload
best_model = None
champion_info: Optional[ChampionSelection] = None
model_name_info = "Aucun modèle chargé"


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
    break_within_horizon: int = Field(..., description="Predicted class: 1=break within H years, 0=no")
    probability: float = Field(..., ge=0.0, le=1.0, description="P(break_within_horizon = 1)")
    model_name: str
    model_type: str
    run_id: str
    pr_auc_test: float
    overfit_f1_gap: float
    selection_mode: str


def _features_frame(payload: PipeBreakRequest) -> pd.DataFrame:
    row = payload.model_dump()
    df = pd.DataFrame([row], columns=INFERENCE_INPUT_COLUMNS)
    # Align dtypes with the logged MLflow signature (doubles).
    for col in FEATURE_COLUMNS:
        if col == "material":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df[FEATURE_COLUMNS]


def _predict_proba_positive(model, frame: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)
        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        return float(proba[0])
    if hasattr(model, "decision_function"):
        import math

        score = float(model.decision_function(frame)[0])
        return 1.0 / (1.0 + math.exp(-score))
    # Last resort: hard label as probability proxy.
    return float(model.predict(frame)[0])


def load_best_model_from_mlflow() -> None:
    """Select the gate champion and load its sklearn pipeline from MLflow."""
    global best_model, champion_info, model_name_info

    try:
        tracking_uri = resolve_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        print(f"📡 MLflow tracking URI: {tracking_uri}")

        selection = select_champion_run(tracking_uri=tracking_uri)
        print(explain_selection(selection))

        # Prefer the sklearn flavor so predict_proba remains available.
        model = mlflow.sklearn.load_model(selection.model_uri)

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


# Load champion at startup
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
async def predict(payload: PipeBreakRequest):
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
        label = int(best_model.predict(frame)[0])

        return PipeBreakPredictionResponse(
            break_within_horizon=label,
            probability=probability,
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
