"""
FastAPI inference service for KW water-main break risk classification.

Phase 5:
- Request schema = pipe physical / time-aware features (not aircraft)
- Champion selection via src.model_gate (PR-AUC + F1 overfit gate)
- Response = class label + break probability
- Optional years_until_break estimate from regression champion when label=1
"""

from __future__ import annotations

import traceback
from typing import Optional

import mlflow
import mlflow.h2o
import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.model_gate import (
    ChampionSelection,
    RegressorChampionSelection,
    explain_regressor_selection,
    explain_selection,
    resolve_tracking_uri,
    select_champion_regressor,
    select_champion_run,
)
from src.schema import FEATURE_COLUMNS, INFERENCE_INPUT_COLUMNS, TARGET_COLUMN


app = FastAPI(
    title="KW Water Main Break Risk API",
    description=(
        "Binary classification API: will this pipe break within the next H years? "
        "Champion model is selected by the industrial Model Gate (PR-AUC / F1 overfit). "
        "When the classifier predicts break (label=1), an optional regressor estimates "
        "years_until_break."
    ),
    version="0.6.0",
)

best_model = None
champion_info: Optional[ChampionSelection] = None
model_name_info = "Aucun modèle chargé"

regressor_model = None
regressor_champion_info: Optional[RegressorChampionSelection] = None
regressor_name_info = "Aucun régresseur chargé"


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
            "Estimated years until break from the regression champion; "
            "populated only when break_within_horizon=1 and a regressor is loaded"
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
    """Select the regression champion (lowest RMSE) and load it from MLflow."""
    global regressor_model, regressor_champion_info, regressor_name_info

    try:
        tracking_uri = resolve_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)

        selection = select_champion_regressor(tracking_uri=tracking_uri)
        print(explain_regressor_selection(selection))

        try:
            model = mlflow.sklearn.load_model(selection.model_uri)
        except Exception:
            model = mlflow.pyfunc.load_model(selection.model_uri)

        regressor_model = model
        regressor_champion_info = selection
        regressor_name_info = selection.summary()
        print(f"✅ Regressor champion loaded: {regressor_name_info}")
    except Exception as exc:
        regressor_model = None
        regressor_champion_info = None
        regressor_name_info = "Aucun régresseur chargé"
        print(f"⚠️ Failed to load regressor champion (classification still available): {exc}")
        traceback.print_exc()


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

        estimated_years: Optional[float] = None
        if label == 1 and regressor_model is not None:
            raw_pred = regressor_model.predict(frame)
            if isinstance(raw_pred, pd.DataFrame):
                value = float(raw_pred.iloc[0, 0])
            elif isinstance(raw_pred, pd.Series):
                value = float(raw_pred.iloc[0])
            else:
                value = float(np.asarray(raw_pred).ravel()[0])
            estimated_years = float(max(0.0, value))

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
            estimated_years_until_break=estimated_years,
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
    if regressor_model is None or regressor_champion_info is None:
        return {"status": "error", "model_name": "Aucun régresseur"}
    return {
        "status": "success",
        "model_name": regressor_name_info,
        "champion": regressor_champion_info.to_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "tracking_uri": resolve_tracking_uri(),
    }


@app.post("/reload-model")
def reload_model():
    """
    Reload both the classification champion and the regression champion.

    Choice: a single /reload-model endpoint reloads classifier + regressor so the
    Streamlit "Recharger le champion" button keeps working without a second call.
    Use GET /regressor-info to inspect the regressor independently.
    """
    try:
        load_best_model_from_mlflow()
        load_best_regressor_from_mlflow()
        if best_model is None or champion_info is None:
            return {
                "status": "error",
                "message": "Failed to load classification champion. Check MLflow tracking URI and runs.",
                "regressor_loaded": regressor_model is not None,
            }
        return {
            "status": "success",
            "message": (
                f"Loaded classifier '{model_name_info}'"
                + (
                    f" and regressor '{regressor_name_info}'."
                    if regressor_model is not None
                    else " (no regressor available)."
                )
            ),
            "champion": champion_info.to_dict(),
            "regressor": (
                regressor_champion_info.to_dict() if regressor_champion_info is not None else None
            ),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
