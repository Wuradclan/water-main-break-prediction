"""
Industrial Model Gate for KW water-main break classification.

Selection rules (Phase 4)
-------------------------
1. Consider only top-level MLflow runs (exclude Optuna nested trial_* runs).
2. Require logged classification metrics: pr_auc_test, f1_train, f1_test.
3. Reject models whose F1 overfit gap exceeds OVERFIT_F1_GAP_THRESHOLD:
       overfit_f1_gap = max(0, f1_train - f1_test)
4. Among surviving models, choose the champion with the highest pr_auc_test.
5. Fallback (all rejected): pick the run with the smallest overfit_f1_gap.

This module is intentionally API-agnostic. Phase 5 will wire it into FastAPI.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd

try:
    from src.config import (
        MLFLOW_EXPERIMENT_NAME,
        MLFLOW_TRACKING_URI_DEFAULT,
        OVERFIT_F1_GAP_THRESHOLD,
        PRIMARY_METRIC,
    )
except ModuleNotFoundError:
    from config import (
        MLFLOW_EXPERIMENT_NAME,
        MLFLOW_TRACKING_URI_DEFAULT,
        OVERFIT_F1_GAP_THRESHOLD,
        PRIMARY_METRIC,
    )


REQUIRED_METRICS = (
    "metrics.pr_auc_test",
    "metrics.f1_train",
    "metrics.f1_test",
)

REQUIRED_REGRESSION_METRICS = (
    "metrics.rmse_test",
    "metrics.mae_test",
    "metrics.r2_test",
)


@dataclass
class ChampionSelection:
    run_id: str
    run_name: str
    model_type: str
    model_uri: str
    horizon_years: int
    pr_auc_test: float
    f1_train: float
    f1_test: float
    overfit_f1_gap: float
    roc_auc_test: Optional[float]
    recall_at_k_test: Optional[float]
    passes_overfit_gate: bool
    selection_mode: str  # "champion" | "fallback"
    overfit_threshold: float
    n_candidates: int
    n_rejected_overfit: int

    def summary(self) -> str:
        status = "PASS" if self.passes_overfit_gate else "FALLBACK"
        return (
            f"{self.model_type} [{status}] "
            f"PR-AUC_test={self.pr_auc_test:.4f} | "
            f"F1_train/test={self.f1_train:.4f}/{self.f1_test:.4f} | "
            f"overfit_f1_gap={self.overfit_f1_gap:.4f}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RegressorChampionSelection:
    """Champion metadata for the years_until_break regression Model Gate."""

    run_id: str
    run_name: str
    model_type: str
    model_uri: str
    rmse_test: float
    mae_test: float
    r2_test: float

    def summary(self) -> str:
        return (
            f"{self.model_type} [regression] "
            f"RMSE_test={self.rmse_test:.4f} | MAE_test={self.mae_test:.4f} | "
            f"R2_test={self.r2_test:.4f}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_tracking_uri(tracking_uri: Optional[str] = None) -> str:
    """
    Resolve MLflow tracking URI for Docker and local execution.

    Priority:
      1. Explicit argument
      2. MLFLOW_TRACKING_URI environment variable
      3. http://mlflow:5000 when running inside a Docker container
      4. Local SQLite default (MLFLOW_TRACKING_URI_DEFAULT)
    """
    if tracking_uri:
        return tracking_uri
    env_uri = os.getenv("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri
    if Path("/.dockerenv").exists():
        return "http://mlflow:5000"
    return MLFLOW_TRACKING_URI_DEFAULT


def _has_logged_model(client: mlflow.tracking.MlflowClient, run_id: str) -> bool:
    """True if the run logged a model artifact under the conventional 'model' path."""
    try:
        artifacts = client.list_artifacts(run_id, path="model")
        return len(artifacts) > 0
    except Exception:
        # Older / partial runs may not expose the directory listing cleanly.
        try:
            root = client.list_artifacts(run_id)
            return any(a.path == "model" or a.path.startswith("model/") for a in root)
        except Exception:
            return False


def fetch_candidate_runs(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    tracking_uri: Optional[str] = None,
    max_results: int = 100,
) -> pd.DataFrame:
    """Load top-level classification runs that expose the gate metrics."""
    mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri))
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"MLflow experiment not found: {experiment_name!r}. "
            "Train at least one classification model first."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "metrics.pr_auc_test >= 0 AND "
            "metrics.f1_train >= 0 AND "
            "metrics.f1_test >= 0"
        ),
        max_results=max_results,
        order_by=["metrics.pr_auc_test DESC"],
    )
    if runs.empty:
        return runs

    # Exclude Optuna nested trials / parent-less incomplete children.
    if "tags.mlflow.parentRunId" in runs.columns:
        runs = runs[runs["tags.mlflow.parentRunId"].isnull()].copy()
    if "tags.mlflow.runName" in runs.columns:
        runs = runs[~runs["tags.mlflow.runName"].astype(str).str.startswith("trial_")].copy()
        # Note: Optuna_Study_* parent runs are NOT excluded here — train.py logs the
        # tuned champion's model artifact directly onto that parent run, so it must
        # remain eligible. `_has_logged_model` below is what actually screens out
        # study runs that never got a champion refit.

    missing = [c for c in REQUIRED_METRICS if c not in runs.columns]
    if missing:
        raise ValueError(f"Candidate runs missing required metrics: {missing}")

    runs = runs.dropna(subset=list(REQUIRED_METRICS)).copy()

    client = mlflow.tracking.MlflowClient()
    has_model = runs["run_id"].apply(lambda rid: _has_logged_model(client, rid))
    runs = runs[has_model].copy()
    return runs.reset_index(drop=True)


def fetch_candidate_regression_runs(
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
    tracking_uri: Optional[str] = None,
    max_results: int = 100,
) -> pd.DataFrame:
    """Load top-level regression runs (params.task == 'regression') that expose the gate metrics."""
    mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri))
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"MLflow experiment not found: {experiment_name!r}. "
            "Train at least one regression model first (--task regression)."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "params.task = 'regression' AND "
            "metrics.rmse_test >= 0 AND "
            "metrics.mae_test >= 0"
        ),
        max_results=max_results,
        order_by=["metrics.rmse_test ASC"],
    )
    if runs.empty:
        return runs

    # Exclude Optuna nested trials, same rationale as fetch_candidate_runs.
    if "tags.mlflow.parentRunId" in runs.columns:
        runs = runs[runs["tags.mlflow.parentRunId"].isnull()].copy()
    if "tags.mlflow.runName" in runs.columns:
        runs = runs[~runs["tags.mlflow.runName"].astype(str).str.startswith("trial_")].copy()

    missing = [c for c in REQUIRED_REGRESSION_METRICS if c not in runs.columns]
    if missing:
        raise ValueError(f"Candidate regression runs missing required metrics: {missing}")

    runs = runs.dropna(subset=list(REQUIRED_REGRESSION_METRICS)).copy()

    client = mlflow.tracking.MlflowClient()
    has_model = runs["run_id"].apply(lambda rid: _has_logged_model(client, rid))
    runs = runs[has_model].copy()
    return runs.reset_index(drop=True)


def select_champion_regressor(
    runs: Optional[pd.DataFrame] = None,
    tracking_uri: Optional[str] = None,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> RegressorChampionSelection:
    """
    Regression Model Gate: same fetch/filter logic as select_champion_run, but
    restricted to params.task == 'regression' runs and ranked by the lowest
    test RMSE instead of the highest PR-AUC.
    """
    if runs is None:
        runs = fetch_candidate_regression_runs(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
        )
    if runs.empty:
        raise ValueError(
            "No eligible regression runs found for the Model Gate "
            f"(experiment={experiment_name!r}). Train with --task regression first."
        )

    best_idx = runs["metrics.rmse_test"].idxmin()
    best = runs.loc[best_idx]

    run_name = str(best.get("tags.mlflow.runName", "unnamed"))
    model_type = str(best.get("params.model_type", run_name))
    run_id = str(best["run_id"])

    return RegressorChampionSelection(
        run_id=run_id,
        run_name=run_name,
        model_type=model_type,
        model_uri=f"runs:/{run_id}/model",
        rmse_test=float(best["metrics.rmse_test"]),
        mae_test=float(best["metrics.mae_test"]),
        r2_test=float(best["metrics.r2_test"]),
    )


def explain_regression_selection(selection: RegressorChampionSelection) -> str:
    return "\n".join(
        [
            "=== KW Water-Main Regression Model Gate ===",
            "Primary metric     : rmse_test (lower is better)",
            f"Champion run       : {selection.run_name} ({selection.run_id})",
            f"Model type         : {selection.model_type}",
            f"Model URI          : {selection.model_uri}",
            f"RMSE test          : {selection.rmse_test:.6f}",
            f"MAE test           : {selection.mae_test:.6f}",
            f"R2 test            : {selection.r2_test:.6f}",
            f"Summary            : {selection.summary()}",
        ]
    )


def compute_overfit_f1_gap(runs: pd.DataFrame) -> pd.DataFrame:
    """Attach overfit_f1_gap = max(0, f1_train - f1_test)."""
    out = runs.copy()
    if "metrics.overfit_f1_gap" in out.columns and out["metrics.overfit_f1_gap"].notna().any():
        out["overfit_f1_gap"] = out["metrics.overfit_f1_gap"].fillna(0.0).clip(lower=0.0)
    else:
        out["overfit_f1_gap"] = (
            out["metrics.f1_train"] - out["metrics.f1_test"]
        ).clip(lower=0.0)
    return out


def select_champion_run(
    runs: Optional[pd.DataFrame] = None,
    overfit_threshold: float = OVERFIT_F1_GAP_THRESHOLD,
    tracking_uri: Optional[str] = None,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
) -> ChampionSelection:
    """
    Apply the classification Model Gate and return the champion metadata.

    Does not load the model artifact (API wiring is Phase 5).
    """
    if runs is None:
        runs = fetch_candidate_runs(
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
        )
    if runs.empty:
        raise ValueError(
            "No eligible classification runs found for the Model Gate "
            f"(experiment={experiment_name!r})."
        )

    runs = compute_overfit_f1_gap(runs)
    valid = runs[runs["overfit_f1_gap"] <= overfit_threshold].copy()
    n_rejected = int(len(runs) - len(valid))

    if not valid.empty:
        # Primary champion metric: highest PR-AUC on the temporal test set.
        best_idx = valid["metrics.pr_auc_test"].idxmax()
        best = valid.loc[best_idx]
        mode = "champion"
        passes = True
    else:
        # Safety fallback: least-overfit model if every candidate fails the gate.
        best_idx = runs["overfit_f1_gap"].idxmin()
        best = runs.loc[best_idx]
        mode = "fallback"
        passes = False

    def _optional_metric(col: str) -> Optional[float]:
        if col in best.index and pd.notna(best[col]):
            return float(best[col])
        return None

    run_name = str(best.get("tags.mlflow.runName", "unnamed"))
    model_type = str(best.get("params.model_type", run_name))
    run_id = str(best["run_id"])
    # --- EXTRACTION DE L'HORIZON ---
    raw_horizon = best.get("params.horizon_years", 5)
    try:
        horizon_years = int(float(raw_horizon)) if pd.notna(raw_horizon) else 5
    except (ValueError, TypeError):
        horizon_years = 5
    # -------------------------------

    return ChampionSelection(
        run_id=run_id,
        run_name=run_name,
        model_type=model_type,
        model_uri=f"runs:/{run_id}/model",
        horizon_years=horizon_years,
        pr_auc_test=float(best["metrics.pr_auc_test"]),
        f1_train=float(best["metrics.f1_train"]),
        f1_test=float(best["metrics.f1_test"]),
        overfit_f1_gap=float(best["overfit_f1_gap"]),
        roc_auc_test=_optional_metric("metrics.roc_auc_test"),
        recall_at_k_test=_optional_metric("metrics.recall_at_k_test"),
        passes_overfit_gate=passes,
        selection_mode=mode,
        overfit_threshold=float(overfit_threshold),
        n_candidates=int(len(runs)),
        n_rejected_overfit=n_rejected,
    )


def explain_selection(selection: ChampionSelection) -> str:
    lines = [
        "=== KW Water-Main Model Gate ===",
        f"Primary metric     : {PRIMARY_METRIC} (higher is better)",
        f"Overfit rule       : reject if (f1_train - f1_test) > {selection.overfit_threshold:.2f}",
        f"Candidates         : {selection.n_candidates}",
        f"Rejected (overfit) : {selection.n_rejected_overfit}",
        f"Selection mode     : {selection.selection_mode}",
        f"Champion run       : {selection.run_name} ({selection.run_id})",
        f"Model type         : {selection.model_type}",
        f"Model URI          : {selection.model_uri}",
        f"PR-AUC test        : {selection.pr_auc_test:.6f}",
        f"F1 train / test    : {selection.f1_train:.6f} / {selection.f1_test:.6f}",
        f"Overfit F1 gap     : {selection.overfit_f1_gap:.6f}",
    ]
    if selection.roc_auc_test is not None:
        lines.append(f"ROC-AUC test       : {selection.roc_auc_test:.6f}")
    if selection.recall_at_k_test is not None:
        lines.append(f"recall@k test      : {selection.recall_at_k_test:.6f}")
    lines.append(f"Summary            : {selection.summary()}")
    return "\n".join(lines)


if __name__ == "__main__":
    champion = select_champion_run()
    print(explain_selection(champion))
