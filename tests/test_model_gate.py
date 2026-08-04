"""Phase 4: classification Model Gate tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import OVERFIT_F1_GAP_THRESHOLD
from src.model_gate import (
    compute_overfit_f1_gap,
    explain_selection,
    fetch_candidate_runs,
    select_champion_run,
)


def _fake_runs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "overfit_high_prauc",
                "tags.mlflow.runName": "run_bad",
                "tags.mlflow.parentRunId": None,
                "params.model_type": "xgboost",
                "metrics.pr_auc_test": 0.90,
                "metrics.f1_train": 0.95,
                "metrics.f1_test": 0.40,  # gap 0.55 > 0.30
                "metrics.roc_auc_test": 0.80,
                "metrics.recall_at_k_test": 0.50,
            },
            {
                "run_id": "stable_mid_prauc",
                "tags.mlflow.runName": "run_good",
                "tags.mlflow.parentRunId": None,
                "params.model_type": "logistic",
                "metrics.pr_auc_test": 0.70,
                "metrics.f1_train": 0.55,
                "metrics.f1_test": 0.50,  # gap 0.05
                "metrics.roc_auc_test": 0.66,
                "metrics.recall_at_k_test": 0.30,
            },
            {
                "run_id": "stable_low_prauc",
                "tags.mlflow.runName": "run_ok",
                "tags.mlflow.parentRunId": None,
                "params.model_type": "random_forest",
                "metrics.pr_auc_test": 0.60,
                "metrics.f1_train": 0.52,
                "metrics.f1_test": 0.51,
                "metrics.roc_auc_test": 0.61,
                "metrics.recall_at_k_test": 0.25,
            },
        ]
    )


def test_overfit_gap_computation():
    runs = compute_overfit_f1_gap(_fake_runs())
    by_id = runs.set_index("run_id")["overfit_f1_gap"]
    assert by_id["overfit_high_prauc"] == pytest.approx(0.55)
    assert by_id["stable_mid_prauc"] == pytest.approx(0.05)
    assert by_id["stable_low_prauc"] == pytest.approx(0.01)


def test_gate_rejects_overfit_and_picks_best_pr_auc(monkeypatch):
    # Bypass artifact existence checks for the synthetic table.
    monkeypatch.setattr(
        "src.model_gate._has_logged_model",
        lambda client, run_id: True,
    )
    selection = select_champion_run(
        runs=_fake_runs(),
        overfit_threshold=OVERFIT_F1_GAP_THRESHOLD,
    )
    assert selection.selection_mode == "champion"
    assert selection.passes_overfit_gate is True
    assert selection.run_id == "stable_mid_prauc"
    assert selection.model_type == "logistic"
    assert selection.n_rejected_overfit == 1
    assert selection.pr_auc_test == pytest.approx(0.70)


def test_gate_fallback_when_all_overfit():
    runs = _fake_runs()
    runs["metrics.f1_train"] = 0.99
    runs["metrics.f1_test"] = 0.10
    selection = select_champion_run(runs=runs, overfit_threshold=0.30)
    assert selection.selection_mode == "fallback"
    assert selection.passes_overfit_gate is False
    # All gaps equal (0.89); any selected run is acceptable, but gap must be min.
    assert selection.overfit_f1_gap == pytest.approx(0.89)


def test_live_mlflow_gate_selects_classification_champion():
    try:
        candidates = fetch_candidate_runs()
    except ValueError as exc:
        pytest.skip(str(exc))

    if candidates.empty:
        pytest.skip("No live classification runs available")

    selection = select_champion_run(runs=candidates)
    assert selection.run_id
    assert selection.model_uri.startswith("runs:/")
    assert selection.pr_auc_test >= 0.0
    assert "PR-AUC" in explain_selection(selection)
