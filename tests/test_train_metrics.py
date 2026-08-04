"""Phase 3 unit checks for classification metrics helpers."""

from __future__ import annotations

import numpy as np

from src.train import build_optuna_score, calculate_classification_metrics, recall_at_k


def test_recall_at_k_perfect_ranking():
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    # top 20% = 2 samples, both positives => recall 1.0
    assert recall_at_k(y_true, y_proba, k_fraction=0.2) == 1.0


def test_classification_metrics_keys():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4])
    metrics = calculate_classification_metrics(y_true, y_pred, y_proba)
    assert {"f1", "roc_auc", "pr_auc", "recall_at_k"} <= set(metrics)


def test_optuna_score_penalizes_f1_overfit():
    score_ok, gap_ok = build_optuna_score(pr_auc_cv=0.8, f1_train=0.7, f1_cv=0.65)
    score_bad, gap_bad = build_optuna_score(pr_auc_cv=0.8, f1_train=0.95, f1_cv=0.50)
    assert gap_ok <= 0.30
    assert gap_bad > 0.30
    assert score_ok == 0.8
    assert score_bad < score_ok
