from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from category_encoders import TargetEncoder
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

try:
    from src.config import (
        CALIBRATION_CV_FOLDS,
        CALIBRATION_MIN_MINORITY_FRACTION_FOR_ISOTONIC,
        CALIBRATION_MIN_SAMPLES_FOR_ISOTONIC,
        CLASSIFICATION_THRESHOLD,
        HORIZON_YEARS,
        MLFLOW_EXPERIMENT_NAME,
        OVERFIT_F1_GAP_THRESHOLD,
        PRIMARY_METRIC,
        RECALL_AT_K_FRACTION,
        TEMPORAL_SPLIT_DATE,
        dataset_checksums_path,
        processed_snapshots_path,
        raw_breaks_path,
    )
    from src.labeling import add_years_until_next_break, compute_file_sha256, load_break_events
    from src.preprocessing import (
        CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
        prepare_pipe_break_data,
    )
    from src.schema import FEATURE_COLUMNS, REGRESSION_TARGET_COLUMN, TARGET_COLUMN
except ModuleNotFoundError:
    from config import (
        CALIBRATION_CV_FOLDS,
        CALIBRATION_MIN_MINORITY_FRACTION_FOR_ISOTONIC,
        CALIBRATION_MIN_SAMPLES_FOR_ISOTONIC,
        CLASSIFICATION_THRESHOLD,
        HORIZON_YEARS,
        MLFLOW_EXPERIMENT_NAME,
        OVERFIT_F1_GAP_THRESHOLD,
        PRIMARY_METRIC,
        RECALL_AT_K_FRACTION,
        TEMPORAL_SPLIT_DATE,
        dataset_checksums_path,
        processed_snapshots_path,
        raw_breaks_path,
    )
    from labeling import add_years_until_next_break, compute_file_sha256, load_break_events
    from preprocessing import (
        CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
        prepare_pipe_break_data,
    )
    from schema import FEATURE_COLUMNS, REGRESSION_TARGET_COLUMN, TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model.pkl"
MODEL_OUTPUT_PATH_REGRESSOR = PROJECT_ROOT / "models" / "model_regressor.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports"
_DEFAULT_SQLITE = (PROJECT_ROOT / "mlflow.db").resolve()
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{_DEFAULT_SQLITE}")

CLASSIFICATION_MODEL_TYPES = [
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
]

# Regression model types (Phase 6): predict years_until_break for pipes the
# classifier flags as at-risk (label=1). Uncensored target only, see
# src.labeling.add_years_until_next_break and src.schema.REGRESSION_TARGET_COLUMN.
REGRESSION_MODEL_TYPES = ["linear", "ridge_reg", "lasso_reg", "rf_reg", "xgb_reg"]


def build_preprocessor(feature_columns):
    numeric_features = [c for c in feature_columns if c in NUMERIC_COLUMNS]
    categorical_features = [c for c in feature_columns if c in CATEGORICAL_COLUMNS]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("target_encoder", TargetEncoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def get_experiment_models(feature_columns, n_estimators=100, max_depth=10, learning_rate=0.1, alpha=1.0):
    preprocessor = build_preprocessor(feature_columns)
    c_value = float(1.0 / max(alpha, 1e-6))

    base_estimators = [
        (
            "xgb",
            XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                eval_metric="logloss",
                random_state=42,
                n_jobs=4,
            ),
        ),
        (
            "extra_trees",
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=4,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=4,
            ),
        ),
        ("logreg", LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs")),
    ]

    cv_fixed = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stacking_model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=2000),
        cv=cv_fixed,
        n_jobs=4,
        stack_method="predict_proba",
    )

    return {
        "run_01_logistic": Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=2000, solver="lbfgs"))]),
        "run_02_ridge": Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs"))]),
        "run_05_lasso": Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(penalty="l1", C=c_value, max_iter=3000, solver="saga"))]),
        "run_03_rf": Pipeline([("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=4))]),
        "run_04_xgboost": Pipeline([("preprocessor", preprocessor), ("model", XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, eval_metric="logloss", random_state=42, n_jobs=4))]),
        "run_07_extra_trees": Pipeline([("preprocessor", preprocessor), ("model", ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=4))]),
        "run_08_knn": Pipeline([("preprocessor", preprocessor), ("model", KNeighborsClassifier(n_neighbors=5))]),
        "run_09_svc": Pipeline([("preprocessor", preprocessor), ("model", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42))]),
        "run_10_mlp": Pipeline([("preprocessor", preprocessor), ("model", MLPClassifier(hidden_layer_sizes=(100, 50), activation="relu", max_iter=500, early_stopping=True, random_state=42))]),
        "run_06_stacking": Pipeline([("preprocessor", preprocessor), ("model", stacking_model)]),
    }


def get_regression_models(feature_columns, n_estimators=100, max_depth=10, learning_rate=0.1, alpha=1.0):
    """
    Regression models estimating years_until_break, sharing the same
    ColumnTransformer preprocessing (imputation, StandardScaler, TargetEncoder)
    as the classification pipelines.
    """
    preprocessor = build_preprocessor(feature_columns)

    return {
        "linear": Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())]),
        "ridge_reg": Pipeline([("preprocessor", preprocessor), ("model", Ridge(alpha=alpha, random_state=42))]),
        "lasso_reg": Pipeline([("preprocessor", preprocessor), ("model", Lasso(alpha=alpha, random_state=42, max_iter=5000))]),
        "rf_reg": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=4,
                    ),
                ),
            ]
        ),
        "xgb_reg": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        random_state=42,
                        n_jobs=4,
                    ),
                ),
            ]
        ),
    }


def predict_proba_positive(pipeline, X) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(pipeline, "decision_function"):
        scores = np.asarray(pipeline.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    return np.asarray(pipeline.predict(X), dtype=float)


def choose_calibration_method(
    y_train,
    min_samples_for_isotonic: int = CALIBRATION_MIN_SAMPLES_FOR_ISOTONIC,
    min_minority_fraction_for_isotonic: float = CALIBRATION_MIN_MINORITY_FRACTION_FOR_ISOTONIC,
) -> str:
    """Pick a CalibratedClassifierCV method without touching the test set.

    "isotonic" is non-parametric and can fit calibration curves more closely,
    but it needs enough samples per CV fold (especially positives) to avoid
    overfitting the calibration map itself. "sigmoid" (Platt scaling) is a
    safer default on small or imbalanced training sets.
    """
    y_train = np.asarray(y_train).astype(int)
    n_samples = int(len(y_train))
    if n_samples == 0:
        return "sigmoid"
    minority_fraction = float(min(np.mean(y_train == 1), np.mean(y_train == 0)))
    if n_samples >= min_samples_for_isotonic and minority_fraction >= min_minority_fraction_for_isotonic:
        return "isotonic"
    return "sigmoid"


def _log_calibration_curve(y_true, y_proba, model_type: str, n_bins: int = 10) -> None:
    """Log a reliability diagram (predicted vs. observed frequency) to MLflow.

    Best-effort only: calibration is still evaluated via the Brier score even
    if plotting is unavailable (e.g. matplotlib missing at runtime), so this
    never fails the training run.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy="quantile"
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Calibration parfaite")
        ax.plot(mean_predicted_value, fraction_of_positives, marker="o", label=model_type)
        ax.set_xlabel("Probabilité prédite (moyenne par bin)")
        ax.set_ylabel("Fréquence observée de la classe positive")
        ax.set_title(f"Courbe de calibration — {model_type} (test)")
        ax.legend(loc="best")
        fig.tight_layout()
        mlflow.log_figure(fig, "calibration_curve_test.png")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - purely cosmetic artifact
        print(f"⚠️  Calibration plot skipped ({exc}).", flush=True)


CONFUSION_MATRIX_CLASS_LABELS = ("Pas de bris ≤ 5 ans", "Bris ≤ 5 ans")


def compute_confusion_counts(y_true, y_pred) -> dict:
    """Confusion matrix counts for the fixed-horizon (5y) binary target.

    labels=[0, 1] pins the matrix layout regardless of which classes happen
    to be present in a given slice, so `.ravel()` always unpacks as
    (tn, fp, fn, tp) — 0 = no break within the horizon, 1 = break within it.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def save_confusion_matrix_csv(counts: dict, threshold: float, reports_dir: Path = REPORTS_DIR) -> Path:
    """Write reports/confusion_matrix_test.csv (temporal test set only)."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "confusion_matrix_test.csv"
    pd.DataFrame(
        [
            {
                "horizon_years": int(HORIZON_YEARS),
                "threshold": float(threshold),
                "true_negatives": counts["true_negatives"],
                "false_positives": counts["false_positives"],
                "false_negatives": counts["false_negatives"],
                "true_positives": counts["true_positives"],
            }
        ]
    ).to_csv(csv_path, index=False)
    return csv_path


def save_confusion_matrix_png(counts: dict, threshold: float, reports_dir: Path = REPORTS_DIR) -> Path:
    """Render reports/confusion_matrix_test.png with business-facing labels.

    Rows = true classes, columns = predicted classes (standard confusion
    matrix orientation), fixed to the project's 5-year horizon.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array(
        [
            [counts["true_negatives"], counts["false_positives"]],
            [counts["false_negatives"], counts["true_positives"]],
        ]
    )

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CONFUSION_MATRIX_CLASS_LABELS)
    ax.set_yticklabels(CONFUSION_MATRIX_CLASS_LABELS)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Vraie classe")
    ax.set_title(
        f"Matrice de confusion — Test temporel — Horizon {int(HORIZON_YEARS)} ans\n"
        f"(seuil={float(threshold):.2f})"
    )
    vmax = matrix.max() if matrix.max() > 0 else 1
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color="white" if matrix[i, j] > vmax / 2 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    png_path = reports_dir / "confusion_matrix_test.png"
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def log_confusion_matrix_artifacts(
    y_test,
    y_pred_test,
    threshold: float = CLASSIFICATION_THRESHOLD,
    reports_dir: Path = REPORTS_DIR,
) -> dict:
    """Build + log the fixed-horizon (5y) confusion matrix on the test set only.

    Computed exclusively on X_test/y_test predictions from the *calibrated*
    model thresholded at CLASSIFICATION_THRESHOLD — never on train data.
    Logs test_true_negatives / test_false_positives / test_false_negatives /
    test_true_positives as MLflow metrics, plus the CSV/PNG as artifacts.
    """
    counts = compute_confusion_counts(y_test, y_pred_test)
    csv_path = save_confusion_matrix_csv(counts, threshold, reports_dir=reports_dir)
    png_path = save_confusion_matrix_png(counts, threshold, reports_dir=reports_dir)

    mlflow.log_artifact(str(csv_path))
    mlflow.log_artifact(str(png_path))

    return {
        "test_true_negatives": float(counts["true_negatives"]),
        "test_false_positives": float(counts["false_positives"]),
        "test_false_negatives": float(counts["false_negatives"]),
        "test_true_positives": float(counts["true_positives"]),
    }


def recall_at_k(y_true, y_proba, k_fraction: float = RECALL_AT_K_FRACTION) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0 or len(y_true) == 0:
        return 0.0
    top_n = max(1, int(np.ceil(len(y_true) * float(k_fraction))))
    top_idx = np.argsort(y_proba)[::-1][:top_n]
    return float((y_true[top_idx] == 1).sum() / n_pos)


def calculate_classification_metrics(y_true, y_pred, y_proba, prefix: str = "") -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    return {
        f"{prefix}f1": float(f1_score(y_true, y_pred, zero_division=0)),
        f"{prefix}roc_auc": float(roc_auc_score(y_true, y_proba)),
        f"{prefix}pr_auc": float(average_precision_score(y_true, y_proba)),
        f"{prefix}recall_at_k": float(recall_at_k(y_true, y_proba)),
    }


def build_optuna_score(pr_auc_cv: float, f1_train: float, f1_cv: float) -> tuple[float, float]:
    overfit_gap = float(max(0.0, f1_train - f1_cv))
    score = pr_auc_cv / (1.0 + overfit_gap) if overfit_gap > OVERFIT_F1_GAP_THRESHOLD else pr_auc_cv
    return float(score), overfit_gap


def calculate_regression_metrics(y_true, y_pred, prefix: str = "") -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        f"{prefix}mse": mse,
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}r2": r2,
    }


def make_input_example(X: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    return X.head(n).copy()


def _print_summary(model_type, metrics):
    print(f"\n✅ Model {model_type} finished.", flush=True)
    print(f"🟣 PR-AUC Train: {metrics.get('pr_auc_train', 0):.4f}", flush=True)
    print(f"🟡 PR-AUC CV:    {metrics.get('pr_auc_cv', 0):.4f}", flush=True)
    print(f"🟢 PR-AUC Test:  {metrics.get('pr_auc_test', 0):.4f}", flush=True)
    print(
        f"🔵 F1 Train/Test: {metrics.get('f1_train', 0):.4f} / {metrics.get('f1_test', 0):.4f}",
        flush=True,
    )
    print(f"🟠 Overfit F1 gap: {metrics.get('overfit_f1_gap', 0):.4f}", flush=True)
    print(
        f"[Résultat] PR-AUC test={metrics.get('pr_auc_test', 0):.4f} | "
        f"F1 train/test={metrics.get('f1_train', 0):.4f}/{metrics.get('f1_test', 0):.4f}",
        flush=True,
    )


def log_and_save_model(pipeline, metrics, model_path, model_type, X_example, params=None):
    if params:
        safe_params = dict(params)
        safe_params.pop("model_type", None)
        safe_params.pop("leader_model_type", None)
        mlflow.log_params(safe_params)

    mlflow.log_param("model_type", model_type)
    try:
        mlflow.log_param("horizon_years", HORIZON_YEARS)
    except Exception:
        pass  # Ignore si le paramètre a déjà été enregistré pour ce run
    mlflow.log_metrics(metrics)

    input_example = make_input_example(X_example)
    pred_labels = np.asarray(pipeline.predict(input_example)).astype(int)
    signature = infer_signature(input_example, pred_labels)
    proba_example = predict_proba_positive(pipeline, input_example)

    mlflow.log_dict(
        {
            "input_columns": list(input_example.columns),
            "output_predict": "break_within_horizon class label {0,1}",
            "output_predict_proba_positive": "P(break_within_horizon=1)",
            "probability_example": [float(x) for x in np.asarray(proba_example).ravel().tolist()],
            "label_example": [int(x) for x in pred_labels.ravel().tolist()],
        },
        "model_io_schema.json",
    )

    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        serialization_format="cloudpickle",
        signature=signature,
        input_example=input_example,
    )

    active = mlflow.active_run()
    if active is not None:
        print(f"[MLflow] Run enregistré : {active.info.run_id}", flush=True)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    _print_summary(model_type, metrics)


def log_and_save_h2o(best_model, metrics, leader_model_type, params=None):
    import mlflow.h2o

    if params:
        safe_params = dict(params)
        safe_params.pop("model_type", None)
        safe_params.pop("leader_model_type", None)
        mlflow.log_params(safe_params)

    mlflow.log_param("model_type", "h2o")
    mlflow.log_param("leader_model_type", leader_model_type)
    mlflow.log_param("horizon_years", HORIZON_YEARS)
    mlflow.log_metrics(metrics)
    mlflow.set_tag("leader_model_id", best_model.model_id)
    mlflow.set_tag("leader_model_type", leader_model_type)
    mlflow.h2o.log_model(best_model, artifact_path="model")
    active = mlflow.active_run()
    if active is not None:
        print(f"[MLflow] Run enregistré : {active.info.run_id}", flush=True)
    _print_summary(f"H2O_{leader_model_type}", metrics)


def prepare_regression_data(horizon_years: int = HORIZON_YEARS, cutoff: Optional[str] = None):
    """
    Build the regression training frame for years_until_break.

    Reuses the classification snapshot/feature pipeline (prepare_pipe_break_data),
    then attaches years_until_break from raw break events and drops censored rows
    (no recorded future break for that asset) since the target is undefined for
    them. Rows kept for classification are unaffected by this filter.
    """
    _, _, _, _, cleaned_df = prepare_pipe_break_data(horizon_years=horizon_years)
    events = load_break_events()
    with_target = add_years_until_next_break(cleaned_df, events)
    with_target = with_target.dropna(subset=[REGRESSION_TARGET_COLUMN]).copy()

    if with_target.empty:
        raise ValueError(
            "No rows with a valid years_until_break target: every pipe in this "
            "dataset is right-censored (no recorded future break). Cannot train "
            "the regression model."
        )

    split_date = pd.Timestamp(cutoff or TEMPORAL_SPLIT_DATE)
    train_df = with_target[with_target["snapshot_date"] < split_date].copy()
    test_df = with_target[with_target["snapshot_date"] >= split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Temporal split at {split_date.date()} produced an empty regression "
            f"train or test set (train={len(train_df)}, test={len(test_df)})."
        )

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df[REGRESSION_TARGET_COLUMN].copy()
    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df[REGRESSION_TARGET_COLUMN].copy()
    return X_train, X_test, y_train, y_test, with_target


def _print_regression_summary(model_type, metrics):
    print(f"\n✅ Regressor {model_type} finished.")
    print(f"🟣 RMSE Train: {metrics.get('rmse_train', 0):.4f}")
    print(f"🟢 RMSE Test:  {metrics.get('rmse_test', 0):.4f}")
    print(f"🔵 MAE Train/Test: {metrics.get('mae_train', 0):.4f} / {metrics.get('mae_test', 0):.4f}")
    print(f"🟠 R2 Train/Test:  {metrics.get('r2_train', 0):.4f} / {metrics.get('r2_test', 0):.4f}")


def log_and_save_regressor(pipeline, metrics, model_path, model_type, X_example, params=None):
    if params:
        safe_params = dict(params)
        safe_params.pop("model_type", None)
        safe_params.pop("task", None)
        mlflow.log_params(safe_params)

    # Tag task=regression so the Model Gate (select_champion_regressor) can
    # separate regression runs from classification runs sharing the experiment.
    mlflow.log_params({"task": "regression", "model_type": model_type})
    try:
        mlflow.log_param("horizon_years", HORIZON_YEARS)
    except Exception:
        pass  # Already logged for this run.
    mlflow.log_metrics(metrics)

    input_example = make_input_example(X_example)
    pred_example = np.asarray(pipeline.predict(input_example), dtype=float)

    mlflow.log_dict(
        {
            "input_columns": list(input_example.columns),
            "output_predict": "years_until_break estimate (float, years)",
            "prediction_example": [float(x) for x in pred_example.ravel().tolist()],
        },
        "model_io_schema.json",
    )

    mlflow.sklearn.log_model(pipeline, "model")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    _print_regression_summary(model_type, metrics)


def train_evaluate_and_log_regressor(
    pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    model_path,
    model_type,
    params=None,
    is_optimized=False,
):
    pipeline.fit(X_train, y_train)

    train_pred = pipeline.predict(X_train)
    train_metrics = calculate_regression_metrics(y_train, train_pred, prefix="")

    test_pred = pipeline.predict(X_test)
    test_metrics = calculate_regression_metrics(y_test, test_pred, prefix="")

    metrics = {
        "mse_train": train_metrics["mse"],
        "rmse_train": train_metrics["rmse"],
        "mae_train": train_metrics["mae"],
        "r2_train": train_metrics["r2"],
        "mse_test": test_metrics["mse"],
        "rmse_test": test_metrics["rmse"],
        "mae_test": test_metrics["mae"],
        "r2_test": test_metrics["r2"],
    }

    if is_optimized and params is not None:
        params = {**params, "optimized": True}

    log_and_save_regressor(pipeline, metrics, model_path, model_type, X_example=X_train, params=params)
    return metrics


def train_evaluate_and_log(
    pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
    model_path,
    model_type,
    params=None,
    is_optimized=False,
):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"[Étape] Validation croisée stratifiée pour {model_type}...", flush=True)
    print(f"🔄 Stratified CV probabilities for {model_type}...", flush=True)
    cv_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv, n_jobs=-1, method="predict_proba")[:, 1]
    cv_pred = (cv_proba >= 0.5).astype(int)
    cv_metrics = calculate_classification_metrics(y_train, cv_pred, cv_proba)

    # --- Probability calibration -------------------------------------------------
    # CalibratedClassifierCV(cv=<int>) clones `pipeline` (discarding any prior
    # fit state) and, for each of the CALIBRATION_CV_FOLDS folds, fits it on
    # the fold's training rows and learns the calibration map on the held-out
    # rows — entirely inside X_train/y_train. X_test is never touched here, so
    # there is no leakage into the test metrics computed below.
    calibration_method = choose_calibration_method(y_train)
    print(
        f"[Étape] Calibrage des probabilités ({calibration_method}, "
        f"cv={CALIBRATION_CV_FOLDS}) pour {model_type}...",
        flush=True,
    )
    calibrated_pipeline = CalibratedClassifierCV(
        estimator=pipeline,
        method=calibration_method,
        cv=CALIBRATION_CV_FOLDS,
    )
    calibrated_pipeline.fit(X_train, y_train)

    # The calibrated wrapper is the model actually saved/served — its
    # predict_proba is what /predict exposes to the Streamlit app.
    train_proba = predict_proba_positive(calibrated_pipeline, X_train)
    train_pred = calibrated_pipeline.predict(X_train)
    train_metrics = calculate_classification_metrics(y_train, train_pred, train_proba)

    test_proba = predict_proba_positive(calibrated_pipeline, X_test)
    test_pred = calibrated_pipeline.predict(X_test)
    test_metrics = calculate_classification_metrics(y_test, test_pred, test_proba)

    brier_train = float(brier_score_loss(y_train, train_proba))
    brier_test = float(brier_score_loss(y_test, test_proba))
    _log_calibration_curve(y_test, test_proba, model_type)

    # --- Confusion matrix (fixed 5-year horizon, temporal test set only) --------
    # Decision rule pinned to CLASSIFICATION_THRESHOLD (not test_pred's implicit
    # 0.5 cutoff) and computed strictly from the calibrated test probabilities —
    # X_train/y_train are never involved here.
    y_pred_test_at_threshold = (test_proba >= CLASSIFICATION_THRESHOLD).astype(int)
    confusion_metrics = log_confusion_matrix_artifacts(
        y_test, y_pred_test_at_threshold, threshold=CLASSIFICATION_THRESHOLD
    )

    optuna_score, overfit_cv_gap = build_optuna_score(cv_metrics["pr_auc"], train_metrics["f1"], cv_metrics["f1"])
    overfit_f1_gap = float(max(0.0, train_metrics["f1"] - test_metrics["f1"]))

    metrics = {
        "pr_auc_cv": cv_metrics["pr_auc"],
        "f1_cv": cv_metrics["f1"],
        "roc_auc_cv": cv_metrics["roc_auc"],
        "recall_at_k_cv": cv_metrics["recall_at_k"],
        "pr_auc_train": train_metrics["pr_auc"],
        "f1_train": train_metrics["f1"],
        "roc_auc_train": train_metrics["roc_auc"],
        "pr_auc_test": test_metrics["pr_auc"],
        "f1_test": test_metrics["f1"],
        "roc_auc_test": test_metrics["roc_auc"],
        "recall_at_k_test": test_metrics["recall_at_k"],
        "overfit_f1_gap_cv": overfit_cv_gap,
        "overfit_f1_gap": overfit_f1_gap,
        "optuna_score": optuna_score,
        "overfit_threshold": float(OVERFIT_F1_GAP_THRESHOLD),
        "passes_overfit_gate": float(overfit_f1_gap <= OVERFIT_F1_GAP_THRESHOLD),
        "brier_train": brier_train,
        "brier_test": brier_test,
        **confusion_metrics,
    }

    if is_optimized and params is not None:
        params = {**params, "optimized": True}
    params = {**(params or {}), "calibration_method": calibration_method}

    log_and_save_model(calibrated_pipeline, metrics, model_path, model_type, X_example=X_train, params=params)
    return metrics


def log_mlflow_data(params, metrics):
    if params:
        safe_params = dict(params)
        safe_params.pop("model_type", None)
        safe_params.pop("leader_model_type", None)
        mlflow.log_params(safe_params)
    mlflow.log_metrics(metrics)


def _dataset_params(horizon_years: int = HORIZON_YEARS) -> dict:
    params = {
        "horizon_years": int(horizon_years),
        "temporal_split_date": TEMPORAL_SPLIT_DATE,
        "primary_metric": PRIMARY_METRIC,
        "overfit_f1_gap_threshold": OVERFIT_F1_GAP_THRESHOLD,
        "recall_at_k_fraction": RECALL_AT_K_FRACTION,
        "target_column": TARGET_COLUMN,
        "feature_columns": ",".join(FEATURE_COLUMNS),
    }
    if Path(raw_breaks_path).exists():
        params["dataset_raw_sha256"] = compute_file_sha256(raw_breaks_path)
    if Path(processed_snapshots_path).exists():
        params["dataset_snapshots_sha256"] = compute_file_sha256(processed_snapshots_path)
    if Path(dataset_checksums_path).exists():
        params["dataset_checksums_file"] = str(dataset_checksums_path)
    return params


def train_h2o_automl(df: pd.DataFrame, target: str, feature_columns, max_runtime_secs: int = 120):
    import h2o
    from h2o.automl import H2OAutoML

    h2o.init(nthreads=-1, strict_version_check=False)
    keep_cols = list(feature_columns) + [target]
    hf = h2o.H2OFrame(df[keep_cols])
    hf["years_since_last_break"] = hf["years_since_last_break"].asnumeric()
    hf[target] = hf[target].asfactor()

    train, test = hf.split_frame(ratios=[0.8], seed=42)
    x_columns = list(feature_columns)

    aml = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        seed=42,
        project_name="kw_water_main_break_risk",
        sort_metric="AUCPR",
        exclude_algos=["XGBoost"],
    )
    aml.train(x=x_columns, y=target, training_frame=train)
    return aml.leader, aml, train, test


def _h2o_perf_metrics(perf) -> dict:
    try:
        f1 = float(perf.F1()[0][1]) if perf.F1() else 0.0
    except Exception:
        f1 = 0.0
    return {"roc_auc": float(perf.auc()), "pr_auc": float(perf.aucpr()), "f1": f1}


def run_h2o_branch(max_runtime_secs: int = 120, horizon_years: int = HORIZON_YEARS):
    print("[Étape] Chargement des données...", flush=True)
    print("[Modèle] h2o", flush=True)
    X_train, X_test, y_train, y_test, cleaned_df = prepare_pipe_break_data(horizon_years=horizon_years)
    feature_columns = X_train.columns.tolist()
    print(
        f"[Étape] Split temporel terminé : train={len(X_train)}, test={len(X_test)}",
        flush=True,
    )

    full_df = pd.concat(
        [
            pd.concat([X_train, y_train.rename(TARGET_COLUMN)], axis=1),
            pd.concat([X_test, y_test.rename(TARGET_COLUMN)], axis=1),
        ],
        axis=0,
    )

    print(
        f"🚀 Launching H2O AutoML (classification, horizon={horizon_years}y)...",
        flush=True,
    )
    with mlflow.start_run(run_name="run_h2o_automl"):
        mlflow.log_params(
            {
                **_dataset_params(horizon_years=horizon_years),
                "model_type": "h2o",
                "h2o_max_runtime_secs": max_runtime_secs,
            }
        )
        best_model, aml_obj, train_frame, test_frame = train_h2o_automl(
            full_df, TARGET_COLUMN, feature_columns, max_runtime_secs=max_runtime_secs
        )

        perf_train = best_model.model_performance(train=True)
        perf_test = best_model.model_performance(test_data=test_frame)
        perf_cv = best_model.model_performance(xval=True)

        algo_name = best_model.algo
        print("\n📊 === H2O LEADERBOARD ===", flush=True)
        print(aml_obj.leaderboard.as_data_frame(), flush=True)

        metrics_train = _h2o_perf_metrics(perf_train)
        metrics_cv = _h2o_perf_metrics(perf_cv)
        metrics_test = _h2o_perf_metrics(perf_test)

        overfit_f1_gap = float(max(0.0, metrics_train["f1"] - metrics_test["f1"]))
        optuna_score, overfit_cv_gap = build_optuna_score(metrics_cv["pr_auc"], metrics_train["f1"], metrics_cv["f1"])

        metrics = {
            "pr_auc_cv": metrics_cv["pr_auc"],
            "f1_cv": metrics_cv["f1"],
            "roc_auc_cv": metrics_cv["roc_auc"],
            "pr_auc_train": metrics_train["pr_auc"],
            "f1_train": metrics_train["f1"],
            "roc_auc_train": metrics_train["roc_auc"],
            "pr_auc_test": metrics_test["pr_auc"],
            "f1_test": metrics_test["f1"],
            "roc_auc_test": metrics_test["roc_auc"],
            "overfit_f1_gap": overfit_f1_gap,
            "overfit_f1_gap_cv": overfit_cv_gap,
            "optuna_score": optuna_score,
            "overfit_threshold": float(OVERFIT_F1_GAP_THRESHOLD),
            "passes_overfit_gate": float(overfit_f1_gap <= OVERFIT_F1_GAP_THRESHOLD),
        }

        log_and_save_h2o(
            best_model,
            metrics,
            f"H2O_{algo_name.upper()}",
            params={
                "model_type": "h2o",
                "leader_model_type": f"H2O_{algo_name.upper()}",
                "h2o_max_runtime_secs": max_runtime_secs,
                "horizon_years": horizon_years,
                **_dataset_params(horizon_years=horizon_years),
            },
        )

    _ = cleaned_df


def run_regression(args) -> None:
    """
    Train a years_until_break regressor (Phase 6).

    Mirrors run_classification's structure but targets the uncensored
    years_until_break rows produced by prepare_regression_data(), reusing the
    same feature preprocessing (build_preprocessor / get_regression_models).
    """
    X_train, X_test, y_train, y_test, cleaned_df = prepare_regression_data(horizon_years=args.horizon_years)
    for frame in (X_train, X_test):
        for col in frame.select_dtypes(include=["integer"]).columns:
            frame[col] = frame[col].astype(float)

    print(
        f"[regression] Horizon={args.horizon_years}y | Temporal split @ {TEMPORAL_SPLIT_DATE}: "
        f"train={len(X_train)} test={len(X_test)} (uncensored rows only)"
    )

    shared_params = _dataset_params(horizon_years=args.horizon_years)
    shared_params["target_column"] = REGRESSION_TARGET_COLUMN

    if args.tune:
        print(f"🎯 Optuna tuning for {args.model_type} ({args.n_trials} trials, minimize RMSE)...")
        with mlflow.start_run(run_name=f"Optuna_Study_reg_{args.model_type}"):
            mlflow.log_params({**shared_params, "task": "regression", "model_type": args.model_type, "tune": True})

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", args.n_est_min, args.n_est_max, step=50),
                    "max_depth": trial.suggest_int("max_depth", args.depth_min, args.depth_max),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "alpha": trial.suggest_float("alpha", 0.1, 10.0, log=True),
                }
                trial_pipeline = get_regression_models(X_train.columns.tolist(), **params).get(args.model_type)

                with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                    cv = KFold(n_splits=5, shuffle=True, random_state=42)
                    cv_pred = cross_val_predict(trial_pipeline, X_train, y_train, cv=cv, n_jobs=-1)
                    cv_metrics = calculate_regression_metrics(y_train, cv_pred)

                    trial_pipeline.fit(X_train, y_train)
                    train_pred = trial_pipeline.predict(X_train)
                    train_metrics = calculate_regression_metrics(y_train, train_pred)

                    test_pred = trial_pipeline.predict(X_test)
                    test_metrics = calculate_regression_metrics(y_test, test_pred)

                    trial_metrics = {
                        "mse_cv": cv_metrics["mse"],
                        "rmse_cv": cv_metrics["rmse"],
                        "mae_cv": cv_metrics["mae"],
                        "r2_cv": cv_metrics["r2"],
                        "mse_train": train_metrics["mse"],
                        "rmse_train": train_metrics["rmse"],
                        "mae_train": train_metrics["mae"],
                        "r2_train": train_metrics["r2"],
                        "mse_test": test_metrics["mse"],
                        "rmse_test": test_metrics["rmse"],
                        "mae_test": test_metrics["mae"],
                        "r2_test": test_metrics["r2"],
                    }
                    log_mlflow_data(params={**params, "task": "regression", "model_type": args.model_type}, metrics=trial_metrics)
                    return trial_metrics["rmse_cv"]

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=args.n_trials)
            print(f"\n🏆 Best params (min RMSE): {study.best_params}")

            final_params = {**study.best_params, "model_type": args.model_type, **shared_params}
            champion = get_regression_models(
                X_train.columns.tolist(),
                n_estimators=study.best_params.get("n_estimators", args.n_estimators),
                max_depth=study.best_params.get("max_depth", args.max_depth),
                learning_rate=study.best_params.get("learning_rate", args.learning_rate),
                alpha=study.best_params.get("alpha", args.alpha),
            ).get(args.model_type)

            train_evaluate_and_log_regressor(
                champion,
                X_train,
                y_train,
                X_test,
                y_test,
                MODEL_OUTPUT_PATH_REGRESSOR,
                args.model_type,
                params=final_params,
                is_optimized=True,
            )
        return

    models = get_regression_models(
        X_train.columns.tolist(),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
    )
    pipeline = models[args.model_type]
    with mlflow.start_run(run_name=f"run_reg_{args.model_type}"):
        train_evaluate_and_log_regressor(
            pipeline,
            X_train,
            y_train,
            X_test,
            y_test,
            MODEL_OUTPUT_PATH_REGRESSOR,
            args.model_type,
            params={**vars(args), **shared_params},
        )
    _ = cleaned_df


def run_classification(args) -> None:
    if args.model_type == "h2o":
        run_h2o_branch(max_runtime_secs=args.h2o_max_runtime_secs, horizon_years=args.horizon_years)
        return

    print("[Étape] Chargement des données...", flush=True)
    print(f"[Modèle] {args.model_type}", flush=True)
    X_train, X_test, y_train, y_test, cleaned_df = prepare_pipe_break_data(horizon_years=args.horizon_years)
    for frame in (X_train, X_test):
        for col in frame.select_dtypes(include=["integer"]).columns:
            frame[col] = frame[col].astype(float)

    print(
        f"[Étape] Split temporel terminé : train={len(X_train)}, test={len(X_test)}",
        flush=True,
    )
    print(
        f"Horizon={args.horizon_years}y | Temporal split @ {TEMPORAL_SPLIT_DATE}: "
        f"train={len(X_train)} test={len(X_test)} "
        f"(pos_train={int(y_train.sum())}, pos_test={int(y_test.sum())})",
        flush=True,
    )

    run_name_mapping = {
        "logistic": "run_01_logistic",
        "ridge": "run_02_ridge",
        "lasso": "run_05_lasso",
        "random_forest": "run_03_rf",
        "xgboost": "run_04_xgboost",
        "extra_trees": "run_07_extra_trees",
        "knn": "run_08_knn",
        "svc": "run_09_svc",
        "mlp": "run_10_mlp",
        "stacking": "run_06_stacking",
    }
    shared_params = _dataset_params(horizon_years=args.horizon_years)

    if args.tune:
        print(
            f"🎯 Optuna tuning for {args.model_type} "
            f"({args.n_trials} trials, maximize PR-AUC with F1 overfit penalty)...",
            flush=True,
        )
        print(f"[Optuna] Démarrage de l'étude ({args.n_trials} essais)...", flush=True)
        with mlflow.start_run(run_name=f"Optuna_Study_{args.model_type}"):
            mlflow.log_params({**shared_params, "model_type": args.model_type, "tune": True})

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", args.n_est_min, args.n_est_max, step=50),
                    "max_depth": trial.suggest_int("max_depth", args.depth_min, args.depth_max),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "alpha": trial.suggest_float("alpha", 0.1, 10.0, log=True),
                }
                trial_pipeline = get_experiment_models(X_train.columns.tolist(), **params).get(run_name_mapping[args.model_type])

                with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cv_proba = cross_val_predict(trial_pipeline, X_train, y_train, cv=cv, n_jobs=-1, method="predict_proba")[:, 1]
                    cv_pred = (cv_proba >= 0.5).astype(int)
                    cv_metrics = calculate_classification_metrics(y_train, cv_pred, cv_proba)

                    trial_pipeline.fit(X_train, y_train)
                    train_proba = predict_proba_positive(trial_pipeline, X_train)
                    train_pred = trial_pipeline.predict(X_train)
                    train_metrics = calculate_classification_metrics(y_train, train_pred, train_proba)

                    test_proba = predict_proba_positive(trial_pipeline, X_test)
                    test_pred = trial_pipeline.predict(X_test)
                    test_metrics = calculate_classification_metrics(y_test, test_pred, test_proba)

                    optuna_score, overfit_cv_gap = build_optuna_score(cv_metrics["pr_auc"], train_metrics["f1"], cv_metrics["f1"])
                    trial_metrics = {
                        "pr_auc_cv": cv_metrics["pr_auc"],
                        "f1_cv": cv_metrics["f1"],
                        "roc_auc_cv": cv_metrics["roc_auc"],
                        "recall_at_k_cv": cv_metrics["recall_at_k"],
                        "pr_auc_train": train_metrics["pr_auc"],
                        "f1_train": train_metrics["f1"],
                        "pr_auc_test": test_metrics["pr_auc"],
                        "f1_test": test_metrics["f1"],
                        "roc_auc_test": test_metrics["roc_auc"],
                        "recall_at_k_test": test_metrics["recall_at_k"],
                        "overfit_f1_gap_cv": overfit_cv_gap,
                        "overfit_f1_gap": float(max(0.0, train_metrics["f1"] - test_metrics["f1"])),
                        "optuna_score": optuna_score,
                    }
                    log_mlflow_data(params={**params, "model_type": args.model_type}, metrics=trial_metrics)
                    return optuna_score

            def _optuna_progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
                best = study.best_value if study.best_trial is not None else float("nan")
                print(
                    f"[Optuna] Essai {trial.number + 1}/{args.n_trials} | meilleur score={best:.4f}",
                    flush=True,
                )

            study = optuna.create_study(direction="maximize")
            study.optimize(
                objective,
                n_trials=args.n_trials,
                callbacks=[_optuna_progress_callback],
            )
            print(f"\n🏆 Best params: {study.best_params}", flush=True)
            print(f"[Optuna] Meilleur score final={study.best_value:.4f}", flush=True)

            final_params = {**study.best_params, "model_type": args.model_type, **shared_params}
            champion = get_experiment_models(
                X_train.columns.tolist(),
                n_estimators=study.best_params.get("n_estimators", args.n_estimators),
                max_depth=study.best_params.get("max_depth", args.max_depth),
                learning_rate=study.best_params.get("learning_rate", args.learning_rate),
                alpha=study.best_params.get("alpha", args.alpha),
            ).get(run_name_mapping[args.model_type])

            print("[Étape] Entraînement final avec les meilleurs hyperparamètres...", flush=True)
            train_evaluate_and_log(
                champion,
                X_train,
                y_train,
                X_test,
                y_test,
                MODEL_OUTPUT_PATH,
                args.model_type,
                params=final_params,
                is_optimized=True,
            )
        return

    models = get_experiment_models(
        X_train.columns.tolist(),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
    )
    pipeline = models[run_name_mapping[args.model_type]]
    print("[Étape] Entraînement du pipeline...", flush=True)
    with mlflow.start_run(run_name=run_name_mapping[args.model_type]):
        train_evaluate_and_log(
            pipeline,
            X_train,
            y_train,
            X_test,
            y_test,
            MODEL_OUTPUT_PATH,
            args.model_type,
            params={**vars(args), **shared_params},
        )


def main():
    parser = argparse.ArgumentParser(description="Train KW water-main break classifiers/regressors with MLflow tracking.")
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help=(
            "classification (default, unchanged): break_within_horizon. "
            "regression: years_until_break, restricted to uncensored rows."
        ),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        choices=CLASSIFICATION_MODEL_TYPES + REGRESSION_MODEL_TYPES,
        help="Defaults to 'xgboost' for classification, 'xgb_reg' for regression.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--n_est_min", type=int, default=100)
    parser.add_argument("--n_est_max", type=int, default=1000)
    parser.add_argument("--depth_min", type=int, default=3)
    parser.add_argument("--depth_max", type=int, default=15)
    parser.add_argument("--h2o_max_runtime_secs", type=int, default=120)
    parser.add_argument("--horizon_years", type=int, default=HORIZON_YEARS, choices=[1, 2, 5])
    args = parser.parse_args()

    if args.model_type is None:
        args.model_type = "xgb_reg" if args.task == "regression" else "xgboost"

    if args.task == "regression" and args.model_type not in REGRESSION_MODEL_TYPES:
        parser.error(f"--model_type {args.model_type!r} is not valid for --task regression. Choose from {REGRESSION_MODEL_TYPES}.")
    if args.task == "classification" and args.model_type not in CLASSIFICATION_MODEL_TYPES:
        parser.error(f"--model_type {args.model_type!r} is not valid for --task classification. Choose from {CLASSIFICATION_MODEL_TYPES}.")

    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    if args.task == "regression":
        run_regression(args)
    else:
        run_classification(args)


if __name__ == "__main__":
    main()