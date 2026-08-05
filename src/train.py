"""
Binary classification training for KW water-main break risk.

Task: P(break within H years | features at snapshot date t)
Primary metric: PR-AUC (champion ranking)
Overfit monitor: F1 gap (train - cv / train - test), threshold from config
Temporal split: preserved via prepare_pipe_break_data (no random split)

Three training modes (mirrors the aircraft-speed-prediction pipeline):
  A. Classic manual training  -> python -m src.train --model_type xgboost
  B. Optuna hyperparameter search -> python -m src.train --model_type xgboost --tune --n_trials 30
  C. H2O AutoML (classification) -> python -m src.train --model_type h2o
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# macOS Homebrew OpenMP may live outside the default @rpath searched by XGBoost.
_OMP_CANDIDATES = (
    Path("/opt/homebrew/opt/libomp/lib"),
    Path("/usr/local/opt/libomp/lib"),
    Path("/Volumes/Evo/system/homebrew/opt/libomp/lib"),
)
for _omp in _OMP_CANDIDATES:
    if (_omp / "libomp.dylib").exists():
        os.environ["DYLD_LIBRARY_PATH"] = f"{_omp}{os.pathsep}{os.environ.get('DYLD_LIBRARY_PATH', '')}"
        break

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from category_encoders import TargetEncoder
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

try:
    from src.config import (
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
    from src.labeling import compute_file_sha256
    from src.preprocessing import (
        CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
        prepare_pipe_break_data,
    )
    from src.schema import FEATURE_COLUMNS, TARGET_COLUMN
except ModuleNotFoundError:
    from config import (
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
    from labeling import compute_file_sha256
    from preprocessing import (
        CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
        prepare_pipe_break_data,
    )
    from schema import FEATURE_COLUMNS, TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model.pkl"
# Prefer an explicit env URI (Docker: http://mlflow:5000). Local default uses SQLite
# because recent MLflow versions reject the legacy filesystem FileStore backend.
_DEFAULT_SQLITE = (PROJECT_ROOT / "mlflow.db").resolve()
DEFAULT_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{_DEFAULT_SQLITE}",
)


# ---------------------------------------------------------------------------
# Preprocessing + model factory (Scikit-Learn)
# ---------------------------------------------------------------------------

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


def get_experiment_models(
    feature_columns,
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    alpha=1.0,
):
    """Return classification pipelines keyed by MLflow run name."""
    preprocessor = build_preprocessor(feature_columns)
    # Map ridge/lasso strength: larger alpha => smaller C.
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
        (
            "logreg",
            LogisticRegression(C=c_value, max_iter=2000, solver="lbfgs"),
        ),
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
        "run_01_logistic": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=2000, solver="lbfgs")),
            ]
        ),
        "run_02_ridge": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2", C=c_value, max_iter=2000, solver="lbfgs"
                    ),
                ),
            ]
        ),
        "run_05_lasso": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        penalty="l1", C=c_value, max_iter=3000, solver="saga"
                    ),
                ),
            ]
        ),
        "run_03_rf": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=4,
                    ),
                ),
            ]
        ),
        "run_04_xgboost": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=4,
                    ),
                ),
            ]
        ),
        "run_07_extra_trees": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42,
                        n_jobs=4,
                    ),
                ),
            ]
        ),
        "run_08_knn": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "run_09_svc": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        C=10.0,
                        gamma="scale",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "run_10_mlp": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(100, 50),
                        activation="relu",
                        max_iter=500,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "run_06_stacking": Pipeline(
            steps=[("preprocessor", preprocessor), ("model", stacking_model)]
        ),
    }


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def predict_proba_positive(pipeline, X) -> np.ndarray:
    """Return P(y=1) for a fitted classifier pipeline."""
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(pipeline, "decision_function"):
        scores = np.asarray(pipeline.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-scores))
    preds = np.asarray(pipeline.predict(X), dtype=float)
    return preds


def recall_at_k(y_true, y_proba, k_fraction: float = RECALL_AT_K_FRACTION) -> float:
    """Recall among the top-k fraction highest-risk predictions."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0 or len(y_true) == 0:
        return 0.0
    top_n = max(1, int(np.ceil(len(y_true) * float(k_fraction))))
    top_idx = np.argsort(y_proba)[::-1][:top_n]
    return float((y_true[top_idx] == 1).sum() / n_pos)


def calculate_classification_metrics(y_true, y_pred, y_proba, prefix: str = "") -> dict:
    """Compute F1 / ROC-AUC / PR-AUC / recall@k for one split."""
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
    """
    Maximize PR-AUC with an F1 overfit penalty (anti-overfit philosophy).

    overfit_gap = f1_train - f1_cv
    If gap > threshold, shrink the objective proportional to the violation.
    """
    overfit_gap = float(max(0.0, f1_train - f1_cv))
    if overfit_gap > OVERFIT_F1_GAP_THRESHOLD:
        score = pr_auc_cv / (1.0 + overfit_gap)
    else:
        score = pr_auc_cv
    return float(score), overfit_gap


def make_input_example(X: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    return X.head(n).copy()


# ---------------------------------------------------------------------------
# Logging helpers (sklearn + H2O)
# ---------------------------------------------------------------------------

def log_and_save_model(pipeline, metrics, model_path, model_type, X_example, params=None):
    """Log metrics + sklearn model with explicit signature/input example."""
    if params:
        mlflow.log_params(params)

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
        name="model",
        serialization_format="cloudpickle",
        signature=signature,
        input_example=input_example,
    )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    _print_summary(model_type, metrics)


def log_and_save_h2o(best_model, metrics, model_type, params=None):
    """Log + register an H2O model (mlflow.h2o flavor)."""
    import mlflow.h2o  # local import: only required when --model_type h2o

    if params:
        mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.h2o.log_model(best_model, name="model")

    _print_summary(model_type, metrics)


def _print_summary(model_type, metrics):
    print(f"\n✅ Model {model_type} finished.")
    print(f"🟣 PR-AUC Train: {metrics.get('pr_auc_train', 0):.4f}")
    print(f"🟡 PR-AUC CV:    {metrics.get('pr_auc_cv', 0):.4f}")
    print(f"🟢 PR-AUC Test:  {metrics.get('pr_auc_test', 0):.4f}")
    print(f"🔵 F1 Train/Test: {metrics.get('f1_train', 0):.4f} / {metrics.get('f1_test', 0):.4f}")
    print(f"🟠 Overfit F1 gap: {metrics.get('overfit_f1_gap', 0):.4f}")


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
    """Fit, evaluate (CV + train + temporal test), and log to MLflow."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"🔄 Stratified CV probabilities for {model_type}...")
    cv_proba = cross_val_predict(
        pipeline, X_train, y_train, cv=cv, n_jobs=-1, method="predict_proba"
    )[:, 1]
    cv_pred = (cv_proba >= 0.5).astype(int)
    metrics_cv = calculate_classification_metrics(y_train, cv_pred, cv_proba, prefix="")
    metrics_cv.update(
        {
            "f1_cv": metrics_cv["f1"],
            "roc_auc_cv": metrics_cv["roc_auc"],
            "pr_auc_cv": metrics_cv["pr_auc"],
            "recall_at_k_cv": metrics_cv["recall_at_k"],
        }
    )

    pipeline.fit(X_train, y_train)

    train_proba = predict_proba_positive(pipeline, X_train)
    train_pred = pipeline.predict(X_train)
    metrics_train = {
        f"{k}_train": v
        for k, v in calculate_classification_metrics(y_train, train_pred, train_proba).items()
    }

    test_proba = predict_proba_positive(pipeline, X_test)
    test_pred = pipeline.predict(X_test)
    metrics_test = {
        f"{k}_test": v
        for k, v in calculate_classification_metrics(y_test, test_pred, test_proba).items()
    }

    overfit_f1_gap = float(max(0.0, metrics_train["f1_train"] - metrics_test["f1_test"]))
    optuna_score, overfit_cv_gap = build_optuna_score(
        metrics_cv["pr_auc_cv"], metrics_train["f1_train"], metrics_cv["f1_cv"]
    )

    metrics = {
        **metrics_cv,
        **metrics_train,
        **metrics_test,
        "overfit_f1_gap": overfit_f1_gap,
        "overfit_f1_gap_cv": overfit_cv_gap,
        "optuna_score": optuna_score,
        "overfit_threshold": float(OVERFIT_F1_GAP_THRESHOLD),
        "recall_at_k_fraction": float(RECALL_AT_K_FRACTION),
        "passes_overfit_gate": float(overfit_f1_gap <= OVERFIT_F1_GAP_THRESHOLD),
    }

    if is_optimized and params is not None:
        params = {**params, "optimized": True}

    log_and_save_model(pipeline, metrics, model_path, model_type, X_example=X_train, params=params)
    return metrics


def log_mlflow_data(params, metrics):
    if params:
        mlflow.log_params(params)
    mlflow.log_metrics(metrics)


# ---------------------------------------------------------------------------
# H2O AutoML (classification)
# ---------------------------------------------------------------------------

def train_h2o_automl(df: pd.DataFrame, target: str, feature_columns, max_runtime_secs: int = 120):
    """Run H2O AutoML for binary classification on the pipe-break dataset."""
    import h2o
    from h2o.automl import H2OAutoML

    h2o.init(nthreads=-1, strict_version_check=False)

    keep_cols = list(feature_columns) + [target]
    hf = h2o.H2OFrame(df[keep_cols])
    hf[target] = hf[target].asfactor()  # force binary classification, not regression

    train, test = hf.split_frame(ratios=[0.8], seed=42)
    x_columns = [c for c in feature_columns]

    aml = H2OAutoML(
        max_runtime_secs=max_runtime_secs,
        seed=42,
        project_name="kw_water_main_break_risk",
        sort_metric="AUCPR",
    )
    aml.train(x=x_columns, y=target, training_frame=train)
    return aml.leader, aml, train, test


def _h2o_perf_metrics(perf) -> dict:
    """Extract classification metrics from an H2OModelMetrics object."""
    try:
        f1 = float(perf.F1()[0][1]) if perf.F1() else 0.0
    except Exception:
        f1 = 0.0
    return {
        "roc_auc": float(perf.auc()),
        "pr_auc": float(perf.aucpr()),
        "f1": f1,
    }


def run_h2o_branch(max_runtime_secs: int = 120):
    """H2O AutoML classification branch, mirrors the sklearn metric contract."""
    X_train, X_test, y_train, y_test, cleaned_df = prepare_pipe_break_data()
    feature_columns = X_train.columns.tolist()

    full_df = pd.concat(
        [
            pd.concat([X_train, y_train.rename(TARGET_COLUMN)], axis=1),
            pd.concat([X_test, y_test.rename(TARGET_COLUMN)], axis=1),
        ],
        axis=0,
    )

    print("🚀 Launching H2O AutoML (classification)...")
    with mlflow.start_run(run_name="run_h2o_automl"):
        best_model, aml_obj, train_frame, test_frame = train_h2o_automl(
            full_df, TARGET_COLUMN, feature_columns, max_runtime_secs=max_runtime_secs
        )

        perf_train = best_model.model_performance(train=True)
        perf_test = best_model.model_performance(test_data=test_frame)
        perf_cv = best_model.model_performance(xval=True)

        algo_name = best_model.algo
        print("\n📊 === H2O LEADERBOARD ===")
        print(aml_obj.leaderboard.as_data_frame())

        metrics_train = _h2o_perf_metrics(perf_train)
        metrics_cv = _h2o_perf_metrics(perf_cv)
        metrics_test = _h2o_perf_metrics(perf_test)

        overfit_f1_gap = float(max(0.0, metrics_train["f1"] - metrics_test["f1"]))
        optuna_score, overfit_cv_gap = build_optuna_score(
            metrics_cv["pr_auc"], metrics_train["f1"], metrics_cv["f1"]
        )

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
            params={"model_type": f"H2O_{algo_name.upper()}"},
        )


# ---------------------------------------------------------------------------
# Dataset params (shared across all branches)
# ---------------------------------------------------------------------------

def _dataset_params() -> dict:
    params = {
        "horizon_years": HORIZON_YEARS,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train KW water-main break classifiers with MLflow tracking."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="xgboost",
        choices=[
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
        ],
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter search")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--n_est_min", type=int, default=100)
    parser.add_argument("--n_est_max", type=int, default=1000)
    parser.add_argument("--depth_min", type=int, default=3)
    parser.add_argument("--depth_max", type=int, default=15)
    parser.add_argument(
        "--h2o_max_runtime_secs",
        type=int,
        default=120,
        help="H2O AutoML time budget in seconds",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # --- Branch A: H2O AutoML classification ---
    if args.model_type == "h2o":
        run_h2o_branch(max_runtime_secs=args.h2o_max_runtime_secs)
        return

    # 1. Temporal data preparation (no random split).
    X_train, X_test, y_train, y_test, cleaned_df = prepare_pipe_break_data()
    for frame in (X_train, X_test):
        for col in frame.select_dtypes(include=["integer"]).columns:
            frame[col] = frame[col].astype(float)
    print(
        f"Temporal split @ {TEMPORAL_SPLIT_DATE}: "
        f"train={len(X_train)} test={len(X_test)} "
        f"(pos_train={int(y_train.sum())}, pos_test={int(y_test.sum())})"
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
    shared_params = _dataset_params()

    # --- Branch B: Optuna hyperparameter search ---
    if args.tune:
        print(
            f"🎯 Optuna tuning for {args.model_type} "
            f"({args.n_trials} trials, maximize PR-AUC with F1 overfit penalty)..."
        )
        with mlflow.start_run(run_name=f"Optuna_Study_{args.model_type}"):
            mlflow.log_params({**shared_params, "model_type": args.model_type, "tune": True})

            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int(
                        "n_estimators", args.n_est_min, args.n_est_max, step=50
                    ),
                    "max_depth": trial.suggest_int("max_depth", args.depth_min, args.depth_max),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "alpha": trial.suggest_float("alpha", 0.1, 10.0, log=True),
                }
                trial_pipeline = get_experiment_models(
                    X_train.columns.tolist(), **params
                ).get(run_name_mapping[args.model_type])

                with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    cv_proba = cross_val_predict(
                        trial_pipeline, X_train, y_train, cv=cv, n_jobs=-1, method="predict_proba"
                    )[:, 1]
                    cv_pred = (cv_proba >= 0.5).astype(int)
                    cv_metrics = calculate_classification_metrics(y_train, cv_pred, cv_proba)

                    trial_pipeline.fit(X_train, y_train)
                    train_proba = predict_proba_positive(trial_pipeline, X_train)
                    train_pred = trial_pipeline.predict(X_train)
                    train_metrics = calculate_classification_metrics(y_train, train_pred, train_proba)

                    test_proba = predict_proba_positive(trial_pipeline, X_test)
                    test_pred = trial_pipeline.predict(X_test)
                    test_metrics = calculate_classification_metrics(y_test, test_pred, test_proba)

                    optuna_score, overfit_cv_gap = build_optuna_score(
                        cv_metrics["pr_auc"], train_metrics["f1"], cv_metrics["f1"]
                    )
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
                        "overfit_f1_gap": float(
                            max(0.0, train_metrics["f1"] - test_metrics["f1"])
                        ),
                        "optuna_score": optuna_score,
                    }
                    log_mlflow_data(
                        params={**params, "model_type": args.model_type}, metrics=trial_metrics
                    )
                    return optuna_score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=args.n_trials)
            print(f"\n🏆 Best params: {study.best_params}")

            final_params = {**study.best_params, "model_type": args.model_type, **shared_params}
            champion = get_experiment_models(
                X_train.columns.tolist(),
                n_estimators=study.best_params.get("n_estimators", args.n_estimators),
                max_depth=study.best_params.get("max_depth", args.max_depth),
                learning_rate=study.best_params.get("learning_rate", args.learning_rate),
                alpha=study.best_params.get("alpha", args.alpha),
            ).get(run_name_mapping[args.model_type])

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

    # --- Branch C: classic manual training ---
    models = get_experiment_models(
        X_train.columns.tolist(),
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        alpha=args.alpha,
    )
    run_name = run_name_mapping[args.model_type]
    pipeline = models[run_name]
    with mlflow.start_run(run_name=run_name):
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

    _ = cleaned_df


if __name__ == "__main__":
    main()