from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Dataset paths (versioned via data/SHA256SUMS)
# ---------------------------------------------------------------------------
raw_breaks_path = project_root / "data" / "raw" / "Water_Main_Breaks.csv"
processed_snapshots_path = project_root / "data" / "processed" / "pipe_break_snapshots.csv"
dataset_checksums_path = project_root / "data" / "SHA256SUMS"

# Legacy aircraft path retained temporarily until later cleanup phases.
file_path = project_root / "data" / "Aiplane_BlueBook.csv"

# ---------------------------------------------------------------------------
# Labeling / prediction horizon
# ---------------------------------------------------------------------------
# Mid-term infrastructure planning horizon for:
# "Will this pipe break within the next H years?"
HORIZON_YEARS = 5

# Observation window of the KW breaks extract (approx.). Snapshots whose
# forward label window is not fully inside this range are dropped to avoid
# censoring artifacts from the recording start date.
OBSERVATION_START = "1985-01-01"
OBSERVATION_END = "2023-01-09"

# Random seed for any sampling inside negative-class generation (not for
# train/test splitting — temporal splits are mandatory).
RANDOM_STATE = 42

# Strict time-based train/test cutoff on snapshot_date:
# train uses snapshots with snapshot_date < cutoff;
# test uses snapshots with snapshot_date >= cutoff.
#
# Chosen so that denser pre-horizon negatives (see labeling) yield BOTH
# classes in train and test under H=5.
TEMPORAL_SPLIT_DATE = "2015-01-01"

# Extra negative snapshot offsets (years before the positive date t=T-H).
# For each break T we keep negatives at t = T - H - k for k in this list,
# only when (t, t+H] contains no break. k=H recovers the original T-2H case.
NEGATIVE_OFFSETS_BEFORE_POSITIVE_YEARS = (1, 2, 3, 4, 5)

# ---------------------------------------------------------------------------
# Training / Model Gate (classification)
# ---------------------------------------------------------------------------
MLFLOW_EXPERIMENT_NAME = "KW_Water_Main_Break_Risk"

# Primary champion metric for ranking models (higher is better).
PRIMARY_METRIC = "pr_auc_test"

# Anti-overfit gate uses F1 gap: (f1_train - f1_test).
# Models exceeding this absolute gap are rejected by the API gate (Phase 4).
OVERFIT_F1_GAP_THRESHOLD = 0.30

# recall@k: fraction of highest-risk predictions inspected (top 10%).
RECALL_AT_K_FRACTION = 0.10

# MLflow tracking:
# - Docker Compose sets MLFLOW_TRACKING_URI=http://mlflow:5000
# - Local Phase-3/4 defaults to a SQLite store (FileStore is deprecated).
MLFLOW_TRACKING_URI_DEFAULT = f"sqlite:///{(project_root / 'mlflow.db').resolve()}"
