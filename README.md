
```markdown
# Water Main Break Risk Prediction (Kitchener–Waterloo) — End-to-End MLOps

MLOps pipeline that predicts whether a water main will break within a fixed horizon, adapted from a prior aircraft-speed regression stack while preserving Docker Compose, MLflow, trainer, FastAPI, Streamlit, and an industrial Model Gate.

**Prediction task:** binary classification  
`break_within_horizon = 1` iff the pipe breaks within the next **H = 5 years**.

**Domain data:** Kitchener–Waterloo water-main break incidents  
(reference: [js3lliott/water-main-break-prediction-KW](https://github.com/js3lliott/water-main-break-prediction-KW)).

---

## 1. Architecture

Four Compose services:

| Service | Role |
|---------|------|
| `mlflow` | Experiment tracking + model artifacts (`http://localhost:5050`) |
| `trainer` | Training / Optuna runs (`src/train.py`) |
| `api` | FastAPI inference + Model Gate champion loading (`http://localhost:8000`) |
| `frontend` | Streamlit UI (`http://localhost:8501`) |

Core Python package under `src/`:

| Module | Responsibility |
|--------|----------------|
| `labeling.py` | Historical Snapshot / Windowing labels |
| `preprocessing.py` | Time-aware features + temporal train/test split |
| `train.py` | Classification training, metrics, MLflow logging |
| `model_gate.py` | Champion selection (PR-AUC / F1 overfit) |
| `schema.py` / `config.py` | Data contract and project constants |

---

## 2. Quick start (Docker)

Clone the repository and start the entire MLOps stack in the background:

```bash
git clone [https://github.com/Wuradclan/water-main-break-prediction.git](https://github.com/Wuradclan/water-main-break-prediction.git)
cd water-main-break-prediction

docker compose up -d --build

```

| Endpoint | URL (Host PC) |
| --- | --- |
| MLflow UI | http://localhost:5050 |
| API docs (Swagger) | http://localhost:8000/docs |
| Streamlit UI | http://localhost:8501 |

### Managing Services & Logs

```bash
# View real-time logs for all services (Press Ctrl+C to exit)
docker compose logs -f

# View logs for a specific service
docker compose logs -f api
docker compose logs -f mlflow

```

### Training Models

Train a model inside an isolated, ephemeral trainer container. It will automatically connect to MLflow and clean itself up (`--rm`) when finished:

```bash
# Classic manual training (e.g., Logistic Regression, Random Forest)
docker compose run --rm trainer python -m src.train --model_type logistic
docker compose run --rm trainer python -m src.train --model_type random_forest

# Optuna hyperparameter tuning (maximizes PR-AUC with F1 overfit penalty)
docker compose run --rm trainer python -m src.train --model_type xgboost --tune --n_trials 15

# H2O AutoML classification
docker compose run --rm trainer python -m src.train --model_type h2o

```

### Reloading the API Champion

After training a new best model, trigger the Model Gate to load the new champion into the live API.

If running this from your **Host PC terminal**:

```bash
curl -X POST http://localhost:8000/reload-model

```

If running this from **inside a Docker container** (e.g., the trainer):

```bash
curl -X POST http://api:8000/reload-model

```

### Stopping the Environment

```bash
# Stop all services
docker compose down

# Stop all services and wipe volumes (removes local DB/runs)
docker compose down -v

```

---

## 3. Local development (without Compose)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Regenerate snapshots (optional; processed file is pinned)
python -m src.labeling

# Train (logs to local sqlite:///mlflow.db by default)
python -m src.train --model_type logistic

# Inspect Model Gate champion
python -m src.model_gate

# API (separate terminal)
export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Streamlit (separate terminal)
export API_BASE_URL="http://localhost:8000"
streamlit run app/streamlit_app.py --server.port 8501

```

On macOS, XGBoost may need OpenMP (`brew install libomp`). Docker images already install `libomp`.

---

## 4. Data contract

Pinned files (see `data/README.md` and `data/SHA256SUMS`):

| Path | Description |
| --- | --- |
| `data/raw/Water_Main_Breaks.csv` | Raw KW break incidents |
| `data/processed/pipe_break_snapshots.csv` | Labeled historical snapshots |
| `data/SHA256SUMS` | Content hashes for reproducibility |

### Features used at inference / training

| Feature | Meaning |
| --- | --- |
| `material` | Pipe material (e.g. CI, DI, PVC) |
| `diameter_mm` | Nominal diameter |
| `install_year` | Installation year |
| `age_years` | Age at snapshot / prediction time `t` |
| `prior_break_count` | Breaks with date **strictly before** `t` |
| `years_since_last_break` | Years since latest prior break (`null` if none) |

**Target:** `break_within_horizon` ∈ {0, 1}

**Excluded (leakage):** break nature/cause, repair flags, post-incident operational fields, and any future break information relative to `t`.

**Deferred:** pipe `length_m` (Water_Mains inventory join).

Every training run logs `dataset_raw_sha256` / `dataset_snapshots_sha256` to MLflow.

---

## 5. Labeling: Historical Snapshot / Windowing

The KW open extract contains **only pipes that have broken at least once** (no confirmed never-broken inventory). We therefore synthesize supervised labels with historical windowing (horizon **H = 5 years**):

For each break at time `T` on a pipe:

1. **Positive** snapshot at `t = T − H` → label `1` (break falls in `(t, t+H]`).
2. **Negative** snapshots at `t = T − H − k` for `k ∈ {1,2,3,4,5}`, kept only if `(t, t+H]` contains **no** break
(`k = 5` recovers the classic `T − 2H` negative).

Features are computed **as of `t` only** (age, prior breaks, material, diameter, install year).

### Residual Positive-Unlabeled (PU) bias

Label `0` means:

> no recorded break in the forward horizon window for a pipe that eventually appears in the break history

It does **not** mean the pipe is permanently healthy, and pipes that never broke are absent from the dataset. Production use on a full municipal inventory must treat scores as relative risk among historically observed break-prone segments unless a true negative inventory is added.

### Temporal validation

* **No random train/test split.**
* Cutoff: `TEMPORAL_SPLIT_DATE = 2015-01-01` on `snapshot_date`.
* Train: `snapshot_date < 2015-01-01` · Test: `snapshot_date ≥ 2015-01-01`.
* Both partitions contain positives and negatives (denser negative offsets ensure this).

---

## 6. Training & metrics

Experiment name: `KW_Water_Main_Break_Risk`

### Supported classifiers

`logistic`, `ridge`, `lasso`, `random_forest`, `xgboost`, `extra_trees`, `knn`, `svc`, `mlp`, `stacking`

### Logged metrics

Train / stratified CV / temporal test:

* **PR-AUC** (primary champion metric)
* F1
* ROC-AUC
* recall@k (top 10% highest-risk predictions)

Models are logged with an explicit **MLflow signature** + `input_example` (pipe features).

### Optuna

```bash
docker compose run --rm trainer python -m src.train --model_type xgboost --tune --n_trials 20

```

Objective: **maximize** PR-AUC with an F1 overfit penalty when `(f1_train − f1_cv) > 0.30`.

### Deferred: H2O AutoML

`--model_type h2o` is intentionally disabled for this classification migration. Docker still installs Java/libomp for possible future re-enablement; sklearn/XGBoost is the supported path.

---

## 7. Model Gate

Implemented in `src/model_gate.py`, used by the API at startup and `/reload-model`.

1. Load top-level MLflow runs (exclude Optuna `trial_*` / study parents).
2. Require `pr_auc_test`, `f1_train`, `f1_test` and a logged `model` artifact.
3. **Reject** if `overfit_f1_gap = max(0, f1_train − f1_test) > 0.30`.
4. Among survivors, **champion = argmax(pr_auc_test)**.
5. Fallback if unblocked: smallest overfit gap.

```bash
python -m src.model_gate

```

---

## 8. Inference API

Swagger: http://localhost:8000/docs

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/predict` | Class + probability |
| `GET` | `/model-info` | Champion metadata |
| `POST` | `/reload-model` | Re-run Model Gate |
| `GET` | `/health` | Liveness / tracking URI |

### Example request

From your **Host PC** (e.g., Mac/Windows terminal):

```bash
curl -s http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "material": "CI",
    "diameter_mm": 150.0,
    "install_year": 1959.0,
    "age_years": 46.0,
    "prior_break_count": 3.0,
    "years_since_last_break": 0.75
  }'

```

*(Note: If calling this API programmatically from another Docker container like Streamlit, use `http://api:8000/predict` instead).*

### Example response

```json
{
  "break_within_horizon": 0,
  "probability": 0.328,
  "model_name": "logistic [PASS] PR-AUC_test=0.6676 | ...",
  "model_type": "logistic",
  "run_id": "...",
  "pr_auc_test": 0.6676,
  "overfit_f1_gap": 0.0,
  "selection_mode": "champion"
}

```

Aircraft fields are rejected (`extra="forbid"`).

If `prior_break_count == 0`, send `"years_since_last_break": null` (or omit).

MLflow URI resolution: `MLFLOW_TRACKING_URI` env → Docker `http://mlflow:5000` → local SQLite default.

---

## 9. Streamlit UI

**Browser Access (Host PC):** http://localhost:8501

* Pipe feature inputs aligned with `PipeBreakRequest`
* Displays predicted class, break probability, and champion gate metrics
* “Reload champion from MLflow” calls `/reload-model`

**Docker internal routing:**
Inside `docker-compose.yml`, Streamlit must point to the API container using its service name, not localhost:
`API_BASE_URL="http://api:8000"`

**Local override (Running without Docker):**

```bash
export API_BASE_URL="http://localhost:8000"
streamlit run app/streamlit_app.py

```

---

## 10. Tests

```bash
pytest tests/ -q

```

Coverage includes labeling (ASSETID 33550 worked example), preprocessing / temporal split class balance, training metrics helpers, Model Gate selection, and API schema/predict contracts.

---

## 11. Maintenance

```bash
docker compose down          # stop services
docker compose down -v       # also wipe Compose volumes

```

Git-ignored: `.venv/`, `mlruns/`, `mlflow.db`, `models/`, IDE metadata.

---

## 12. Known limitations & deferred work

| Item | Status |
| --- | --- |
| Pipe length (`Water_Mains.Shape__Length`) | **Deferred** |
| H2O AutoML classification path | **Deferred** |
| Full never-broken inventory (true negatives) | Not in open KW breaks extract |
| Probability calibration / threshold tuning | Not production-tuned (default 0.5 for F1/class) |

---

## 13. Team

Mohamed Houari · Peter El-Hadad · Jaime Alfonso Robledo Villacob · Morad Ait Abdellah

```

```