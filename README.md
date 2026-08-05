
```markdown
# Prédiction des Risques de Bris d'Aqueduc (Kitchener–Waterloo) — MLOps de bout en bout

Pipeline MLOps permettant de prédire si une conduite d'eau se brisera dans un horizon temporel défini. Adapté d'un précédent projet de régression sur la vitesse d'avions, il conserve Docker Compose, MLflow, un module d'entraînement, FastAPI, Streamlit, ainsi qu'une passerelle de modèle (Model Gate) de niveau industriel.

**Tâche de prédiction :** classification binaire  
`break_within_horizon = 1` si et seulement si la conduite se brise dans les **H = 5 prochaines années**.

**Données du domaine :** Incidents de bris d'aqueduc à Kitchener–Waterloo  
(référence : [js3lliott/water-main-break-prediction-KW](https://github.com/js3lliott/water-main-break-prediction-KW)).

---

## 1. Architecture

Quatre services Compose :

| Service | Rôle |
|---------|------|
| `mlflow` | Suivi des expériences + artefacts des modèles (`http://localhost:5050`) |
| `trainer` | Entraînement / Optimisation Optuna (`src/train.py`) |
| `api` | Inférence FastAPI + Chargement du modèle champion (`http://localhost:8000`) |
| `frontend` | Interface utilisateur Streamlit (`http://localhost:8501`) |

Package Python principal sous `src/` :

| Module | Responsabilité |
|--------|----------------|
| `labeling.py` | Étiquettes d'Instantanes Historiques / Fenêtrage (Historical Snapshot) |
| `preprocessing.py` | Caractéristiques temporelles + séparation temporelle train/test |
| `train.py` | Entraînement de classification, métriques, journalisation MLflow |
| `model_gate.py` | Sélection du modèle champion (PR-AUC / surapprentissage F1) |
| `schema.py` / `config.py` | Contrat de données et constantes du projet |

---

## 2. Démarrage rapide (Docker)

Clonez le dépôt et démarrez la stack MLOps complète en arrière-plan :

```bash
git clone [https://github.com/Wuradclan/water-main-break-prediction.git](https://github.com/Wuradclan/water-main-break-prediction.git)
cd water-main-break-prediction

docker compose up -d --build

```

| Point de terminaison | URL (PC Hôte) |
| --- | --- |
| Interface MLflow | http://localhost:5050 |
| Documentation API (Swagger) | http://localhost:8000/docs |
| Interface Streamlit | http://localhost:8501 |

### Gestion des services & Logs

```bash
# Voir les logs en temps réel pour tous les services (Appuyez sur Ctrl+C pour quitter)
docker compose logs -f

# Voir les logs pour un service spécifique
docker compose logs -f api
docker compose logs -f mlflow

```

### Entraînement des modèles

Entraînez un modèle en exécutant le script d'entraînement directement à l'intérieur du conteneur `trainer` en cours d'exécution :

```bash
# Entraînement manuel classique (ex: Régression Logistique, Random Forest)
docker compose exec trainer python -m src.train --model_type logistic
docker compose exec trainer python -m src.train --model_type random_forest

# Optimisation des hyperparamètres Optuna (maximise le PR-AUC avec pénalité de surapprentissage F1)
docker compose exec trainer python -m src.train --model_type xgboost --tune --n_trials 15

# Classification H2O AutoML
docker compose exec trainer python -m src.train --model_type h2o

```

> **Astuce de débogage :** Pour ouvrir un terminal Bash interactif dans le conteneur trainer :
> `docker compose exec trainer bash`

### Rechargement du modèle champion dans l'API

Après avoir entraîné un nouveau meilleur modèle, déclenchez la Model Gate pour charger le nouveau champion dans l'API en direct.

Si vous exécutez ceci depuis le **terminal de votre PC hôte** :

```bash
curl -X POST http://localhost:8000/reload-model

```

Si vous exécutez ceci **depuis un conteneur Docker** (par exemple, le trainer) :

```bash
curl -X POST http://api:8000/reload-model

```

### Arrêt de l'environnement

```bash
# Arrêter tous les services
docker compose down

# Arrêter tous les services et supprimer les volumes (supprime la base de données/les exécutions locales)
docker compose down -v

```

---

## 3. Développement local (sans Compose)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Régénérer les snapshots (optionnel ; le fichier traité est épinglé)
python -m src.labeling

# Entraîner (journalise dans la base locale sqlite:///mlflow.db par défaut)
python -m src.train --model_type logistic

# Inspecter le champion via la Model Gate
python -m src.model_gate

# API (terminal séparé)
export MLFLOW_TRACKING_URI="sqlite:///$(pwd)/mlflow.db"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Streamlit (terminal séparé)
export API_BASE_URL="http://localhost:8000"
streamlit run app/streamlit_app.py --server.port 8501

```

Sur macOS, XGBoost peut nécessiter OpenMP (`brew install libomp`). Les images Docker installent déjà `libomp`.

---

## 4. Contrat de données

Fichiers épinglés (voir `data/README.md` et `data/SHA256SUMS`) :

| Chemin | Description |
| --- | --- |
| `data/raw/Water_Main_Breaks.csv` | Incidents bruts de bris à KW |
| `data/processed/pipe_break_snapshots.csv` | Instantanés historiques étiquetés |
| `data/SHA256SUMS` | Hachages de contenu pour la reproductibilité |

### Caractéristiques utilisées à l'inférence / entraînement

| Caractéristique | Signification |
| --- | --- |
| `material` | Matériau de la conduite (ex: CI, DI, PVC) |
| `diameter_mm` | Diamètre nominal |
| `install_year` | Année d'installation |
| `age_years` | Âge au moment de l'instantané / prédiction `t` |
| `prior_break_count` | Bris avec une date **strictement antérieure** à `t` |
| `years_since_last_break` | Années depuis le dernier bris précédent (`null` si aucun) |

**Cible :** `break_within_horizon` ∈ {0, 1}

**Exclus (fuite de données) :** nature/cause du bris, drapeaux de réparation, champs opérationnels post-incident, et toute information future sur les bris relative à `t`.

**Reporté :** longueur de la conduite `length_m` (jointure avec l'inventaire Water_Mains).

Chaque exécution d'entraînement journalise `dataset_raw_sha256` / `dataset_snapshots_sha256` dans MLflow.

---

## 5. Étiquetage : Instantané Historique / Fenêtrage

L'extrait de données ouvert de KW contient **uniquement des conduites qui se sont brisées au moins une fois** (aucun inventaire confirmé de conduites n'ayant jamais subi de bris). Nous synthétisons donc des étiquettes supervisées avec un fenêtrage historique (horizon **H = 5 ans**) :

Pour chaque bris à l'instant `T` sur une conduite :

1. Instantané **Positif** à `t = T − H` → étiquette `1` (le bris tombe dans `(t, t+H]`).
2. Instantanés **Négatifs** à `t = T − H − k` pour `k ∈ {1,2,3,4,5}`, conservés uniquement si `(t, t+H]` ne contient **aucun** bris
(`k = 5` récupère le négatif classique `T − 2H`).

Les caractéristiques sont calculées **uniquement à partir de `t**` (âge, bris précédents, matériau, diamètre, année d'installation).

### Biais Résiduel Positif-Non Étiqueté (PU)

L'étiquette `0` signifie :

> aucun bris enregistré dans la fenêtre d'horizon future pour une conduite qui apparaît finalement dans l'historique des bris.

Cela ne signifie **pas** que la conduite est saine en permanence, et les conduites qui ne se sont jamais brisées sont absentes du jeu de données. L'utilisation en production sur un inventaire municipal complet doit traiter les scores comme un risque relatif parmi les segments sujets aux bris historiquement observés, à moins qu'un inventaire de vrais négatifs ne soit ajouté.

### Validation Temporelle

* **Aucune séparation aléatoire (train/test split).**
* Coupure : `TEMPORAL_SPLIT_DATE = 2015-01-01` sur la `snapshot_date`.
* Entraînement : `snapshot_date < 2015-01-01` · Test : `snapshot_date ≥ 2015-01-01`.
* Les deux partitions contiennent des positifs et des négatifs (des décalages négatifs plus denses assurent cela).

---

## 6. Entraînement & métriques

Nom de l'expérience : `KW_Water_Main_Break_Risk`

### Classifieurs supportés

`logistic`, `ridge`, `lasso`, `random_forest`, `xgboost`, `extra_trees`, `knn`, `svc`, `mlp`, `stacking`

### Métriques journalisées

Entraînement / CV stratifiée / test temporel :

* **PR-AUC** (métrique principale du champion)
* F1
* ROC-AUC
* recall@k (top 10% des prédictions à plus haut risque)

Les modèles sont journalisés avec une **signature MLflow** explicite + `input_example` (caractéristiques de la conduite).

### Optuna

```bash
docker compose exec trainer python -m src.train --model_type xgboost --tune --n_trials 20

```

Objectif : **maximiser** le PR-AUC avec une pénalité de surapprentissage (overfit) sur le F1 lorsque `(f1_train − f1_cv) > 0.30`.

### Reporté : H2O AutoML

`--model_type h2o` est intentionnellement désactivé pour cette migration de classification. Docker installe toujours Java/libomp pour une éventuelle réactivation future ; sklearn/XGBoost est la voie supportée actuellement.

---

## 7. Passerelle de Modèle (Model Gate)

Implémenté dans `src/model_gate.py`, utilisé par l'API au démarrage et via `/reload-model`.

1. Charger les exécutions MLflow de niveau supérieur (exclure `trial_*` d'Optuna / parents d'études).
2. Exiger `pr_auc_test`, `f1_train`, `f1_test` et un artefact de `modèle` journalisé.
3. **Rejeter** si l'écart de surapprentissage `overfit_f1_gap = max(0, f1_train − f1_test) > 0.30`.
4. Parmi les survivants, **champion = argmax(pr_auc_test)**.
5. Solution de repli si tout est bloqué : le plus petit écart de surapprentissage.

```bash
python -m src.model_gate

```

---

## 8. API d'Inférence

Swagger : http://localhost:8000/docs

| Méthode | Chemin | Objectif |
| --- | --- | --- |
| `POST` | `/predict` | Classe + probabilité |
| `GET` | `/model-info` | Métadonnées du champion |
| `POST` | `/reload-model` | Ré-exécuter la Model Gate |
| `GET` | `/health` | Liveness / URI de suivi |

### Exemple de requête

Depuis votre **PC Hôte** (ex: terminal Mac/Windows) :

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

*(Remarque : Si vous appelez cette API par programmation depuis un autre conteneur Docker comme Streamlit, utilisez `http://api:8000/predict` à la place).*

### Exemple de réponse

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

Les anciens champs du projet d'avion sont rejetés (`extra="forbid"`).

Si `prior_break_count == 0`, envoyez `"years_since_last_break": null` (ou omettez-le).

Résolution de l'URI MLflow : Variable d'environnement `MLFLOW_TRACKING_URI` → Docker `http://mlflow:5000` → Défaut SQLite local.

---

## 9. Interface Utilisateur Streamlit

**Accès Navigateur (PC Hôte) :** http://localhost:8501

* Entrées des caractéristiques des conduites alignées avec `PipeBreakRequest`
* Affiche la classe prédite, la probabilité de bris et les métriques de la Model Gate du champion
* "Reload champion from MLflow" appelle `/reload-model`

**Routage interne Docker :**
Dans `docker-compose.yml`, Streamlit doit pointer vers le conteneur API en utilisant le nom de son service, et non localhost :
`API_BASE_URL="http://api:8000"`

**Surcharge locale (Exécution sans Docker) :**

```bash
export API_BASE_URL="http://localhost:8000"
streamlit run app/streamlit_app.py

```

---

## 10. Tests

```bash
pytest tests/ -q

```

La couverture inclut l'étiquetage (exemple détaillé de l'ASSETID 33550), le prétraitement / l'équilibre des classes de la séparation temporelle, les aides aux métriques d'entraînement, la sélection de la Model Gate et les contrats de schéma/prédiction de l'API.

---

## 11. Maintenance

```bash
docker compose down          # arrêter les services
docker compose down -v       # supprimer également les volumes Compose

```

Ignorés par Git : `.venv/`, `mlruns/`, `mlflow.db`, `models/`, métadonnées de l'IDE.

---

## 12. Limites connues & travaux reportés

| Élément | Statut |
| --- | --- |
| Longueur de conduite (`Water_Mains.Shape__Length`) | **Reporté** |
| Chemin de classification H2O AutoML | **Reporté** |
| Inventaire complet sans bris (vrais négatifs) | Absent de l'extrait de données ouvertes KW |
| Calibration des probabilités / ajustement du seuil | Non ajusté pour la production (défaut 0.5 pour F1/classe) |

---

## 13. Équipe

Mohamed Houari · Peter El-Hadad · Jaime Alfonso Robledo Villacob · Morad Ait Abdellah

```

```
English Version 
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

Train a model by executing the training script directly inside the running trainer container:

```bash
# Classic manual training (e.g., Logistic Regression, Random Forest)
docker compose exec trainer python -m src.train --model_type logistic
docker compose exec trainer python -m src.train --model_type random_forest

# Optuna hyperparameter tuning (maximizes PR-AUC with F1 overfit penalty)
docker compose exec trainer python -m src.train --model_type xgboost --tune --n_trials 15

# H2O AutoML classification
docker compose exec trainer python -m src.train --model_type h2o

```

> **Debug tip:** To open an interactive Bash terminal inside the trainer container:
> `docker compose exec trainer bash`

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
docker compose exec trainer python -m src.train --model_type xgboost --tune --n_trials 20

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
