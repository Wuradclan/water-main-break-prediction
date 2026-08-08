# Prédiction des Risques de Bris d’Aqueduc (Kitchener–Waterloo) — MLOps de bout en bout

Pipeline MLOps permettant de prédire si une conduite d’eau se brisera dans un horizon temporel défini. Adapté d’un précédent projet de régression sur la vitesse d’avions, il conserve Docker Compose, MLflow, un module d’entraînement, FastAPI, Streamlit, ainsi qu’une passerelle de modèle (*Model Gate*) de niveau industriel.

**Tâche de prédiction :** classification binaire.  
`break_within_horizon = 1` si et seulement si une conduite se brise dans les **H prochaines années**.

**Horizons supportés :** 1, 2 ou 5 ans (5 ans par défaut).  
**Données du domaine :** incidents de bris d’aqueduc à Kitchener–Waterloo (référence : [js3lliott/water-main-break-prediction-KW](https://github.com/js3lliott/water-main-break-prediction-KW)).

---

## 1. Architecture

La stack Docker Compose comprend quatre services :

| Service | Rôle |
|---|---|
| `mlflow` | Suivi des expériences, métriques et artefacts des modèles — [http://localhost:5050](http://localhost:5050) |
| `trainer` | Entraînement, validation temporelle et optimisation Optuna (`src/train.py`) |
| `api` | Inférence FastAPI, Model Gate et orchestration asynchrone des entraînements — [http://localhost:8000](http://localhost:8000) |
| `frontend` | Interface utilisateur Streamlit pour les prédictions et le lancement d’entraînements — [http://localhost:8501](http://localhost:8501) |

```text
                  ┌─────────────────────────────┐
                  │          Streamlit           │
                  │ Prédiction + Entraînement UI │
                  └──────────────┬──────────────┘
                                 │ HTTP
                    ┌────────────▼────────────┐
                    │         FastAPI          │
                    │ /predict · /start-training│
                    │ /training-status         │
                    └───────┬──────────┬───────┘
                            │          │ Docker socket (API seulement)
                            │          ▼
                    ┌───────▼──────┐  ┌──────────────────┐
                    │    MLflow    │  │     trainer      │
                    │ runs/models  │  │ src.train        │
                    └──────────────┘  └──────────────────┘
```

### Modules principaux (`src/`)

| Module | Responsabilité |
|---|---|
| `labeling.py` | Création d’étiquettes et d’instantanés historiques (*historical snapshots*) |
| `preprocessing.py` | Caractéristiques temporelles, protections contre les fuites de données et séparation temporelle train/test |
| `train.py` | Entraînement de classification, métriques, logs de progression et journalisation MLflow |
| `model_gate.py` | Sélection du modèle champion selon PR-AUC et filtre de surapprentissage F1 |
| `schema.py` / `config.py` | Contrat de données, variables et constantes du projet |

---

## 2. Démarrage rapide

Clonez le dépôt, placez-vous sur la branche désirée, puis démarrez la stack :

```bash
git clone https://github.com/Wuradclan/water-main-break-prediction.git
cd water-main-break-prediction

git checkout main
# Ou, avant la fusion : git checkout feature/web-training-ui

docker compose up -d --build
```

Vérifiez l’état des services :

```bash
docker compose ps
```

Accédez ensuite aux interfaces :

| Interface | Adresse |
|---|---|
| Streamlit | [http://localhost:8501](http://localhost:8501) |
| FastAPI / Swagger | [http://localhost:8000/docs](http://localhost:8000/docs) |
| MLflow | [http://localhost:5050](http://localhost:5050) |

Pour arrêter la stack :

```bash
docker compose down
```

Les données MLflow restent persistantes dans le volume local `./mlruns` et ne doivent pas être supprimées si vous souhaitez conserver les expériences et les modèles déjà entraînés.

---

## 3. Entraînement depuis Streamlit

La page **Entraînement** dans Streamlit permet de lancer un modèle depuis le navigateur. L’utilisateur choisit :

- l’algorithme;
- l’horizon de prédiction (1, 2 ou 5 ans);
- l’activation optionnelle d’Optuna;
- le nombre d’essais Optuna.

L’interface n’exécute pas Docker directement. Elle envoie plutôt une requête à FastAPI, qui lance le travail dans le conteneur `trainer` en arrière-plan. Cela évite de bloquer l’interface Streamlit pendant un entraînement long.

### Cycle asynchrone

1. Streamlit appelle `POST /start-training`.
2. FastAPI crée immédiatement un `job_id` et lance `docker exec bris-aqueduc-trainer python -m src.train ...` dans un thread d’arrière-plan.
3. La sortie CLI de l’entraînement (`stdout` et `stderr`) est enregistrée dans un fichier de log.
4. Streamlit consulte périodiquement `GET /training-status/{job_id}` et affiche les logs, le statut et la progression Optuna.
5. À la fin, FastAPI récupère le modèle champion via le Model Gate.
6. L’utilisateur peut cliquer sur **Recharger le champion depuis MLflow** pour que l’API serve le meilleur modèle actif.

Un seul entraînement est autorisé simultanément. Une nouvelle requête pendant un job en cours retourne une réponse HTTP `409 Conflict`.

### Commandes produites par l’interface

Les commandes suivantes restent aussi utilisables manuellement depuis l’hôte Docker :

```bash
# Entraînement classique
docker compose exec trainer python -m src.train --model_type logistic
docker compose exec trainer python -m src.train --model_type random_forest

# Optimisation Optuna : maximise le PR-AUC avec pénalité de surapprentissage F1
docker compose exec trainer python -m src.train --model_type xgboost --tune --n_trials 15

# H2O AutoML
docker compose exec trainer python -m src.train --model_type h2o
```

Exemple avec horizon de 2 ans :

```bash
docker compose exec trainer python -m src.train --model_type random_forest --horizon_years 2
```

---

## 4. API FastAPI

| Endpoint | Méthode | Description |
|---|---|---|
| `/health` | `GET` | État de santé de l’API et état du modèle chargé |
| `/predict` | `POST` | Prédiction d’un risque de bris avec seuil configurable |
| `/model-info` | `GET` | Informations sur le modèle champion chargé |
| `/reload-model` | `POST` | Recharge le champion MLflow sans redémarrer l’API |
| `/start-training` | `POST` | Démarre un entraînement asynchrone et retourne un `job_id` |
| `/training-status/{job_id}` | `GET` | Retourne statut, dernières lignes de logs, code de sortie et résultat du job |

### Exemple de prédiction

```bash
curl -X POST "http://localhost:8000/predict?threshold=0.5" \
  -H "Content-Type: application/json" \
  -d '{
    "material": "CI",
    "diameter_mm": 150,
    "install_year": 1959,
    "age_years": 46,
    "prior_break_count": 3,
    "years_since_last_break": 0.75
  }'
```

---

## 5. Données et prévention des fuites

Les données brutes sont stockées sous `data/raw/Water_Main_Breaks.csv`. Les instantanés préparés sont sous `data/processed/pipe_break_snapshots.csv`.

Les variables de modèle sont :

```text
material
 diameter_mm
install_year
age_years
prior_break_count
years_since_last_break
```

Le pipeline applique les règles suivantes :

- `years_since_last_break` utilise uniquement les bris antérieurs à la date du snapshot;
- les champs post-bris sont exclus via `LEAKAGE_FORBIDDEN_COLUMNS`;
- le train/test split est strictement temporel, jamais aléatoire;
- chaque partition doit contenir les deux classes;
- `years_since_last_break` doit être nul si `prior_break_count = 0` et défini sinon.

---

## 6. Modèles, métriques et Model Gate

### Modèles disponibles

- Logistic Regression, Ridge et Lasso;
- Random Forest et Extra Trees;
- XGBoost;
- KNN et SVC;
- MLP Classifier;
- Stacking Classifier;
- H2O AutoML.

### Métriques enregistrées

- PR-AUC test — métrique principale;
- F1 train et F1 test;
- écart de surapprentissage F1 : `max(0, f1_train - f1_test)`;
- ROC-AUC test;
- Recall@K test.

### Règles du Model Gate

1. Ne considère que les runs MLflow de premier niveau avec un artefact de modèle.
2. Exclut les essais Optuna imbriqués `trial_*`.
3. Requiert `pr_auc_test`, `f1_train` et `f1_test`.
4. Rejette les runs dont l’écart de surapprentissage dépasse `OVERFIT_F1_GAP_THRESHOLD`.
5. Sélectionne le run conforme ayant le meilleur PR-AUC test.
6. Si tous les modèles échouent au filtre, choisit le moins surappris en mode `fallback`.

---

## 7. Développement et sécurité

Le service `api` détient le socket Docker (`/var/run/docker.sock`) pour lancer le conteneur `trainer`. Le service `frontend` ne possède pas ce privilège : Streamlit appelle seulement l’API HTTP.

> **Avertissement :** le socket Docker donne un contrôle important sur l’hôte. Pour une VM publique, protégez Streamlit et FastAPI derrière une authentification et HTTPS, limitez les ports ouverts avec un pare-feu/NSG, et ne partagez pas l’accès avec des utilisateurs non fiables.

Pendant le développement local, Uvicorn peut utiliser `--reload`. Pour un déploiement stable (Oracle Cloud), il est recommandé de le retirer afin d’éviter qu’un changement de fichier ne redémarre l’API pendant un entraînement asynchrone.

---

## 8. Tests de diagnostic

Vérifiez que l’API peut déclencher une commande dans le conteneur trainer :

```bash
docker exec bris-aqueduc-api docker --version
docker exec bris-aqueduc-api docker exec bris-aqueduc-trainer echo "test OK"
```

Résultat attendu :

```text
Docker version ...
test OK
```

Logs utiles :

```bash
docker compose logs -f api
docker compose logs -f trainer
docker compose logs -f frontend
```
