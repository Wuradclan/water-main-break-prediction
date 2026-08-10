# Water Main Break Prediction — Kitchener-Waterloo

Projet MLOps de classification binaire pour estimer le risque qu'une conduite d'eau subisse une rupture dans un horizon de **1, 2 ou 5 ans**.

## Objectif

Aider les équipes municipales à prioriser les inspections et les remplacements préventifs à partir des caractéristiques physiques de la conduite et de son historique de ruptures.

## Fonctionnalités

- Préparation de données temporelle, avec prévention explicite de fuite de données
- Entraînement et comparaison de plusieurs familles de modèles
- Optimisation d'hyperparamètres avec Optuna
- Suivi des expériences, métriques et artefacts avec MLflow
- Sélection automatique du modèle champion par un **Model Gate** industriel
- API FastAPI de prédiction et interface Streamlit
- Déploiement multi-services avec Docker Compose

## Démarrage rapide

```bash
git clone -b dev https://github.com/Wuradclan/water-main-break-prediction.git
cd water-main-break-prediction
docker compose up -d --build
docker compose ps
```

Services par défaut :

- Streamlit : `http://localhost:8501`
- FastAPI : `http://localhost:8000`
- Documentation FastAPI : `http://localhost:8000/docs`
- MLflow : `http://localhost:5050`

## Entraînement manuel

```bash
docker compose exec trainer python -m src.train --model_type logistic
docker compose exec trainer python -m src.train --model_type random_forest
docker compose exec trainer python -m src.train --model_type xgboost --tune --n_trials 15
docker compose exec trainer python -m src.train --model_type h2o
```

## Structure

```text
api/        Service FastAPI
app/        Interface Streamlit
data/       Données brutes et snapshots transformés
mlflow/     Configuration MLflow
src/        Prétraitement, entraînement, Model Gate et prédiction
tests/      Tests automatisés
train/      Image / environnement d'entraînement
```

Consulter les documents `PRODUCT-REQUIREMENTS.md`, `ARCHITECTURE.md` et `TECHNICAL-SPECIFICATION.md` pour les détails.
