# Technical Specification

## Technologies

- Python 3.11
- Docker et Docker Compose
- scikit-learn, XGBoost, Optuna, H2O
- MLflow pour le tracking et les artefacts
- FastAPI + Pydantic pour l'API
- Streamlit pour l'interface
- pandas / NumPy pour les données

## Services Docker Compose

| Service | Port hôte | Responsabilité |
|---|---:|---|
| `mlflow` | 5050 | Tracking, metadata et artefacts MLflow |
| `trainer` | — | Exécution des scripts d'entraînement |
| `api` | 8000 | Inférence FastAPI et chargement du champion |
| `frontend` | 8501 | Interface Streamlit |

## Configuration d'exécution

- `MLFLOW_TRACKING_URI=http://mlflow:5000` dans le trainer
- `MLFLOW_TRACKING_URI=http://bris-aqueduc-mlflow:5000` dans l'API
- `PYTHONPATH=/app` pour le trainer
- Volumes persistants : `./mlruns:/mlruns` et `.:/app`

## Modèles

- Logistic Regression, Ridge, Lasso
- Random Forest, Extra Trees, XGBoost
- KNN, SVC, MLP
- Stacking
- H2O AutoML

## Exigences non fonctionnelles

- Reproductibilité : paramètres, métriques et artefacts journalisés dans MLflow
- Robustesse : validation Pydantic et Model Gate anti-surapprentissage
- Observabilité : endpoints de santé, informations du champion et logs
- Isolation : services séparés en conteneurs
- Sécurité : ne pas exposer Docker socket publiquement; protéger l'interface si elle est accessible depuis Internet
