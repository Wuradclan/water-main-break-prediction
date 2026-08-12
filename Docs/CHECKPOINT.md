# Checkpoint

## État actuel

Le projet dispose d'une chaîne MLOps complète pour la classification du risque de bris de conduites d'eau de Kitchener-Waterloo.

## Réalisé

- Dataset brut et dataset de snapshots transformés
- Features physiques et temporelles sous contrat strict
- Prévention des fuites temporelles et split train/test temporel
- Entraînement de 10 familles de modèles, Optuna et H2O AutoML
- Journalisation MLflow des modèles, paramètres, métriques et artefacts
- Model Gate : filtre de surapprentissage F1, champion PR-AUC, fallback
- API FastAPI avec validation Pydantic
- Interface Streamlit de prédiction et rechargement du champion
- Services Docker Compose : MLflow, trainer, API, frontend
- Tests automatisés pour API, prétraitement, étiquetage, Model Gate et métriques

## À vérifier avant livraison / démonstration

- Lancer un entraînement final sur l'horizon retenu
- Noter PR-AUC, F1 train/test, ROC-AUC et Recall@K du champion
- Vérifier dans MLflow que l'artefact `model` est bien enregistré
- Appeler `/reload-model`, puis tester `/predict` et l'interface Streamlit
- Capturer une preuve visuelle : MLflow, API /docs et Streamlit
- Vérifier volumes `./mlruns` avant redéploiement de la VM

## Risques ouverts

- Résultats numériques finaux non documentés dans ce checkpoint
- Entraînements longs : prévoir une exécution asynchrone pour ne pas bloquer l'interface
- Sécurité : ne pas exposer le socket Docker ni Streamlit publiquement sans authentification

## Prochaine action recommandée

Finaliser le run d'entraînement, documenter ses métriques, puis tester la stack complète sur la VM Oracle Cloud avec Docker Compose.
