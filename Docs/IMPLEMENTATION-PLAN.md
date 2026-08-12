# Implementation Plan

## Phase 1 — Données et étiquetage

- Charger les événements bruts de rupture
- Construire des snapshots historiques par actif
- Générer les labels à horizon 1, 2 ou 5 ans
- Produire / valider `pipe_break_snapshots.csv`

**Critère d'acceptation :** chaque snapshot contient une date, un actif, les caractéristiques disponibles à t et une cible binaire.

## Phase 2 — Prétraitement temporel

- Normaliser le matériau et convertir les variables numériques
- Construire `years_since_last_break`
- Bloquer les colonnes interdites et valider la cohérence des variables
- Réaliser le split train/test temporel

**Critère d'acceptation :** aucune colonne post-événement n'est disponible pour le modèle.

## Phase 3 — Entraînement et expérimentation

- Construire les pipelines scikit-learn
- Entraîner les modèles de référence et les ensembles
- Réaliser le tuning Optuna lorsque demandé
- Enregistrer runs, paramètres, métriques et modèles dans MLflow

**Critère d'acceptation :** chaque run éligible possède PR-AUC test, F1 train/test et un artefact `model`.

## Phase 4 — Model Gate

- Calculer l'écart de surapprentissage F1
- Rejeter les candidats au-delà du seuil
- Sélectionner le meilleur PR-AUC parmi les candidats restants
- Appliquer un fallback si aucun candidat ne passe

**Critère d'acceptation :** la sélection retourne un objet `ChampionSelection` traçable.

## Phase 5 — Inference et UI

- Charger le champion dans FastAPI
- Valider les requêtes avec Pydantic
- Exposer prédiction, santé, informations et rechargement
- Construire l'interface Streamlit

**Critère d'acceptation :** une prédiction est réalisable de l'UI à l'API avec les métadonnées du champion.

## Phase 6 — Déploiement et démonstration

- Construire les images Docker
- Démarrer les services avec Docker Compose
- Vérifier les tests et endpoints
- Documenter les commandes et la procédure de déploiement Oracle Cloud
