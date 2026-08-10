# Product Requirements Document (PRD)

## Produit

**Water Main Break Prediction** est une plateforme MLOps qui estime la probabilité qu'une conduite municipale subisse une rupture dans un horizon futur de 1, 2 ou 5 ans.

## Problème

Les municipalités gèrent un réseau vieillissant. Une intervention réactive après une rupture génère des interruptions de service, des coûts d'urgence et des risques de dommages. Les ressources étant limitées, les équipes ont besoin d'un moyen transparent de prioriser les actifs les plus risqués.

## Utilisateurs cibles

- Gestionnaires d'infrastructures et services municipaux d'aqueduc
- Analystes de données / équipe MLOps
- Équipe du projet 420-D62

## Objectifs

1. Produire une probabilité de rupture pour une conduite et un horizon choisi.
2. Fournir une classe de décision configurable par un seuil.
3. Comparer plusieurs modèles et sélectionner automatiquement un champion robuste.
4. Permettre l'entraînement, le suivi et la démonstration dans une stack conteneurisée.

## Fonctionnalités principales

- Saisie : matériau, diamètre, année d'installation, âge, nombre de ruptures antérieures et délai depuis la dernière rupture
- Prédiction via API FastAPI et interface Streamlit
- Entraînement de modèles classiques, ensembles, XGBoost, MLP, stacking et H2O AutoML
- Suivi MLflow : paramètres, métriques, artefacts et runs
- Model Gate : PR-AUC test comme métrique principale et filtre d'écart F1 train/test
- Rechargement à chaud du champion depuis MLflow

## Indicateurs de succès

- PR-AUC sur l'ensemble de test temporel
- F1 test et écart de surapprentissage F1
- ROC-AUC test et Recall@K test
- Modèle champion disponible dans l'API après un entraînement réussi
- Démonstration fonctionnelle via Docker Compose

## Hors périmètre actuel

- Garantie absolue d'absence de rupture
- Ordonnancement opérationnel réel des travaux municipaux
- Intégration des données de capteurs en temps réel
- Longueur de conduite : variable explicitement différée
