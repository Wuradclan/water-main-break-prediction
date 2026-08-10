# Product Specification

## Cas d'usage principal : prédire le risque d'une conduite

1. L'utilisateur choisit l'horizon de prédiction : 1, 2 ou 5 ans.
2. Il fournit les caractéristiques de la conduite.
3. Streamlit envoie la requête à l'API FastAPI.
4. L'API charge le modèle champion sélectionné dans MLflow.
5. L'API retourne la classe `break_within_horizon`, la probabilité, le seuil et les informations de sélection.

## Entrées de prédiction

| Champ | Type | Règle |
|---|---:|---|
| `material` | texte | Code matériau; valeurs invalides normalisées vers `UNKNOWN` |
| `diameter_mm` | nombre | Strictement supérieur à 0 |
| `install_year` | nombre | Entre 1800 et 2100 |
| `age_years` | nombre | Supérieur ou égal à 0 |
| `prior_break_count` | nombre | Supérieur ou égal à 0 |
| `years_since_last_break` | nombre/null | Obligatoire si `prior_break_count > 0`; absent/null sinon |

## Sortie de prédiction

```json
{
  "break_within_horizon": 1,
  "probability": 0.74,
  "threshold": 0.5,
  "model_name": "xgboost [PASS] ...",
  "model_type": "xgboost",
  "run_id": "...",
  "pr_auc_test": 0.0,
  "overfit_f1_gap": 0.0,
  "selection_mode": "champion"
}
```

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Santé du service et disponibilité d'un modèle |
| `POST /predict` | Réalise une prédiction pour une conduite |
| `GET /model-info` | Métadonnées et métriques du champion actif |
| `POST /reload-model` | Recharge le champion depuis MLflow |

## Règles d'acceptation

- Une requête invalide retourne une erreur HTTP 422 ou 400 explicite.
- Une prédiction est refusée avec HTTP 503 si aucun champion n'est chargé.
- La réponse contient une probabilité dans l'intervalle [0, 1].
- Le modèle utilisé est traçable par son `run_id` MLflow.
