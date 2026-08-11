# Architecture Decision Records — KW water-main break prediction

## ADR-001 — Approche hybride classification + régression

**Statut :** Accepté

**Décision :** entraîner un second modèle de régression estimant `years_until_break`,
utilisé uniquement quand le classifieur prédit une rupture (`label=1`).

**Raison :** répondre non seulement à "est-ce que ça va casser" mais aussi
"dans combien de temps", pour prioriser les interventions.

**Limite assumée :** le modèle de régression est entraîné uniquement sur les
conduites ayant une rupture future déjà enregistrée dans l'historique
(données non censurées). Les lignes avec `years_until_break = NaN` sont exclues
de l'entraînement de la régression mais restent utilisées pour la classification.
Le régresseur répond à "si ça casse, dans combien de temps pour un profil similaire",
pas à une date garantie pour toute conduite.

**Conséquences :**
- CLI : `python -m src.train --task regression --model_type xgb_reg`
- MLflow : runs tagués `params.task=regression` ; Model Gate choisit le RMSE test minimal
- API : champ optionnel `estimated_years_until_break` sur `/predict` si `break_within_horizon=1`
- Streamlit : affiche l'estimation temporelle uniquement pour la classe prédite 1
