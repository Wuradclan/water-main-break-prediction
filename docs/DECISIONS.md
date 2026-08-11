# Architecture Decision Records — water-main-break-prediction

## ADR-001 — Approche hybride classification + régression

**Décision** : entraîner un second modèle de régression estimant `years_until_break`,
utilisé uniquement quand le classifieur prédit une rupture (`label=1`).

**Raison** : répondre non seulement à "est-ce que ça va casser" mais aussi
"dans combien de temps", pour prioriser les interventions.

**Limite assumée** : le modèle de régression est entraîné uniquement sur les
conduites ayant une rupture future déjà enregistrée dans l'historique
(données non censurées). Il répond à "si ça casse, dans combien de temps
pour un profil similaire", pas à une date garantie pour toute conduite.
