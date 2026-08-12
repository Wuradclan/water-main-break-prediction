# Architecture Decision Records (ADR)

## ADR-001 — Validation temporelle au lieu d'un split aléatoire

**Décision :** séparer l'entraînement et le test à partir de `snapshot_date`.

**Raison :** simuler une prédiction réelle du futur et éviter qu'une information plus récente soit présente dans les deux partitions.

**Conséquence :** performance parfois plus prudente, mais évaluation plus crédible.

## ADR-002 — PR-AUC comme métrique primaire

**Décision :** sélectionner prioritairement selon `pr_auc_test`.

**Raison :** le problème de rupture est déséquilibré; l'accuracy peut être trompeuse.

## ADR-003 — Model Gate avec écart F1

**Décision :** rejeter un modèle lorsque `max(0, f1_train - f1_test)` dépasse le seuil configuré.

**Raison :** le meilleur score d'entraînement n'est pas nécessairement généralisable.

## ADR-004 — Fallback de modèle

**Décision :** si aucun modèle ne passe le Gate, choisir le moins surappris et l'indiquer comme `fallback`.

**Raison :** garder l'API disponible sans masquer le risque de qualité.

## ADR-005 — MLflow pour le cycle de vie

**Décision :** journaliser les modèles et métriques dans MLflow.

**Raison :** assurer la traçabilité du champion, des paramètres et des artefacts.

## ADR-006 — Docker Compose sur une VM unique

**Décision :** orchestrer MLflow, trainer, API et frontend via Docker Compose.

**Raison :** simplicité, reproductibilité et adéquation au contexte pédagogique.

## ADR-007 — Longueur de conduite différée

**Décision :** ne pas utiliser `length_m` dans l'inférence actuelle.

**Raison :** la variable est explicitement différée dans le contrat d'interface; elle pourra être réintroduite après validation de qualité.
