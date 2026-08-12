# Data Model

## Sources

- `data/raw/Water_Main_Breaks.csv` : événements historiques de rupture
- `data/processed/pipe_break_snapshots.csv` : instantanés générés à des dates de prédiction

## Contrat de modélisation

### Métadonnées

| Colonne | Rôle |
|---|---|
| `asset_id` | Identifiant de conduite; exclu des features |
| `snapshot_date` | Date de l'instantané; utilisée pour le split temporel |
| `horizon_years` | Horizon de label : 1, 2 ou 5 |

### Features

| Colonne | Type | Description |
|---|---|---|
| `material` | catégorielle | Matériau de la conduite |
| `diameter_mm` | numérique | Diamètre nominal en millimètres |
| `install_year` | numérique | Année d'installation |
| `age_years` | numérique | Âge à la date de snapshot |
| `prior_break_count` | numérique | Nombre de ruptures strictement antérieures au snapshot |
| `years_since_last_break` | numérique/null | Délai depuis la dernière rupture antérieure |

### Cible

| Colonne | Type | Description |
|---|---|---|
| `break_within_horizon` | binaire | 1 si rupture dans l'horizon futur; 0 sinon |

## Règles temporelles

- `years_since_last_break` n'utilise que les événements avec `incident_date < snapshot_date`.
- Les labels sont créés dans l'étape d'étiquetage, jamais recalculés à partir des snapshots.
- Les champs post-bris ou post-réparation sont interdits dans le frame de modélisation.
- Si `prior_break_count = 0`, `years_since_last_break` doit être nul.

## Split

- Train : `snapshot_date < TEMPORAL_SPLIT_DATE`
- Test : `snapshot_date >= TEMPORAL_SPLIT_DATE`
- Chaque partition doit contenir les classes 0 et 1.
