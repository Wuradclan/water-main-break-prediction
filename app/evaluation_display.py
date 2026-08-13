"""
Pure, Streamlit-free helpers for displaying the fixed-5-year-horizon
classification evaluation (metrics + confusion matrix) fetched from
GET /models/{run_id}/evaluation and GET /models/{run_id}/confusion-matrix.

Kept outside app/pages/ for the same reason as app/formatting.py and
app/risk_messaging.py: importable both by Streamlit and by pytest, and
free of any `streamlit` import so this logic is directly unit-testable.

Design constraints (see task spec):
- This feature concerns *only* the fixed 5-year classification horizon.
- Nothing here should ever read a local reports/ file: every value must
  come from the /evaluation and /confusion-matrix API responses, which are
  themselves scoped to a single run_id (source of truth = the MLflow run).
"""

from __future__ import annotations

from typing import Optional

HORIZON_YEARS = 5

CONFUSION_MATRIX_SECTION_CAPTION = (
    "Matrice de confusion — jeu de test temporel — horizon fixe de 5 ans."
)

CONFUSION_MATRIX_IMAGE_CAPTION = (
    "Matrice de confusion — test temporel — "
    "prédiction de bris dans les 5 prochaines années"
)

CONFUSION_MATRIX_EXPLANATION = (
    "Les faux négatifs correspondent à des conduites ayant subi un bris "
    "dans les 5 ans sans avoir été signalées par le modèle.\n"
    "Les faux positifs correspondent à des alertes préventives non suivies "
    "d'un bris observé dans l'horizon."
)

CONFUSION_MATRIX_UNAVAILABLE_MESSAGE = (
    "Matrice de confusion indisponible pour ce run "
    "(entraîné avant l'ajout de cette fonctionnalité, ou artefact manquant)."
)


def should_display_confusion_matrix(task: Optional[str]) -> bool:
    """
    True only for classification runs.

    This feature never applies to regression (years_until_break): the
    confusion matrix is meaningless there, so callers must skip the whole
    block rather than show empty/dash placeholders.
    """
    return str(task).strip().lower() == "classification"


def format_confusion_count(value) -> str:
    """
    Render one TN/FP/FN/TP count for st.metric()-style display.

    Returns "—" (never "None"/"nan") when the value is missing, which
    happens for older classification runs predating this feature or a run
    whose confusion-matrix metrics weren't logged.
    """
    if value is None:
        return "—"
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "—"


def extract_confusion_counts(evaluation: Optional[dict]) -> dict:
    """
    Pull the four counts out of a /models/{run_id}/evaluation JSON payload.

    Always returns the four keys (defaulting to None when absent) so the
    caller can format them uniformly, whether or not the run is a
    classification run or has the artifact available.
    """
    confusion_matrix = (evaluation or {}).get("confusion_matrix") or {}
    return {
        "true_negatives": confusion_matrix.get("true_negatives"),
        "false_positives": confusion_matrix.get("false_positives"),
        "false_negatives": confusion_matrix.get("false_negatives"),
        "true_positives": confusion_matrix.get("true_positives"),
    }


def confusion_matrix_image_url(api_base_url: str, run_id: str) -> str:
    """Build the GET /models/{run_id}/confusion-matrix URL for st.image()."""
    return f"{api_base_url.rstrip('/')}/models/{run_id}/confusion-matrix"
