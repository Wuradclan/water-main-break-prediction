"""Pure, Streamlit-free UI logic reconciling classification and regression.

Rationale for duplicating HORIZON_YEARS / CLASSIFICATION_THRESHOLD /
THRESHOLD_MARGIN here instead of importing them from `src.config`: the
Streamlit frontend container only ever has `app/` on `sys.path` (Streamlit's
bootstrap inserts the *script's own directory*, not the repository root), so
`from src.config import ...` is not reliably importable there — the same
constraint that led to `app/formatting.py` being self-contained. Both files
are placed outside `app/pages/` so Streamlit does not register them as pages.

Everything below is a pure function (no I/O, no Streamlit calls) so it can be
unit-tested directly, per the "sans dépendre de Streamlit" requirement.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Centralized constants (mirror src/config.py — see module docstring above).
# ---------------------------------------------------------------------------
HORIZON_YEARS = 5
CLASSIFICATION_THRESHOLD = 0.50
THRESHOLD_MARGIN = 0.05

REGRESSION_UNCERTAINTY_NOTICE = (
    "Cette estimation est produite par un modèle distinct du classifieur et "
    "comporte une incertitude."
)

# How close the regression estimate must be to the horizon for a divergence
# with the classifier to be flagged as "near the horizon" (in years).
DIVERGENCE_NEAR_HORIZON_YEARS = 1.0


def _horizon_phrase(horizon_years: float) -> str:
    horizon_int = int(round(horizon_years))
    if horizon_int == 1:
        return "dans l'année à venir"
    return f"dans les {horizon_int} ans"


def format_number_fr(value: float, decimals: int = 1) -> str:
    """Format a number with a French decimal comma, e.g. 5.1 -> "5,1"."""
    return f"{float(value):.{decimals}f}".replace(".", ",")


def format_percent_fr(value: float, decimals: int = 1) -> str:
    """Format a 0..1 probability as a French percentage, e.g. 0.549 -> "54,9%"."""
    return f"{format_number_fr(float(value) * 100.0, decimals=decimals)}%"


def is_near_threshold(
    break_probability: float,
    threshold: float = CLASSIFICATION_THRESHOLD,
    margin: float = THRESHOLD_MARGIN,
) -> bool:
    """True when the probability sits inside the uncertainty band around the threshold.

    A tiny epsilon absorbs binary floating-point rounding (e.g. 0.55 - 0.50
    can evaluate to 0.050000000000000044 in IEEE-754), so boundary values
    such as 0.450 or 0.550 are still treated as inclusive bounds.
    """
    diff = abs(float(break_probability) - float(threshold))
    return diff <= float(margin) + 1e-9


def classify_risk_level(
    break_probability: float,
    threshold: float = CLASSIFICATION_THRESHOLD,
    margin: float = THRESHOLD_MARGIN,
) -> str:
    """Return "low", "uncertain" or "high" for the given calibrated probability."""
    if is_near_threshold(break_probability, threshold, margin):
        return "uncertain"
    return "high" if float(break_probability) > float(threshold) else "low"


def classification_risk_message(
    break_probability: float,
    horizon_years: float = HORIZON_YEARS,
    threshold: float = CLASSIFICATION_THRESHOLD,
    margin: float = THRESHOLD_MARGIN,
) -> str:
    """Qualitative, non-categorical risk narrative for the classifier's output.

    Rules (bounds inclusive on the uncertain band):
    - risk level "low"       -> "Risque estimé faible de rupture {horizon}."
    - risk level "uncertain" -> "Risque incertain : probabilité proche du
                                  seuil de décision de {threshold}%."
    - risk level "high"      -> "Risque élevé de rupture {horizon}."
    """
    level = classify_risk_level(break_probability, threshold, margin)
    if level == "uncertain":
        return (
            "Risque incertain : probabilité proche du seuil de décision de "
            f"{format_percent_fr(threshold, decimals=0)}."
        )
    horizon_phrase = _horizon_phrase(horizon_years)
    if level == "low":
        return f"Risque estimé faible de rupture {horizon_phrase}."
    return f"Risque élevé de rupture {horizon_phrase}."


def probability_and_threshold_caption(
    break_probability: float,
    horizon_years: float = HORIZON_YEARS,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> str:
    """Always-visible reminder of the calibrated probability and decision threshold."""
    horizon_phrase = _horizon_phrase(horizon_years)
    return (
        f"Probabilité calibrée de rupture {horizon_phrase} : "
        f"{format_percent_fr(break_probability, decimals=1)}.\n"
        f"Seuil de décision : {format_percent_fr(threshold, decimals=0)}."
    )


def regression_indicative_message(
    predicted_years_until_break: float,
    reference_year: int,
) -> str:
    """Non-assertive phrasing for the regression estimate (years + indicative date).

    Never invents a confidence interval: if the project can compute one, pass
    it separately and display it alongside this message — this function only
    formats the point estimate itself, indicatively.
    """
    indicative_year = int(reference_year) + int(round(float(predicted_years_until_break)))
    return (
        "Estimation indicative du délai avant rupture : environ "
        f"{format_number_fr(predicted_years_until_break, decimals=1)} ans.\n"
        f"Date indicative : vers {indicative_year}."
    )


def detect_prediction_divergence(
    break_probability: float,
    predicted_years_until_break: float,
    horizon_years: float = HORIZON_YEARS,
    threshold: float = CLASSIFICATION_THRESHOLD,
    near_horizon_years: float = DIVERGENCE_NEAR_HORIZON_YEARS,
) -> bool:
    """Detect a coherence issue between the classifier and the regressor near the horizon.

    Two symmetric cases, both restricted to the neighbourhood of the horizon
    (so long-range agreements/disagreements are not flagged as "divergence"):
    1. Classifier says positive (>= threshold) but the regressor's estimate
       falls slightly *after* the horizon.
    2. Classifier says negative (< threshold) but the regressor's estimate
       falls slightly *before or at* the horizon.
    """
    predicted_years_until_break = float(predicted_years_until_break)
    horizon_years = float(horizon_years)
    near_horizon = abs(predicted_years_until_break - horizon_years) <= float(near_horizon_years)
    if not near_horizon:
        return False

    classification_positive = float(break_probability) >= float(threshold)
    regression_after_horizon = predicted_years_until_break > horizon_years
    if classification_positive and regression_after_horizon:
        return True

    classification_negative = not classification_positive
    regression_before_or_at_horizon = predicted_years_until_break <= horizon_years
    return classification_negative and regression_before_or_at_horizon


def divergence_warning_message(
    break_probability: float,
    predicted_years_until_break: float,
    horizon_years: float = HORIZON_YEARS,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> str:
    """Warning text describing a divergence detected by `detect_prediction_divergence`.

    Chooses the directional wording (classifier-high/regressor-late vs.
    classifier-low/regressor-early) based on which side of the threshold the
    probability sits on.
    """
    horizon_int = int(round(float(horizon_years)))
    if float(break_probability) >= float(threshold):
        return (
            f"Prédictions proches de l'horizon de {horizon_int} ans : "
            "le classifieur estime un risque légèrement supérieur au seuil, "
            "tandis que le modèle de régression situe la rupture légèrement "
            "après cet horizon. Interprétez ce résultat comme incertain."
        )
    return (
        f"Prédictions proches de l'horizon de {horizon_int} ans : "
        "le classifieur estime un risque légèrement inférieur au seuil, "
        "tandis que le modèle de régression situe la rupture légèrement "
        "avant (ou au niveau de) cet horizon. Interprétez ce résultat comme incertain."
    )


def parse_horizon_years(value, default: int = HORIZON_YEARS) -> int:
    """Robustly coerce a horizon value (int/float/str/None/"—") to an int."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)
