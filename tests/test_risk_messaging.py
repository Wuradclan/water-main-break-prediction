"""Unit tests for app/risk_messaging.py — pure functions, no Streamlit needed.

Covers the coherence requirements between the classifier (P(break <= horizon))
and the regressor (years_until_break estimate): uncertainty zone around the
decision threshold, indicative regression wording, and divergence detection
near the horizon.
"""

from __future__ import annotations

import pytest

from app.risk_messaging import (
    CLASSIFICATION_THRESHOLD,
    HORIZON_YEARS,
    REGRESSION_UNCERTAINTY_NOTICE,
    THRESHOLD_MARGIN,
    classification_risk_message,
    classify_risk_level,
    detect_prediction_divergence,
    divergence_warning_message,
    is_near_threshold,
    parse_horizon_years,
    probability_and_threshold_caption,
    regression_indicative_message,
)


def test_centralized_constants_match_spec():
    assert HORIZON_YEARS == 5
    assert CLASSIFICATION_THRESHOLD == 0.50
    assert THRESHOLD_MARGIN == 0.05


# ---------------------------------------------------------------------------
# 1-5: probability -> risk level (bounds of the uncertainty zone are inclusive)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "probability,expected_level,expected_keyword",
    [
        (0.449, "low", "faible"),
        (0.450, "uncertain", "incertain"),
        (0.549, "uncertain", "incertain"),
        (0.550, "uncertain", "incertain"),
        (0.551, "high", "élevé"),
    ],
)
def test_risk_level_and_message_boundaries(probability, expected_level, expected_keyword):
    assert classify_risk_level(probability) == expected_level
    message = classification_risk_message(probability)
    assert expected_keyword in message


def test_is_near_threshold_matches_classify_risk_level():
    for probability in (0.449, 0.450, 0.549, 0.550, 0.551):
        assert is_near_threshold(probability) == (classify_risk_level(probability) == "uncertain")


def test_low_risk_message_mentions_horizon_and_not_categorical_break():
    message = classification_risk_message(0.449)
    assert "5 ans" in message
    assert "rupture probable" not in message.lower()


def test_high_risk_message_does_not_hide_uncertainty_language_elsewhere():
    # The high-risk message itself may be assertive, but the *uncertain* zone
    # must never be collapsed into a categorical statement.
    message = classification_risk_message(0.551)
    assert "élevé" in message


def test_probability_and_threshold_always_shown():
    caption = probability_and_threshold_caption(0.549, horizon_years=5)
    assert "54,9%" in caption
    assert "50%" in caption
    assert "Probabilité calibrée" in caption
    assert "Seuil de décision" in caption


# ---------------------------------------------------------------------------
# 6-8: divergence detection near the horizon
# ---------------------------------------------------------------------------
def test_divergence_flagged_when_classifier_positive_and_regression_slightly_after_horizon():
    assert detect_prediction_divergence(0.549, 5.1) is True
    message = divergence_warning_message(0.549, 5.1)
    assert "horizon" in message.lower()
    assert "incertain" in message.lower()


def test_no_divergence_when_both_predictions_strongly_agree_on_high_risk():
    assert detect_prediction_divergence(0.80, 2.0) is False


def test_no_divergence_when_both_predictions_strongly_agree_on_low_risk():
    assert detect_prediction_divergence(0.20, 9.0) is False


def test_symmetric_divergence_when_classifier_negative_and_regression_slightly_before_horizon():
    # Classifier says "no break within horizon" but the regressor estimates
    # a break just before/at the horizon -> also a divergence.
    assert detect_prediction_divergence(0.40, 4.6) is True
    message = divergence_warning_message(0.40, 4.6)
    assert "incertain" in message.lower()


def test_no_divergence_far_from_horizon_even_if_probability_is_borderline():
    # Near-threshold probability but regression estimate far from the horizon:
    # not a "near horizon" divergence case.
    assert detect_prediction_divergence(0.549, 20.0) is False


# ---------------------------------------------------------------------------
# 9: regression wording is always indicative, never a categorical guarantee
# ---------------------------------------------------------------------------
def test_regression_message_is_indicative_and_mentions_uncertainty():
    message = regression_indicative_message(5.1, reference_year=2026)
    assert "indicative" in message.lower()
    assert "5,1" in message
    assert "2031" in message
    assert "environ" in message.lower()


def test_regression_uncertainty_notice_mentions_incertitude_and_distinct_model():
    assert "incertitude" in REGRESSION_UNCERTAINTY_NOTICE.lower()
    assert "distinct" in REGRESSION_UNCERTAINTY_NOTICE.lower()


def test_regression_message_never_states_an_exact_or_guaranteed_date():
    message = regression_indicative_message(5.1, reference_year=2026)
    assert "exactement" not in message.lower()
    assert "garanti" not in message.lower()


# ---------------------------------------------------------------------------
# parse_horizon_years: robust coercion used to feed the real champion horizon
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw_value,expected",
    [
        (5, 5),
        (5.0, 5),
        ("5", 5),
        (None, HORIZON_YEARS),
        ("—", HORIZON_YEARS),
    ],
)
def test_parse_horizon_years_is_robust(raw_value, expected):
    assert parse_horizon_years(raw_value) == expected
