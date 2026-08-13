"""Unit tests for app/evaluation_display.py — pure functions, no Streamlit needed.

Covers the confusion-matrix display rules: only shown for classification
runs, always framed around the fixed 5-year horizon, "—" fallback for
missing counts, and the confusion-matrix image URL builder.
"""

from __future__ import annotations

import pytest

from app.evaluation_display import (
    CONFUSION_MATRIX_EXPLANATION,
    CONFUSION_MATRIX_IMAGE_CAPTION,
    CONFUSION_MATRIX_SECTION_CAPTION,
    HORIZON_YEARS,
    confusion_matrix_image_url,
    extract_confusion_counts,
    format_confusion_count,
    should_display_confusion_matrix,
)


# ---------------------------------------------------------------------------
# should_display_confusion_matrix — classification only
# ---------------------------------------------------------------------------


def test_should_display_confusion_matrix_true_for_classification():
    assert should_display_confusion_matrix("classification") is True


def test_should_display_confusion_matrix_false_for_regression():
    assert should_display_confusion_matrix("regression") is False


@pytest.mark.parametrize("task", [None, "", "unknown", "REGRESSION"])
def test_should_display_confusion_matrix_false_for_anything_else(task):
    assert should_display_confusion_matrix(task) is False


def test_should_display_confusion_matrix_is_case_insensitive():
    assert should_display_confusion_matrix("Classification") is True


# ---------------------------------------------------------------------------
# format_confusion_count — "—" fallback, thousands separator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "0"),
        (1, "1"),
        (1234, "1,234"),
        (12.0, "12"),
        (None, "—"),
    ],
)
def test_format_confusion_count(value, expected):
    assert format_confusion_count(value) == expected


def test_format_confusion_count_rejects_non_numeric():
    assert format_confusion_count("not-a-number") == "—"


# ---------------------------------------------------------------------------
# extract_confusion_counts
# ---------------------------------------------------------------------------


def test_extract_confusion_counts_from_full_evaluation_payload():
    evaluation = {
        "run_id": "abc123",
        "task": "classification",
        "confusion_matrix": {
            "true_negatives": 10,
            "false_positives": 2,
            "false_negatives": 3,
            "true_positives": 7,
        },
    }
    counts = extract_confusion_counts(evaluation)
    assert counts == {
        "true_negatives": 10,
        "false_positives": 2,
        "false_negatives": 3,
        "true_positives": 7,
    }


def test_extract_confusion_counts_defaults_to_none_when_missing():
    assert extract_confusion_counts({"task": "regression", "confusion_matrix": None}) == {
        "true_negatives": None,
        "false_positives": None,
        "false_negatives": None,
        "true_positives": None,
    }
    assert extract_confusion_counts(None) == {
        "true_negatives": None,
        "false_positives": None,
        "false_negatives": None,
        "true_positives": None,
    }


def test_extract_confusion_counts_formatted_for_ui_metrics():
    """End-to-end: the UI must show real TN/FP/FN/TP counts for a
    classification evaluation payload (spec verification item 6)."""
    evaluation = {
        "task": "classification",
        "confusion_matrix": {
            "true_negatives": 120,
            "false_positives": 15,
            "false_negatives": 8,
            "true_positives": 42,
        },
    }
    counts = extract_confusion_counts(evaluation)
    formatted = {key: format_confusion_count(value) for key, value in counts.items()}
    assert formatted == {
        "true_negatives": "120",
        "false_positives": "15",
        "false_negatives": "8",
        "true_positives": "42",
    }


# ---------------------------------------------------------------------------
# confusion_matrix_image_url
# ---------------------------------------------------------------------------


def test_confusion_matrix_image_url_builds_expected_path():
    assert (
        confusion_matrix_image_url("http://api:8000", "run123")
        == "http://api:8000/models/run123/confusion-matrix"
    )


def test_confusion_matrix_image_url_strips_trailing_slash():
    assert (
        confusion_matrix_image_url("http://api:8000/", "run123")
        == "http://api:8000/models/run123/confusion-matrix"
    )


# ---------------------------------------------------------------------------
# Fixed 5-year horizon (spec verification item 7)
# ---------------------------------------------------------------------------


def test_horizon_years_constant_is_fixed_to_five():
    assert HORIZON_YEARS == 5


def test_confusion_matrix_captions_always_mention_five_years():
    assert "5 ans" in CONFUSION_MATRIX_SECTION_CAPTION
    assert "5 prochaines années" in CONFUSION_MATRIX_IMAGE_CAPTION


def test_confusion_matrix_explanation_mentions_false_negatives_and_positives():
    assert "faux négatifs" in CONFUSION_MATRIX_EXPLANATION
    assert "faux positifs" in CONFUSION_MATRIX_EXPLANATION
