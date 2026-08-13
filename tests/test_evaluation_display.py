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
    ConfusionMatrixFetchError,
    HORIZON_YEARS,
    confusion_matrix_image_url,
    extract_confusion_counts,
    fetch_confusion_matrix_image,
    format_confusion_count,
    should_display_confusion_matrix,
)


class _FakeResponse:
    """Minimal stand-in for requests.Response, only what fetch_confusion_matrix_image reads."""

    def __init__(self, status_code: int, content: bytes = b"", content_type: str = "image/png"):
        self.status_code = status_code
        self.content = content
        self.headers = {"Content-Type": content_type}


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


# ---------------------------------------------------------------------------
# fetch_confusion_matrix_image — server-side fetch (never a bare API URL,
# which only resolves inside the Docker network, never in the user's browser)
# ---------------------------------------------------------------------------


def test_fetch_confusion_matrix_image_returns_bytes_for_200_png(monkeypatch):
    png_bytes = b"\x89PNG\r\n\x1a\nfake-bytes"
    captured = {}

    def fake_get(url, timeout=None):
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeResponse(200, content=png_bytes, content_type="image/png")

    monkeypatch.setattr("app.evaluation_display.requests.get", fake_get)

    result = fetch_confusion_matrix_image("http://api:8000", "run123")

    assert result == png_bytes
    assert isinstance(result, bytes)
    assert captured["url"] == "http://api:8000/models/run123/confusion-matrix"
    assert captured["timeout"] == 30


def test_fetch_confusion_matrix_image_accepts_custom_timeout(monkeypatch):
    captured = {}

    def fake_get(url, timeout=None):
        captured["timeout"] = timeout
        return _FakeResponse(200, content=b"\x89PNG", content_type="image/png")

    monkeypatch.setattr("app.evaluation_display.requests.get", fake_get)

    fetch_confusion_matrix_image("http://api:8000", "run123", timeout=5)

    assert captured["timeout"] == 5


def test_fetch_confusion_matrix_image_returns_none_for_404(monkeypatch):
    monkeypatch.setattr(
        "app.evaluation_display.requests.get",
        lambda url, timeout=None: _FakeResponse(404),
    )

    result = fetch_confusion_matrix_image("http://api:8000", "unknown-run")

    assert result is None


@pytest.mark.parametrize("status_code", [500, 503, 400])
def test_fetch_confusion_matrix_image_raises_for_other_error_statuses(monkeypatch, status_code):
    monkeypatch.setattr(
        "app.evaluation_display.requests.get",
        lambda url, timeout=None: _FakeResponse(status_code),
    )

    with pytest.raises(ConfusionMatrixFetchError):
        fetch_confusion_matrix_image("http://api:8000", "run123")


def test_fetch_confusion_matrix_image_raises_for_unexpected_content_type(monkeypatch):
    monkeypatch.setattr(
        "app.evaluation_display.requests.get",
        lambda url, timeout=None: _FakeResponse(200, content=b"{}", content_type="application/json"),
    )

    with pytest.raises(ConfusionMatrixFetchError):
        fetch_confusion_matrix_image("http://api:8000", "run123")


# ---------------------------------------------------------------------------
# Streamlit pages must pass image *bytes* to st.image(), never a bare
# http://api:8000/... URL (unresolvable from the end user's browser).
# ---------------------------------------------------------------------------


def test_entrainement_page_passes_image_bytes_not_url_to_st_image(monkeypatch):
    import importlib

    page = importlib.import_module("app.pages.1_Entrainement")

    png_bytes = b"\x89PNG\r\n\x1a\nfake-bytes"
    captured_image_calls = []

    monkeypatch.setattr(
        page,
        "api_get",
        lambda path, timeout=10.0: _EvalResponse(
            200, {"task": "classification", "artifact_available": True, "confusion_matrix": {
                "true_negatives": 1, "false_positives": 2, "false_negatives": 3, "true_positives": 4,
            }},
        ),
    )
    monkeypatch.setattr(page, "fetch_confusion_matrix_image", lambda api_base_url, run_id: png_bytes)
    monkeypatch.setattr(page.st, "image", lambda image, **kwargs: captured_image_calls.append(image))
    monkeypatch.setattr(page.st, "columns", lambda n: tuple(_NoOpMetricColumn() for _ in range(n)))
    monkeypatch.setattr(page.st, "subheader", lambda *a, **kw: None)
    monkeypatch.setattr(page.st, "caption", lambda *a, **kw: None)

    page.render_confusion_matrix_section("run123")

    assert len(captured_image_calls) == 1
    passed_image = captured_image_calls[0]
    assert isinstance(passed_image, bytes)
    assert passed_image == png_bytes
    assert not isinstance(passed_image, str)
    assert "http://" not in str(passed_image)[:4]  # never a bare URL string


class _EvalResponse:
    """Minimal stand-in for requests.Response used to fake GET .../evaluation."""

    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _NoOpMetricColumn:
    def metric(self, *args, **kwargs) -> None:
        return None
