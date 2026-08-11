"""Phase 5: API schema + champion loading tests (no httpx TestClient required)."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from api.main import (
    PipeBreakRequest,
    get_model_info,
    health,
    predict,
)


def test_health_reports_tracking_uri():
    body = health()
    assert "tracking_uri" in body
    assert "model_loaded" in body


def test_model_info_exposes_champion_when_loaded():
    payload = get_model_info()
    if payload["status"] != "success":
        pytest.skip("No champion model available in local MLflow store")
    assert "champion" in payload
    assert payload["champion"]["model_type"]
    assert "pr_auc_test" in payload["champion"]
    assert "material" in payload["feature_columns"]


def test_predict_rejects_aircraft_fields():
    with pytest.raises(ValidationError):
        PipeBreakRequest.model_validate(
            {
                "Engine Type": "Piston",
                "HP or lbs thr ea engine": 300,
            }
        )


def test_predict_pipe_features():
    info = get_model_info()
    if info.get("status") != "success":
        pytest.skip("No champion model available in local MLflow store")

    payload = PipeBreakRequest(
        material="CI",
        diameter_mm=150.0,
        install_year=1959.0,
        age_years=46.0,
        prior_break_count=3.0,
        years_since_last_break=0.75,
    )
    response = asyncio.run(predict(payload, threshold=0.5))
    body = response.model_dump()
    assert body["break_within_horizon"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_type"]
    assert body["run_id"]
    assert "pr_auc_test" in body
    if body["break_within_horizon"] == 1:
        # When a regressor champion is loaded, years estimate must be present and >= 0
        years = body.get("estimated_years_until_break")
        if years is not None:
            assert years >= 0.0
    else:
        assert body.get("estimated_years_until_break") is None


def test_predict_zero_prior_requires_null_years():
    info = get_model_info()
    if info.get("status") != "success":
        pytest.skip("No champion model available in local MLflow store")

    with pytest.raises(Exception):
        asyncio.run(
            predict(
                PipeBreakRequest(
                    material="CI",
                    diameter_mm=150.0,
                    install_year=1959.0,
                    age_years=30.0,
                    prior_break_count=0,
                    years_since_last_break=5.0,
                ),
                threshold=0.5,
            )
        )

    ok = asyncio.run(
        predict(
            PipeBreakRequest(
                material="CI",
                diameter_mm=150.0,
                install_year=1959.0,
                age_years=30.0,
                prior_break_count=0,
                years_since_last_break=None,
            ),
            threshold=0.5,
        )
    )
    assert ok.break_within_horizon in (0, 1)
