"""
Streamlit UI for KW water-main break risk classification.

Talks to the FastAPI service (/predict, /model-info, /reload-model).
Payloads follow PipeBreakRequest (Phase 5).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000").rstrip("/")
HORIZON_YEARS = 5

MATERIAL_OPTIONS = [
    "CI",
    "DI",
    "PVC",
    "PVCO",
    "HDPE",
    "AC",
    "CPP",
    "COP",
    "PE",
    "UNKNOWN",
]

st.set_page_config(
    page_title="KW Water Main Break Risk",
    page_icon="💧",
    layout="wide",
)


def api_get(path: str, timeout: float = 10.0):
    return requests.get(f"{API_BASE_URL}{path}", timeout=timeout)


def api_post(path: str, payload: dict | None = None, timeout: float = 30.0):
    return requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)


def fetch_model_info() -> dict:
    try:
        response = api_get("/model-info")
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return {"status": "error", "model_name": "API inaccessible"}


def build_payload(
    material: str,
    diameter_mm: float,
    install_year: float,
    age_years: float,
    prior_break_count: int,
    years_since_last_break: float | None,
) -> dict:
    payload = {
        "material": material,
        "diameter_mm": float(diameter_mm),
        "install_year": float(install_year),
        "age_years": float(age_years),
        "prior_break_count": float(prior_break_count),
        "years_since_last_break": (
            None if prior_break_count == 0 else float(years_since_last_break)
        ),
    }
    return payload


# ---------------------------------------------------------------------------
# Sidebar: maintenance + pipe inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Maintenance")
if st.sidebar.button("Reload champion from MLflow"):
    with st.sidebar.spinner("Reloading model..."):
        try:
            response = api_post("/reload-model")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    st.sidebar.success(data.get("message", "Model reloaded."))
                    st.cache_data.clear()
                else:
                    st.sidebar.error(data.get("message", "Reload failed."))
            else:
                st.sidebar.error(f"API error ({response.status_code})")
        except requests.RequestException as exc:
            st.sidebar.error(f"API unreachable: {exc}")

st.sidebar.markdown("---")
st.sidebar.header("Pipe features")

material = st.sidebar.selectbox("Material", options=MATERIAL_OPTIONS, index=0)
diameter_mm = st.sidebar.number_input(
    "Diameter (mm)",
    min_value=1.0,
    max_value=2000.0,
    value=150.0,
    step=1.0,
)
install_year = st.sidebar.number_input(
    "Install year",
    min_value=1800,
    max_value=2100,
    value=1959,
    step=1,
)
age_years = st.sidebar.number_input(
    "Age at prediction time (years)",
    min_value=0.0,
    max_value=200.0,
    value=46.0,
    step=1.0,
    help="Age of the pipe at snapshot/prediction date t.",
)
prior_break_count = st.sidebar.number_input(
    "Prior break count (before t)",
    min_value=0,
    max_value=50,
    value=3,
    step=1,
)

years_since_last_break = None
if prior_break_count > 0:
    years_since_last_break = st.sidebar.number_input(
        "Years since last break",
        min_value=0.0,
        max_value=100.0,
        value=0.75,
        step=0.25,
        help="Required when the pipe already has prior breaks.",
    )
else:
    st.sidebar.caption("`years_since_last_break` omitted (no prior breaks).")

predict_clicked = st.sidebar.button("Predict break risk", type="primary")

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("Water Main Break Risk — Kitchener-Waterloo")
st.caption(
    f"Binary classification: will this pipe break within the next **{HORIZON_YEARS} years**? "
    "Model selected by the industrial Model Gate (PR-AUC champion, F1 overfit filter)."
)

st.info(
    "**MLOps team:** Mohamed Houari | Peter El-Hadad | "
    "Jaime Alfonso Robledo Villacob | Morad Ait Abdellah"
)

model_info = fetch_model_info()
champion = model_info.get("champion") or {}

st.subheader("Champion model in production")
if model_info.get("status") == "success" and champion:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", str(champion.get("model_type", "—")))
    c2.metric("PR-AUC (test)", f"{float(champion.get('pr_auc_test', 0)):.3f}")
    c3.metric("Overfit F1 gap", f"{float(champion.get('overfit_f1_gap', 0)):.3f}")
    c4.metric("Gate mode", str(champion.get("selection_mode", "—")))
    st.caption(
        f"Run: `{champion.get('run_id', '—')}` · "
        f"F1 train/test: {float(champion.get('f1_train', 0)):.3f} / "
        f"{float(champion.get('f1_test', 0)):.3f}"
    )
    if champion.get("roc_auc_test") is not None:
        st.caption(
            f"ROC-AUC test: {float(champion['roc_auc_test']):.3f} · "
            f"recall@k test: {float(champion.get('recall_at_k_test') or 0):.3f}"
        )
else:
    st.warning(
        "No champion model loaded. Start the API / MLflow stack and use "
        "**Reload champion from MLflow**."
    )

st.markdown("---")
st.subheader("Prediction")

payload = build_payload(
    material=material,
    diameter_mm=diameter_mm,
    install_year=float(install_year),
    age_years=float(age_years),
    prior_break_count=int(prior_break_count),
    years_since_last_break=years_since_last_break,
)

with st.expander("Request payload (PipeBreakRequest)", expanded=False):
    st.json(payload)

if predict_clicked:
    try:
        with st.spinner("Calling FastAPI /predict..."):
            response = api_post("/predict", payload)

        if response.status_code == 200:
            result = response.json()
            label = int(result["break_within_horizon"])
            probability = float(result["probability"])

            if label == 1:
                st.error(
                    f"**Predicted class = 1** — break likely within {HORIZON_YEARS} years "
                    f"(probability = **{probability:.1%}**)."
                )
            else:
                st.success(
                    f"**Predicted class = 0** — no break predicted within {HORIZON_YEARS} years "
                    f"(probability = **{probability:.1%}**)."
                )

            m1, m2, m3 = st.columns(3)
            m1.metric("Class", label)
            m2.metric("P(break)", f"{probability:.3f}")
            m3.metric("Champion PR-AUC", f"{float(result.get('pr_auc_test', 0)):.3f}")

            st.caption(
                f"Served by `{result.get('model_type')}` · "
                f"run `{result.get('run_id')}` · "
                f"selection `{result.get('selection_mode')}` · "
                f"overfit F1 gap `{float(result.get('overfit_f1_gap', 0)):.3f}`"
            )

            with st.expander("Raw API response"):
                st.json(result)
        else:
            st.error(f"API error ({response.status_code}): {response.text}")
    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot reach API at `{API_BASE_URL}`. "
            "Ensure the FastAPI container/service is running."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.markdown("---")
st.write(
    f"MLOps stack: Docker · MLflow · FastAPI · Streamlit · API `{API_BASE_URL}`"
)
st.caption(
    "Limitation: negatives mean “no recorded break in the forward window,” "
    "not confirmed permanently healthy pipes. Pipe length is deferred."
)
