"""
Interface Streamlit pour la classification du risque de rupture de conduites d'eau.

Parle au service FastAPI (/predict, /model-info, /reload-model).
Les payloads suivent PipeBreakRequest (Phase 5).
"""

from __future__ import annotations

import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000").rstrip("/")

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
    page_title="Risque de rupture de conduite",
    page_icon="💧",
    layout="wide",
)


def api_get(path: str, timeout: float = 10.0):
    return requests.get(f"{API_BASE_URL}{path}", timeout=timeout)


def api_post(path: str, payload: dict | None = None, params: dict | None = None, timeout: float = 30.0):
    return requests.post(f"{API_BASE_URL}{path}", json=payload, params=params, timeout=timeout)


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
    return {
        "material": material,
        "diameter_mm": float(diameter_mm),
        "install_year": float(install_year),
        "age_years": float(age_years),
        "prior_break_count": float(prior_break_count),
        "years_since_last_break": None if prior_break_count == 0 else float(years_since_last_break),
    }


# ---------------------------------------------------------------------------
# Sidebar: maintenance + pipe inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Maintenance")
if st.sidebar.button("Recharger le champion depuis MLflow"):
    with st.sidebar.spinner("Rechargement du modèle..."):
        try:
            response = api_post("/reload-model")
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    st.sidebar.success(data.get("message", "Modèle rechargé."))
                    st.cache_data.clear()
                else:
                    st.sidebar.error(data.get("message", "Échec du rechargement."))
            else:
                st.sidebar.error(f"Erreur API ({response.status_code})")
        except requests.RequestException as exc:
            st.sidebar.error(f"API inaccessible : {exc}")


st.sidebar.markdown("---")
st.sidebar.header("Caractéristiques de la conduite")

material = st.sidebar.selectbox("Matériau", options=MATERIAL_OPTIONS, index=0)
diameter_mm = st.sidebar.number_input(
    "Diamètre (mm)",
    min_value=1.0,
    max_value=2000.0,
    value=150.0,
    step=1.0,
)
install_year = st.sidebar.number_input(
    "Année d’installation",
    min_value=1800,
    max_value=2100,
    value=1959,
    step=1,
)
age_years = st.sidebar.number_input(
    "Âge à la date de prédiction (années)",
    min_value=0.0,
    max_value=200.0,
    value=46.0,
    step=1.0,
    help="Âge de la conduite à la date d’observation/prédiction t.",
)
prior_break_count = st.sidebar.number_input(
    "Nombre de ruptures antérieures (avant t)",
    min_value=0,
    max_value=50,
    value=3,
    step=1,
)

years_since_last_break = None
if prior_break_count > 0:
    years_since_last_break = st.sidebar.number_input(
        "Années depuis la dernière rupture",
        min_value=0.0,
        max_value=100.0,
        value=0.75,
        step=0.25,
        help="Obligatoire lorsque la conduite a déjà connu au moins une rupture.",
    )
else:
    st.sidebar.caption("`years_since_last_break` omis (aucune rupture antérieure).")

threshold = st.sidebar.slider(
    "Seuil de décision",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
    help="La classe 1 est prédite si la probabilité est supérieure ou égale à ce seuil.",
)

predict_clicked = st.sidebar.button("Prédire le risque de rupture", type="primary")


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("Risque de rupture de conduite — Kitchener-Waterloo")

model_info = fetch_model_info()
champion = model_info.get("champion") or {}
horizon_years = champion.get("horizon_years", 5)

st.caption(
    f"Classification binaire : cette conduite va-t-elle rompre dans les **{horizon_years} prochaines années** ? "
    "Le modèle champion est sélectionné par la Model Gate industrielle (PR-AUC + filtre de surapprentissage F1)."
)

st.info(
    "**Équipe MLOps :** Mohamed Houari | Peter El-Hadad | Jaime Alfonso Robledo Villacob | Morad Ait Abdellah"
)


st.subheader("Modèle champion en production")
if model_info.get("status") == "success" and champion:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modèle", str(champion.get("model_type", "—")))
    c2.metric("PR-AUC (test)", f"{float(champion.get('pr_auc_test', 0)):.3f}")
    c3.metric("Écart F1 overfit", f"{float(champion.get('overfit_f1_gap', 0)):.3f}")
    c4.metric("Mode de sélection", str(champion.get("selection_mode", "—")))
    st.caption(
        f"Run : `{champion.get('run_id', '—')}` · "
        f"F1 train/test : {float(champion.get('f1_train', 0)):.3f} / {float(champion.get('f1_test', 0)):.3f}"
    )
    if champion.get("roc_auc_test") is not None:
        st.caption(
            f"ROC-AUC test : {float(champion['roc_auc_test']):.3f} · "
            f"recall@k test : {float(champion.get('recall_at_k_test') or 0):.3f}"
        )
else:
    st.warning(
        "Aucun modèle champion chargé. Démarre la stack API / MLflow puis utilise "
        "**Recharger le champion depuis MLflow**."
    )


st.markdown("---")
st.subheader("Prédiction")

payload = build_payload(
    material=material,
    diameter_mm=diameter_mm,
    install_year=float(install_year),
    age_years=float(age_years),
    prior_break_count=int(prior_break_count),
    years_since_last_break=years_since_last_break,
)

with st.expander("Payload de la requête (PipeBreakRequest)", expanded=False):
    st.json(payload)

if predict_clicked:
    try:
        with st.spinner("Appel de /predict sur FastAPI..."):
            response = api_post("/predict", payload, params={"threshold": threshold})

        if response.status_code == 200:
            result = response.json()
            label = int(result["break_within_horizon"])
            probability = float(result["probability"])
            threshold_used = float(result.get("threshold", threshold))

            if label == 1:
                st.error(
                    f"**Classe prédite = 1** — rupture probable dans {horizon_years} ans "
                    f"(probabilité = **{probability:.1%}**, seuil = **{threshold_used:.2f}**)."
                )
            else:
                st.success(
                    f"**Classe prédite = 0** — aucune rupture prédite dans {horizon_years} ans "
                    f"(probabilité = **{probability:.1%}**, seuil = **{threshold_used:.2f}**)."
                )

            m1, m2, m3 = st.columns(3)
            m1.metric("Classe", label)
            m2.metric("P(rupture)", f"{probability:.3f}")
            m3.metric("PR-AUC champion", f"{float(result.get('pr_auc_test', 0)):.3f}")

            st.caption(
                f"Servi par `{result.get('model_type')}` · "
                f"run `{result.get('run_id')}` · "
                f"sélection `{result.get('selection_mode')}` · "
                f"écart F1 overfit `{float(result.get('overfit_f1_gap', 0)):.3f}`"
            )

            with st.expander("Réponse brute de l’API"):
                st.json(result)
        else:
            st.error(f"Erreur API ({response.status_code}) : {response.text}")
    except requests.exceptions.ConnectionError:
        st.error(
            f"Impossible de joindre l’API à `{API_BASE_URL}`. "
            "Vérifie que le conteneur / service FastAPI est bien démarré."
        )
    except Exception as exc:
        st.error(f"Échec de la prédiction : {exc}")


st.markdown("---")
st.write(
    f"Stack MLOps : Docker · MLflow · FastAPI · Streamlit · API `{API_BASE_URL}`"
)
st.caption(
    "Limite : les prédictions négatives signifient ‘aucune rupture enregistrée dans l’horizon forward’, "
    "pas une garantie absolue d’absence de rupture. La longueur de conduite est différée."
)