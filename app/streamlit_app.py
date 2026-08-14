"""
Interface Streamlit pour la classification du risque de rupture de conduites d'eau.

Parle au service FastAPI (/predict, /model-info, /reload-model).
Les payloads suivent PipeBreakRequest (Phase 5).

Corrections appliquées :
- age_years n'est plus saisi manuellement : calculé automatiquement à partir
  de install_year et de la date du jour (évite les incohérences du type
  install_year=1959 / age_years=46 qui ne correspondent qu'à l'année 2005).
- years_since_last_break est dérivé d'une date de dernière rupture choisie
  par l'utilisateur, plutôt que d'un nombre d'années saisi à la main.
- threshold déplacé dans sa propre section "Paramètres de décision".
- horizon_years du modèle champion affiché explicitement : dans le message
  de rechargement (sidebar) et dans les métriques du champion (page principale),
  pour que l'utilisateur sache toujours pour quel horizon le modèle chargé
  a été entraîné.
"""

from __future__ import annotations

import os
from datetime import date

import requests
import streamlit as st

try:
    # Exécution Streamlit réelle : le dossier app/ (contenant streamlit_app.py)
    # est ajouté à sys.path, donc les imports "à plat" fonctionnent.
    from risk_messaging import (
        CLASSIFICATION_THRESHOLD,
        HORIZON_YEARS,
        REGRESSION_UNCERTAINTY_NOTICE,
        classification_risk_message,
        classify_risk_level,
        detect_prediction_divergence,
        divergence_warning_message,
        parse_horizon_years,
        probability_and_threshold_caption,
        regression_indicative_message,
    )
except ModuleNotFoundError:
    # Exécution hors Streamlit (ex: pytest depuis la racine du repo) : le
    # package namespace "app" est importable directement.
    from app.risk_messaging import (
        CLASSIFICATION_THRESHOLD,
        HORIZON_YEARS,
        REGRESSION_UNCERTAINTY_NOTICE,
        classification_risk_message,
        classify_risk_level,
        detect_prediction_divergence,
        divergence_warning_message,
        parse_horizon_years,
        probability_and_threshold_caption,
        regression_indicative_message,
    )


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

TODAY = date.today()

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


def fetch_regressor_info() -> dict:
    try:
        response = api_get("/regressor-info")
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return {"status": "error", "model_name": "API inaccessible"}


def compute_age_years(install_year: int, reference_date: date = TODAY) -> float:
    """Âge de la conduite à la date de référence, dérivé de install_year.

    Garantit la cohérence entre install_year et age_years : il n'existe
    plus deux champs indépendants pouvant se contredire.
    """
    return float(reference_date.year - int(install_year))


def compute_years_since_last_break(last_break_date: date, reference_date: date = TODAY) -> float:
    """Années écoulées depuis la dernière rupture, dérivées d'une date exacte
    plutôt que d'un nombre saisi manuellement (évite les erreurs de calcul)."""
    delta_days = (reference_date - last_break_date).days
    return round(max(delta_days, 0) / 365.25, 2)


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


def format_horizon(horizon_value) -> str:
    """Formate horizon_years de façon lisible, quel que soit le type reçu
    (int, float, str ou "—" si absent)."""
    try:
        horizon_int = int(float(horizon_value))
        suffix = "an" if horizon_int == 1 else "ans"
        return f"{horizon_int} {suffix}"
    except (TypeError, ValueError):
        return "—"


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
                    reloaded_champion = data.get("champion") or {}
                    reloaded_horizon = format_horizon(reloaded_champion.get("horizon_years"))
                    # Affiche un message de succès dans la sidebar, avec l'horizon des deux modèles classification et regression rechargé.
                    classifier = data.get("champion") or {}
                    regressor = data.get("regressor_champion") or {}

                    message = (
                        "✅ Champion de classification rechargé.\n\n"
                        f"Modèle : **{classifier.get('model_type', '—')}**\n\n"
                        f"Horizon : **{format_horizon(classifier.get('horizon_years'))}**"
                    )

                    if regressor:
                        message += (
                            "\n\n✅ Champion de régression rechargé.\n\n"
                            f"Modèle : **{regressor.get('model_type', '—')}**"
                        )
                    else:
                        message += "\n\nℹ️ Aucun champion de régression disponible."

                    st.sidebar.success(message)
                    ###
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
    "Année d'installation",
    min_value=1800,
    max_value=TODAY.year,
    value=1959,
    step=1,
    help="L'âge de la conduite est calculé automatiquement à partir de cette valeur.",
)

# --- Âge calculé automatiquement — plus jamais saisi manuellement ---
age_years = compute_age_years(int(install_year))
st.sidebar.metric("Âge de la conduite (calculé)", f"{age_years:.0f} ans")
st.sidebar.caption(f"Calcul : {TODAY.year} − {int(install_year)} = {age_years:.0f} ans")

prior_break_count = st.sidebar.number_input(
    "Nombre de ruptures antérieures (avant t)",
    min_value=0,
    max_value=50,
    value=3,
    step=1,
)

years_since_last_break = None
if prior_break_count > 0:
    default_last_break = date(max(TODAY.year - 1, int(install_year)), 1, 1)
    last_break_date = st.sidebar.date_input(
        "Date de la dernière rupture",
        value=default_last_break,
        min_value=date(int(install_year), 1, 1),
        max_value=TODAY,
        help="La durée écoulée depuis cette date est calculée automatiquement.",
    )
    years_since_last_break = compute_years_since_last_break(last_break_date)
    st.sidebar.caption(f"→ {years_since_last_break:.2f} années depuis la dernière rupture")
else:
    st.sidebar.caption("`years_since_last_break` omis (aucune rupture antérieure).")

st.sidebar.markdown("---")
st.sidebar.header("Paramètres de décision")

threshold = st.sidebar.slider(
    "Seuil de décision",
    min_value=0.0,
    max_value=1.0,
    value=CLASSIFICATION_THRESHOLD,
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
horizon_years = champion.get("horizon_years", HORIZON_YEARS)
horizon_label = format_horizon(horizon_years)
effective_horizon_years = parse_horizon_years(horizon_years, default=HORIZON_YEARS)

regressor_info = fetch_regressor_info()
regressor_champion = regressor_info.get("champion") or {}

st.caption(
    f"Classification binaire : cette conduite va-t-elle rompre dans les **{horizon_label}** à venir ? "
    "Le modèle champion est sélectionné par la Model Gate industrielle (PR-AUC + filtre de surapprentissage F1)."
)

st.info(
    "**Équipe MLOps :** Mohamed Houari | Peter El-Hadad | Jaime Alfonso Robledo Villacob | Morad Ait Abdellah"
)


st.subheader("Modèle champion en production")
if model_info.get("status") == "success" and champion:
    st.warning(
        f"⏱️ **Ce modèle a été entraîné pour prédire un risque de rupture dans les {horizon_label} à venir.**",
        icon="⏱️",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modèle", str(champion.get("model_type", "—")))
    c2.metric("Horizon", horizon_label)
    c3.metric("PR-AUC (test)", f"{float(champion.get('pr_auc_test', 0)):.3f}")
    c4.metric("Écart F1 overfit", f"{float(champion.get('overfit_f1_gap', 0)):.3f}")

    st.caption(
        f"Run : `{champion.get('run_id', '—')}` · "
        f"F1 train/test : {float(champion.get('f1_train', 0)):.3f} / {float(champion.get('f1_test', 0)):.3f} · "
        f"Mode de sélection : `{champion.get('selection_mode', '—')}`"
    )
    if champion.get("roc_auc_test") is not None:
        st.caption(
            f"ROC-AUC test : {float(champion['roc_auc_test']):.3f} · "
            f"recall@k test : {float(champion.get('recall_at_k_test') or 0):.3f}"
        )

    with st.expander("📐 Comment ces scores sont-ils calculés ?", expanded=False):
        st.markdown("### PR-AUC (test) — métrique principale de sélection")
        st.latex(r"\text{PR-AUC} = \int_0^1 \text{Précision}(r)\, dr")
        st.caption(
            "Aire sous la courbe Précision-Rappel. Mesure la capacité du modèle à bien "
            "classer la classe rare (rupture) sans se laisser tromper par le déséquilibre "
            "des classes. Plus proche de 1 = meilleur."
        )

        st.markdown("### F1-score (train / test)")
        st.latex(r"F_1 = 2 \times \frac{\text{Précision} \times \text{Rappel}}{\text{Précision} + \text{Rappel}}")
        st.caption(
            "Moyenne harmonique entre précision et rappel, calculée séparément sur "
            "l'ensemble d'entraînement et sur l'ensemble de test."
        )

        st.markdown("### Écart de surapprentissage F1 (overfit_f1_gap)")
        st.latex(r"\text{overfit\_f1\_gap} = \max(0,\ F_{1,\text{train}} - F_{1,\text{test}})")
        st.caption(
            "Un écart proche de 0 est souhaitable, mais attention : un écart nul avec "
            "F1_train et F1_test tous deux très bas signifie que le modèle échoue "
            "uniformément, pas qu'il est bon."
        )

        st.markdown("### ROC-AUC (test)")
        st.latex(r"\text{ROC-AUC} = P(\text{score}(x^+) > \text{score}(x^-))")
        st.caption(
            "Probabilité que le modèle classe une conduite ayant réellement rompu "
            "avec un score de risque plus élevé qu'une conduite n'ayant pas rompu."
        )

        st.markdown("### Recall@K (recall_at_k_test)")
        st.latex(r"\text{Recall@K} = \frac{\text{Vraies ruptures dans le top K\%}}{\text{Total des vraies ruptures}}")
        st.caption(
            "Si on ne pouvait inspecter que les K % de conduites jugées les plus à "
            "risque, quelle proportion des vraies ruptures aurait-on identifiée ?"
        )

        st.markdown("### Sélection du champion (Model Gate)")
        st.markdown(
            "1. Ne garder que les modèles ayant un `overfit_f1_gap` sous le seuil autorisé.\n"
            "2. Parmi les survivants, choisir celui avec le **PR-AUC test le plus élevé**.\n"
            "3. Si aucun modèle ne passe le filtre, prendre celui avec le plus petit "
            "`overfit_f1_gap` en mode **fallback**."
        )
else:
    st.warning(
        "Aucun modèle champion chargé. Démarre la stack API / MLflow puis utilise "
        "**Recharger le champion depuis MLflow**."
    )

st.markdown("##### Modèle champion de régression (years_until_break)")
if regressor_info.get("status") == "success" and regressor_champion:
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Modèle", str(regressor_champion.get("model_type", "—")))
    r2.metric("RMSE (test)", f"{float(regressor_champion.get('rmse_test', 0)):.3f}")
    r3.metric("MAE (test)", f"{float(regressor_champion.get('mae_test', 0)):.3f}")
    r4.metric("R² (test)", f"{float(regressor_champion.get('r2_test', 0)):.3f}")
    st.caption(f"Run : `{regressor_champion.get('run_id', '—')}`")
else:
    st.caption(
        "Aucun modèle de régression champion disponible pour le moment "
        "(entraîne-en un avec `--task regression`)."
    )


st.markdown("---")
st.subheader("Prédiction")

payload = build_payload(
    material=material,
    diameter_mm=diameter_mm,
    install_year=float(install_year),
    age_years=age_years,
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

            # Qualitative risk narrative, framed around the centralized
            # CLASSIFICATION_THRESHOLD (not the user-adjustable slider): a
            # calibrated probability close to the standard 50% decision
            # boundary is never stated as a categorical certainty.
            risk_level = classify_risk_level(probability)
            risk_message = classification_risk_message(probability, horizon_years=effective_horizon_years)
            if risk_level == "uncertain":
                st.warning(f"⚠️ {risk_message}")
            elif risk_level == "high":
                st.error(f"🔺 {risk_message}")
            else:
                st.success(f"✅ {risk_message}")

            # Toujours afficher la probabilité calibrée et le seuil, quel que
            # soit le niveau de risque affiché ci-dessus.
            st.caption(probability_and_threshold_caption(probability, horizon_years=effective_horizon_years))
            st.caption(
                f"Décision brute au seuil sélectionné (**{threshold_used:.2f}**) : "
                f"classe prédite = **{label}**."
            )

            estimated_years = result.get("estimated_years_until_break")
            if estimated_years is not None:
                estimated_years = float(estimated_years)
                st.info(
                    "⏳ " + regression_indicative_message(estimated_years, reference_year=date.today().year)
                )
                st.caption(f"ℹ️ {REGRESSION_UNCERTAINTY_NOTICE}")

                if detect_prediction_divergence(probability, estimated_years, horizon_years=effective_horizon_years):
                    st.warning(
                        divergence_warning_message(probability, estimated_years, horizon_years=effective_horizon_years)
                    )

            m1, m2, m3 = st.columns(3)
            m1.metric("Classe (seuil sélectionné)", label)
            m2.metric("P(rupture) calibrée", f"{probability:.3f}")
            m3.metric("PR-AUC champion", f"{float(result.get('pr_auc_test', 0)):.3f}")

            st.caption(
                f"Servi par `{result.get('model_type')}` · "
                f"run `{result.get('run_id')}` · "
                f"sélection `{result.get('selection_mode')}` · "
                f"écart F1 overfit `{float(result.get('overfit_f1_gap', 0)):.3f}`"
            )

            with st.expander("Réponse brute de l'API"):
                st.json(result)
        else:
            st.error(f"Erreur API ({response.status_code}) : {response.text}")
    except requests.exceptions.ConnectionError:
        st.error(
            f"Impossible de joindre l'API à `{API_BASE_URL}`. "
            "Vérifie que le conteneur / service FastAPI est bien démarré."
        )
    except Exception as exc:
        st.error(f"Échec de la prédiction : {exc}")


st.markdown("---")
st.write(
    f"Stack MLOps : Docker · MLflow · FastAPI · Streamlit · API `{API_BASE_URL}`"
)
st.caption(
    "Limite : les prédictions négatives signifient 'aucune rupture enregistrée dans l'horizon forward', "
    "pas une garantie absolue d'absence de rupture. La longueur de conduite est différée."
)