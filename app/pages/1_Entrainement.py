"""
Page Streamlit pour lancer les entraînements du projet MLOps.

Emplacement :
    app/pages/1_Entrainement.py

Architecture :
- Streamlit appelle POST /start-training sur FastAPI (réponse immédiate + job_id)
- FastAPI lance docker exec dans un thread daemon et écrit les logs dans un fichier
- Streamlit interroge GET /training-status/{job_id} toutes les ~3 s via st.fragment
"""

from __future__ import annotations

import os
import re

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000").rstrip("/")

MODEL_CHOICES = {
    "Régression logistique": "logistic",
    "Ridge": "ridge",
    "Lasso": "lasso",
    "Random Forest": "random_forest",
    "Extra Trees": "extra_trees",
    "XGBoost": "xgboost",
    "K-Nearest Neighbors": "knn",
    "SVC": "svc",
    "MLP — Réseau de neurones": "mlp",
    "Stacking": "stacking",
    "H2O AutoML": "h2o",
}

OPTUNA_PROGRESS_RE = re.compile(r"\[Optuna\] Essai (\d+)/(\d+)")


def api_get(path: str, timeout: float = 10.0) -> requests.Response:
    return requests.get(f"{API_BASE_URL}{path}", timeout=timeout)


def api_post(path: str, params: dict | None = None, timeout: float = 10.0) -> requests.Response:
    return requests.post(f"{API_BASE_URL}{path}", params=params, timeout=timeout)


def job_is_active() -> bool:
    """True si un job_id est suivi et n'est pas encore terminé côté UI."""
    job_id = st.session_state.get("job_id")
    if not job_id:
        return False
    status = st.session_state.get("job_status")
    return status in {None, "pending", "running"}


def render_champion_metrics(result: dict) -> None:
    """Affiche les métriques du champion renvoyées par l'API."""
    st.subheader("Résultat — modèle champion")
    c1, c2, c3 = st.columns(3)
    c1.metric("PR-AUC test", f"{float(result.get('pr_auc_test', 0)):.4f}")
    c2.metric(
        "F1 train / test",
        f"{float(result.get('f1_train', 0)):.4f} / {float(result.get('f1_test', 0)):.4f}",
    )
    c3.metric("Écart F1 overfit", f"{float(result.get('overfit_f1_gap', 0)):.4f}")

    c4, c5, c6 = st.columns(3)
    roc = result.get("roc_auc_test")
    c4.metric("ROC-AUC test", f"{float(roc):.4f}" if roc is not None else "—")
    recall_k = result.get("recall_at_k_test")
    c5.metric("recall@K test", f"{float(recall_k):.4f}" if recall_k is not None else "—")
    c6.metric("Mode de sélection", str(result.get("selection_mode", "—")))

    st.caption(
        f"Modèle : `{result.get('model_type', '—')}` · "
        f"run `{result.get('run_id', '—')}` · "
        f"horizon {result.get('horizon_years', '—')} ans"
    )


st.set_page_config(
    page_title="Entraînement — Bris d'aqueduc",
    page_icon="🛠️",
    layout="wide",
)

st.title("🛠️ Entraînement des modèles")
st.caption(
    "Cette interface déclenche l'entraînement via l'API FastAPI "
    f"(`{API_BASE_URL}`), qui pilote le conteneur `bris-aqueduc-trainer`."
)

st.warning(
    "Un seul entraînement à la fois est autorisé. "
    "Les entraînements Optuna et H2O AutoML peuvent prendre plusieurs minutes. "
    "L'interface reste réactive grâce au polling asynchrone.",
    icon="⚠️",
)

if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "job_status" not in st.session_state:
    st.session_state.job_status = None
if "last_tune" not in st.session_state:
    st.session_state.last_tune = False

left_column, right_column = st.columns(2)

with left_column:
    selected_label = st.selectbox(
        "Algorithme à entraîner",
        options=list(MODEL_CHOICES.keys()),
    )
    model_type = MODEL_CHOICES[selected_label]

    horizon_years = st.selectbox(
        "Horizon de prédiction",
        options=[1, 2, 5],
        index=2,
        format_func=lambda value: f"{value} an" if value == 1 else f"{value} ans",
    )

with right_column:
    optuna_supported = model_type != "h2o"

    if optuna_supported:
        tune = st.checkbox(
            "Activer l'optimisation Optuna",
            value=(model_type == "xgboost"),
            help=(
                "Optuna recherche les meilleurs hyperparamètres en maximisant "
                "le PR-AUC tout en pénalisant le surapprentissage F1."
            ),
        )
        if tune:
            n_trials = st.slider(
                "Nombre d'essais Optuna",
                min_value=5,
                max_value=50,
                value=15,
                step=1,
            )
        else:
            n_trials = 15
    else:
        tune = False
        n_trials = 15
        st.info(
            "H2O AutoML gère automatiquement sa recherche de modèles. "
            "L'option Optuna est donc désactivée."
        )

# Aperçu de la commande qui sera exécutée côté API
preview_parts = [
    "docker exec bris-aqueduc-trainer python -m src.train",
    f"--model_type {model_type}",
    f"--horizon_years {horizon_years}",
]
if tune:
    preview_parts.extend(["--tune", f"--n_trials {n_trials}"])

st.subheader("Commande générée")
with st.expander("Afficher la commande Docker", expanded=True):
    st.code(" ".join(preview_parts), language="bash")

launch_disabled = job_is_active()
launch_button = st.button(
    "🚀 Lancer l'entraînement",
    type="primary",
    disabled=launch_disabled,
)

if launch_button:
    try:
        response = api_post(
            "/start-training",
            params={
                "model_type": model_type,
                "tune": tune,
                "n_trials": n_trials,
                "horizon_years": horizon_years,
            },
            timeout=10,
        )
        if response.status_code == 200:
            data = response.json()
            st.session_state.job_id = data["job_id"]
            st.session_state.job_status = data.get("status", "pending")
            st.session_state.last_tune = tune
            st.success(f"Entraînement démarré — job_id=`{data['job_id']}`")
            st.rerun()
        elif response.status_code == 409:
            st.error(response.json().get("detail", "Un entraînement est déjà en cours."))
        else:
            st.error(f"Erreur API ({response.status_code}) : {response.text}")
    except requests.exceptions.ConnectionError:
        st.error(
            f"Impossible de joindre l'API à `{API_BASE_URL}`. "
            "Vérifie que le conteneur FastAPI est bien démarré."
        )
    except Exception as exc:
        st.error(f"Échec du démarrage de l'entraînement : {exc}")


@st.fragment(run_every=3)
def poll_training_status() -> None:
    """Interroge l'API toutes les ~3 s et affiche logs + résultats."""
    job_id = st.session_state.get("job_id")
    if not job_id:
        st.info("Aucun entraînement en cours. Configure les paramètres puis lance un job.")
        return

    try:
        response = api_get(f"/training-status/{job_id}", timeout=10)
    except requests.RequestException as exc:
        st.error(f"Impossible de récupérer le statut du job : {exc}")
        return

    if response.status_code == 404:
        st.error(f"Job `{job_id}` introuvable côté API.")
        if st.button("Réinitialiser l'état local"):
            st.session_state.job_id = None
            st.session_state.job_status = None
            st.rerun()
        return

    if response.status_code != 200:
        st.error(f"Erreur API ({response.status_code}) : {response.text}")
        return

    data = response.json()
    status = data.get("status", "unknown")
    st.session_state.job_status = status
    log_tail = data.get("log_tail") or ""
    result = data.get("result")

    if status == "pending":
        st.info(f"Job `{job_id}` en attente de démarrage…")
    elif status == "running":
        started = data.get("started_at") or "—"
        st.info(f"Job `{job_id}` en cours (démarré : {started})")
    elif status == "completed":
        st.success(f"Job `{job_id}` terminé avec succès.")
    elif status == "failed":
        st.error(
            f"Job `{job_id}` en échec "
            f"(code de retour : {data.get('return_code')})."
        )
    else:
        st.warning(f"Statut inconnu pour le job `{job_id}` : {status}")

    # Progression Optuna si visible dans les logs
    if st.session_state.get("last_tune") or data.get("tune"):
        matches = OPTUNA_PROGRESS_RE.findall(log_tail)
        if matches:
            current, total = matches[-1]
            current_i, total_i = int(current), int(total)
            if total_i > 0:
                st.progress(
                    min(current_i / total_i, 1.0),
                    text=f"Essai Optuna {current_i}/{total_i}",
                )

    st.subheader("Logs d'entraînement (temps réel)")
    st.code(log_tail if log_tail.strip() else "(aucun log pour le moment)", language="text")

    if status == "completed":
        if result:
            render_champion_metrics(result)
        else:
            st.warning(
                "Entraînement terminé, mais aucune métrique champion n'a été renvoyée. "
                "Vérifie MLflow puis recharge le modèle manuellement."
            )

        if st.button("🔄 Recharger le champion depuis MLflow", key="reload_after_train"):
            try:
                reload_resp = api_post("/reload-model", timeout=60)
                if reload_resp.status_code == 200:
                    payload = reload_resp.json()
                    if payload.get("status") == "success":
                        st.success(payload.get("message", "Champion rechargé."))
                        st.session_state.job_id = None
                        st.session_state.job_status = None
                        st.rerun()
                    else:
                        st.error(payload.get("message", "Échec du rechargement."))
                else:
                    st.error(f"Erreur API ({reload_resp.status_code}) : {reload_resp.text}")
            except requests.RequestException as exc:
                st.error(f"Rechargement impossible : {exc}")

        if st.button("Nouveau run (vider le job courant)", key="clear_after_ok"):
            st.session_state.job_id = None
            st.session_state.job_status = None
            st.rerun()

    elif status == "failed":
        st.error(
            "Consulte le log complet ci-dessus pour diagnostiquer l'échec "
            f"(return_code={data.get('return_code')})."
        )
        if st.button("Réessayer (vider le job courant)", key="clear_after_fail"):
            st.session_state.job_id = None
            st.session_state.job_status = None
            st.rerun()


poll_training_status()

st.markdown("---")
st.subheader("Commandes de référence")
st.code(
    """# Entraînement manuel classique
docker exec bris-aqueduc-trainer python -m src.train --model_type logistic
docker exec bris-aqueduc-trainer python -m src.train --model_type random_forest

# Optimisation Optuna
docker exec bris-aqueduc-trainer python -m src.train --model_type xgboost --tune --n_trials 15

# H2O AutoML
docker exec bris-aqueduc-trainer python -m src.train --model_type h2o""",
    language="bash",
)

st.info(
    "Après un entraînement réussi : utilise le bouton "
    "« Recharger le champion depuis MLflow » ci-dessus (ou dans la page principale) "
    "afin que l'API FastAPI charge le nouveau modèle champion.",
    icon="💡",
)
