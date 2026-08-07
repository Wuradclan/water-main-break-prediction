"""
Page Streamlit d'entraînement — pilotage du service `trainer` via Docker.

Ajoute ce bloc à app/streamlit_app.py (ou crée app/pages/1_Entrainement.py
pour une page Streamlit multipage séparée).

Prérequis docker-compose.yml (service frontend) :
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    Et installer le CLI docker dans l'image frontend (voir Dockerfile ci-dessous).
"""

from __future__ import annotations

import shlex
import subprocess
import threading
import queue
import time

import streamlit as st

st.set_page_config(page_title="Entraînement — Bris d'aqueduc", page_icon="🛠️", layout="wide")

MODEL_CHOICES = {
    "Régression Logistique": "logistic",
    "Ridge": "ridge",
    "Lasso": "lasso",
    "Random Forest": "random_forest",
    "Extra Trees": "extra_trees",
    "XGBoost": "xgboost",
    "K-Nearest Neighbors": "knn",
    "SVC": "svc",
    "MLP (réseau de neurones)": "mlp",
    "Stacking": "stacking",
    "H2O AutoML": "h2o",
}

TRAINER_SERVICE = "trainer"


def build_command(model_type: str, tune: bool, n_trials: int, horizon_years: int) -> list[str]:
    cmd = [
        "docker", "compose", "exec", "-T", TRAINER_SERVICE,
        "python", "-m", "src.train",
        "--model_type", model_type,
        "--horizon_years", str(horizon_years),
    ]
    if tune:
        cmd += ["--tune", "--n_trials", str(n_trials)]
    return cmd


def stream_subprocess(cmd: list[str], output_queue: "queue.Queue[str]") -> None:
    """Run a command and push each output line into a queue for live display."""
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in iter(process.stdout.readline, ""):
            output_queue.put(line)
        process.stdout.close()
        process.wait()
        output_queue.put(f"\n[Processus terminé — code de sortie {process.returncode}]\n")
    except FileNotFoundError:
        output_queue.put("[Erreur] Commande docker introuvable dans le conteneur frontend.\n")
    except Exception as exc:
        output_queue.put(f"[Erreur] {exc}\n")
    finally:
        output_queue.put(None)  # sentinel: fin du flux


st.title("🛠️ Lancer un entraînement de modèle")
st.caption(
    "Cette page exécute `docker compose exec trainer python -m src.train ...` "
    "directement sur la VM hébergeant la stack MLOps."
)

st.warning(
    "⚠️ Un seul entraînement à la fois est recommandé. "
    "Vérifie qu'aucun collègue n'a déjà un entraînement en cours avant de lancer le tien.",
    icon="⚠️",
)

col1, col2 = st.columns(2)

with col1:
    model_label = st.selectbox("Type de modèle", options=list(MODEL_CHOICES.keys()))
    model_type = MODEL_CHOICES[model_label]

    horizon_years = st.selectbox("Horizon de prédiction (années)", options=[1, 2, 5], index=2)

with col2:
    is_optuna_capable = model_type not in {"h2o"}
    tune = False
    n_trials = 15
    if is_optuna_capable:
        tune = st.checkbox(
            "Optimisation Optuna (maximise PR-AUC, pénalise le surapprentissage F1)",
            value=(model_type == "xgboost"),
        )
        if tune:
            n_trials = st.slider("Nombre d'essais Optuna (--n_trials)", min_value=5, max_value=50, value=15)
    else:
        st.info("H2O AutoML gère sa propre recherche de modèles ; l'option Optuna est désactivée.")

command = build_command(model_type, tune, n_trials, horizon_years)

with st.expander("Commande qui sera exécutée", expanded=False):
    st.code(" ".join(shlex.quote(c) for c in command), language="bash")

if "training_running" not in st.session_state:
    st.session_state.training_running = False
if "training_log" not in st.session_state:
    st.session_state.training_log = ""

launch = st.button(
    "🚀 Lancer l'entraînement",
    type="primary",
    disabled=st.session_state.training_running,
)

log_placeholder = st.empty()

if launch:
    st.session_state.training_running = True
    st.session_state.training_log = ""
    output_queue: "queue.Queue[str]" = queue.Queue()

    thread = threading.Thread(target=stream_subprocess, args=(command, output_queue), daemon=True)
    thread.start()

    while True:
        try:
            line = output_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if line is None:
            break
        st.session_state.training_log += line
        log_placeholder.code(st.session_state.training_log[-6000:], language="text")

    st.session_state.training_running = False
    st.success("Entraînement terminé. Consulte MLflow pour voir les métriques du nouveau run.")

elif st.session_state.training_log:
    log_placeholder.code(st.session_state.training_log[-6000:], language="text")

st.markdown("---")
st.subheader("Commandes de référence")
st.code(
    "# Entraînement manuel classique (ex: Régression Logistique, Random Forest)\n"
    "docker compose exec trainer python -m src.train --model_type logistic\n"
    "docker compose exec trainer python -m src.train --model_type random_forest\n\n"
    "# Optimisation des hyperparamètres Optuna (maximise le PR-AUC avec pénalité de surapprentissage F1)\n"
    "docker compose exec trainer python -m src.train --model_type xgboost --tune --n_trials 15\n\n"
    "# Classification H2O AutoML\n"
    "docker compose exec trainer python -m src.train --model_type h2o",
    language="bash",
)

st.caption(
    "Après l'entraînement, retourne à la page principale et clique sur "
    "'Recharger le champion depuis MLflow' pour utiliser le nouveau modèle en inférence."
)
