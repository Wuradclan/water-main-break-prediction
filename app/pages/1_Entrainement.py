"""
Page Streamlit pour lancer les entraînements du projet MLOps.

Emplacement :
    app/pages/1_Entrainement.py

Prérequis :
- Le service frontend doit monter /var/run/docker.sock
- L'image frontend doit contenir le CLI Docker
- Le conteneur trainer doit s'appeler bris-aqueduc-trainer
"""

from __future__ import annotations

import queue
import shlex
import subprocess
import threading

import streamlit as st


TRAINER_CONTAINER = "bris-aqueduc-trainer"

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


def build_command(
    model_type: str,
    tune: bool,
    n_trials: int,
    horizon_years: int,
) -> list[str]:
    """
    Construit une commande exécutée dans le conteneur trainer.

    Exemple :
    docker exec bris-aqueduc-trainer python -m src.train
      --model_type xgboost --horizon_years 5 --tune --n_trials 15
    """
    command = [
        "docker",
        "exec",
        TRAINER_CONTAINER,
        "python",
        "-m",
        "src.train",
        "--model_type",
        model_type,
        "--horizon_years",
        str(horizon_years),
    ]

    if tune:
        command.extend(["--tune", "--n_trials", str(n_trials)])

    return command


def stream_training_logs(
    command: list[str],
    output_queue: queue.Queue,
) -> None:
    """
    Lance la commande Docker et envoie les logs ligne par ligne
    vers la page Streamlit.
    """
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout is not None:
            for line in iter(process.stdout.readline, ""):
                output_queue.put(line)

            process.stdout.close()

        return_code = process.wait()

        output_queue.put(
            f"\n{'=' * 70}\n"
            f"Processus terminé — code de sortie : {return_code}\n"
            f"{'=' * 70}\n"
        )

    except FileNotFoundError:
        output_queue.put(
            "\n[ERREUR] La commande Docker est introuvable dans le conteneur "
            "frontend. Vérifie le Dockerfile de app/.\n"
        )

    except Exception as exc:
        output_queue.put(f"\n[ERREUR] Impossible de lancer l'entraînement : {exc}\n")

    finally:
        output_queue.put(None)


st.set_page_config(
    page_title="Entraînement — Bris d'aqueduc",
    page_icon="🛠️",
    layout="wide",
)

st.title("🛠️ Entraînement des modèles")
st.caption(
    "Cette interface déclenche l'entraînement dans le conteneur "
    "`bris-aqueduc-trainer` hébergé sur la VM."
)

st.warning(
    "Un seul entraînement à la fois est recommandé. "
    "Les entraînements Optuna et H2O AutoML peuvent prendre plusieurs minutes.",
    icon="⚠️",
)

if "training_running" not in st.session_state:
    st.session_state.training_running = False

if "training_log" not in st.session_state:
    st.session_state.training_log = ""

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

command = build_command(
    model_type=model_type,
    tune=tune,
    n_trials=n_trials,
    horizon_years=horizon_years,
)

st.subheader("Commande générée")

with st.expander("Afficher la commande Docker", expanded=True):
    command_as_text = " ".join(shlex.quote(argument) for argument in command)

    st.code(
        command_as_text,
        language="bash",
    )

launch_button = st.button(
    "🚀 Lancer l'entraînement",
    type="primary",
    disabled=st.session_state.training_running,
)

log_placeholder = st.empty()

if launch_button:
    st.session_state.training_running = True
    st.session_state.training_log = ""

    output_queue: queue.Queue = queue.Queue()

    thread = threading.Thread(
        target=stream_training_logs,
        args=(command, output_queue),
        daemon=True,
    )

    thread.start()

    with st.spinner("Entraînement en cours... Les logs s'affichent ci-dessous."):
        while True:
            try:
                line = output_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if line is None:
                break

            st.session_state.training_log += line

            log_placeholder.code(
                st.session_state.training_log[-10000:],
                language="text",
            )

    st.session_state.training_running = False

    if "code de sortie : 0" in st.session_state.training_log:
        st.success(
            "Entraînement terminé avec succès. "
            "Consulte MLflow, puis recharge le modèle champion dans la page principale."
        )
    else:
        st.error(
            "L'entraînement s'est terminé avec une erreur. "
            "Consulte les logs affichés ci-dessus."
        )

elif st.session_state.training_log:
    log_placeholder.code(
        st.session_state.training_log[-10000:],
        language="text",
    )

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
    "Après un entraînement réussi : ouvre la page principale de prédiction, "
    "puis clique sur « Recharger le champion depuis MLflow » afin que l'API "
    "FastAPI charge le nouveau modèle champion.",
    icon="💡",
)
