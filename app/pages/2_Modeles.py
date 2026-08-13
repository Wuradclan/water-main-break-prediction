"""
Page Streamlit — Liste des modèles entraînés et suppression.

Emplacement : app/pages/2_Modeles.py

Dépend des endpoints ajoutés à api/main.py :
    GET    /models
    DELETE /models/{run_id}?force=true|false

GET /models renvoie des runs de classification (break_within_horizon :
PR-AUC/F1/ROC-AUC) et de régression (years_until_break : RMSE/MAE/R²), avec
`null` pour les métriques qui ne s'appliquent pas à un run donné. Après
pd.DataFrame(models), ces `null` deviennent des NaN pandas : tout le
formatage ci-dessous passe donc par des helpers pd.isna-safe
(app/formatting.py) plutôt que par des comparaisons `is not None` qui
laisseraient échapper des "nan" / "nan / nan" à l'affichage.
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

try:
    # Exécution Streamlit réelle : le dossier app/ (contenant streamlit_app.py)
    # est ajouté à sys.path, donc les imports "à plat" fonctionnent.
    from formatting import format_f1_pair, format_metric
    from evaluation_display import (
        CONFUSION_MATRIX_EXPLANATION,
        CONFUSION_MATRIX_IMAGE_CAPTION,
        CONFUSION_MATRIX_SECTION_CAPTION,
        CONFUSION_MATRIX_UNAVAILABLE_MESSAGE,
        confusion_matrix_image_url,
        extract_confusion_counts,
        format_confusion_count,
        should_display_confusion_matrix,
    )
except ModuleNotFoundError:
    # Exécution hors Streamlit (ex: pytest depuis la racine du repo) : le
    # package namespace "app" est importable directement.
    from app.formatting import format_f1_pair, format_metric
    from app.evaluation_display import (
        CONFUSION_MATRIX_EXPLANATION,
        CONFUSION_MATRIX_IMAGE_CAPTION,
        CONFUSION_MATRIX_SECTION_CAPTION,
        CONFUSION_MATRIX_UNAVAILABLE_MESSAGE,
        confusion_matrix_image_url,
        extract_confusion_counts,
        format_confusion_count,
        should_display_confusion_matrix,
    )

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000").rstrip("/")

TASK_LABELS = {
    "classification": "Classification",
    "regression": "Régression",
}

st.set_page_config(page_title="Modèles entraînés", page_icon="📦", layout="wide")

st.title("📦 Modèles entraînés")
st.caption(
    "Liste des runs MLflow de premier niveau (essais Optuna imbriqués masqués), "
    "classification et régression confondues. "
    "Le champion actuellement chargé en mémoire est repéré par 🏆."
)


def fetch_models(include_deleted: bool = False) -> list[dict]:
    try:
        response = requests.get(
            f"{API_BASE_URL}/models",
            params={"include_deleted": include_deleted},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "success":
            st.error(data.get("message", "Erreur lors de la récupération des modèles."))
            return []
        return data.get("models", [])
    except requests.RequestException as exc:
        st.error(f"API inaccessible : {exc}")
        return []


def task_label(task: str | None) -> str:
    return TASK_LABELS.get(task, task or "classification")


def fetch_evaluation(run_id: str) -> dict | None:
    """GET /models/{run_id}/evaluation — source de vérité pour ce run précis
    (jamais un fichier reports/ local ni le champion en mémoire)."""
    try:
        response = requests.get(f"{API_BASE_URL}/models/{run_id}/evaluation", timeout=15)
        if response.status_code != 200:
            return None
        return response.json()
    except requests.RequestException:
        return None


def render_confusion_matrix_metrics(counts: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Vrais négatifs", format_confusion_count(counts["true_negatives"]))
    col2.metric("Faux positifs", format_confusion_count(counts["false_positives"]))
    col3.metric("Faux négatifs", format_confusion_count(counts["false_negatives"]))
    col4.metric("Vrais positifs", format_confusion_count(counts["true_positives"]))


include_deleted = st.checkbox("Afficher aussi les modèles déjà supprimés", value=False)

if st.button("🔄 Rafraîchir la liste"):
    st.cache_data.clear()

models = fetch_models(include_deleted=include_deleted)

if not models:
    st.info("Aucun modèle trouvé. Entraîne un modèle depuis la page « Entrainement ».")
    st.stop()

df = pd.DataFrame(models)

df["Champion"] = df["is_current_champion"].apply(lambda x: "🏆" if x else "")
if "task" not in df.columns:
    df["task"] = "classification"
df["Tâche"] = df["task"].apply(task_label)

# Classification metrics — "—" for regression runs (NaN after pd.DataFrame(models)).
df["PR-AUC test"] = df["pr_auc_test"].apply(format_metric)
df["F1 train/test"] = df.apply(
    lambda r: format_f1_pair(r.get("f1_train"), r.get("f1_test")), axis=1
)
df["ROC-AUC test"] = df["roc_auc_test"].apply(format_metric)

# Regression metrics — "—" for classification runs (NaN after pd.DataFrame(models)).
df["RMSE test"] = df["rmse_test"].apply(format_metric)
df["MAE test"] = df["mae_test"].apply(format_metric)
df["R² test"] = df["r2_test"].apply(format_metric)

display_df = df[[
    "Champion", "run_name", "Tâche", "model_type", "horizon_years",
    "PR-AUC test", "F1 train/test", "ROC-AUC test",
    "RMSE test", "MAE test", "R² test",
    "status", "start_time", "run_id",
]].rename(columns={
    "run_name": "Nom du run",
    "model_type": "Modèle",
    "horizon_years": "Horizon (ans)",
    "status": "Statut MLflow",
    "start_time": "Démarré le",
    "run_id": "Run ID",
})

st.subheader(f"{len(models)} modèle(s) trouvé(s)")
st.dataframe(display_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("📊 Matrice de confusion — historique (classification)")
st.caption(
    "Réservé à la classification à horizon fixe de 5 ans : la matrice provient "
    "toujours du run MLflow sélectionné ci-dessous, jamais d'un fichier global."
)

classification_models = [m for m in models if should_display_confusion_matrix(m.get("task"))]

if not classification_models:
    st.info("Aucun run de classification disponible.")
else:
    def _cm_run_label(m: dict) -> str:
        label = f"{m['run_name']} — {m['model_type']} ({m['run_id'][:8]}...)"
        if m["is_current_champion"]:
            label += " 🏆"
        return label

    cm_run_labels = {_cm_run_label(m): m["run_id"] for m in classification_models}
    cm_selected_label = st.selectbox(
        "Choisir un run de classification",
        options=list(cm_run_labels.keys()),
        key="confusion_matrix_run_selector",
    )
    cm_run_id = cm_run_labels[cm_selected_label]

    evaluation = fetch_evaluation(cm_run_id)

    if evaluation is None or not should_display_confusion_matrix(evaluation.get("task")):
        st.warning("Impossible de récupérer la matrice de confusion pour ce run.")
    else:
        st.caption(CONFUSION_MATRIX_SECTION_CAPTION)
        render_confusion_matrix_metrics(extract_confusion_counts(evaluation))

        if evaluation.get("artifact_available"):
            st.image(
                confusion_matrix_image_url(API_BASE_URL, cm_run_id),
                caption=CONFUSION_MATRIX_IMAGE_CAPTION,
                use_container_width=True,
            )
            st.caption(CONFUSION_MATRIX_EXPLANATION)
        else:
            st.info(CONFUSION_MATRIX_UNAVAILABLE_MESSAGE)

st.markdown("---")
st.subheader("🗑️ Supprimer un modèle")


def run_label(m: dict) -> str:
    label = f"{m['run_name']} — {task_label(m.get('task'))} — {m['model_type']} ({m['run_id'][:8]}...)"
    if m["is_current_champion"]:
        label += " 🏆 CHAMPION ACTIF"
    return label


run_labels = {run_label(m): m["run_id"] for m in models}

selected_label = st.selectbox("Choisir le run à supprimer", options=list(run_labels.keys()))
selected_run_id = run_labels[selected_label]
selected_model = next(m for m in models if m["run_id"] == selected_run_id)

st.caption(
    f"Tâche : **{task_label(selected_model.get('task'))}** · "
    f"Modèle : `{selected_model['model_type']}`"
)

detail_cols = st.columns(3)
if selected_model.get("task") == "regression":
    detail_cols[0].metric("RMSE test", format_metric(selected_model.get("rmse_test")))
    detail_cols[1].metric("MAE test", format_metric(selected_model.get("mae_test")))
    detail_cols[2].metric("R² test", format_metric(selected_model.get("r2_test")))
else:
    detail_cols[0].metric("PR-AUC test", format_metric(selected_model.get("pr_auc_test")))
    detail_cols[1].metric(
        "F1 train/test",
        format_f1_pair(selected_model.get("f1_train"), selected_model.get("f1_test")),
    )
    detail_cols[2].metric("ROC-AUC test", format_metric(selected_model.get("roc_auc_test")))

if selected_model["is_current_champion"]:
    st.warning(
        "⚠️ Ce run est actuellement le **champion chargé en mémoire** par l'API "
        f"({task_label(selected_model.get('task'))}). "
        "Le supprimer forcera un rechargement du prochain meilleur modèle disponible "
        "la prochaine fois que quelqu'un cliquera sur « Recharger le champion »."
    )

confirm_checkbox = st.checkbox(
    f"Je confirme vouloir supprimer le run `{selected_run_id}`",
    value=False,
)

delete_clicked = st.button(
    "🗑️ Supprimer définitivement ce modèle",
    type="primary",
    disabled=not confirm_checkbox,
)

if delete_clicked:
    try:
        force = selected_model["is_current_champion"]
        response = requests.delete(
            f"{API_BASE_URL}/models/{selected_run_id}",
            params={"force": force},
            timeout=15,
        )

        if response.status_code == 200:
            result = response.json()
            st.success(result.get("message", "Modèle supprimé."))
            if result.get("was_champion"):
                st.info(
                    "Le champion actif vient d'être supprimé. "
                    "Pense à retourner sur la page principale et cliquer sur "
                    "« Recharger le champion depuis MLflow »."
                )
            st.rerun()
        elif response.status_code == 409:
            st.error(response.json().get("detail", "Suppression refusée : c'est le champion actif."))
        else:
            st.error(f"Erreur API ({response.status_code}) : {response.text}")

    except requests.RequestException as exc:
        st.error(f"API inaccessible : {exc}")

st.markdown("---")
st.caption(
    "Note : MLflow effectue une suppression logique (soft delete). Les artefacts "
    "restent sur disque jusqu'à l'exécution d'une purge côté serveur MLflow "
    "(`mlflow gc`). Un run supprimé n'apparaît plus dans le Model Gate."
)
