"""
Page Streamlit — Liste des modèles entraînés et suppression.

Emplacement : app/pages/2_Modeles.py

Dépend des endpoints ajoutés à api/main.py :
    GET    /models
    DELETE /models/{run_id}?force=true|false
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://api:8000").rstrip("/")

st.set_page_config(page_title="Modèles entraînés", page_icon="📦", layout="wide")

st.title("📦 Modèles entraînés")
st.caption(
    "Liste des runs MLflow de premier niveau (essais Optuna imbriqués masqués). "
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


include_deleted = st.checkbox("Afficher aussi les modèles déjà supprimés", value=False)

if st.button("🔄 Rafraîchir la liste"):
    st.cache_data.clear()

models = fetch_models(include_deleted=include_deleted)

if not models:
    st.info("Aucun modèle trouvé. Entraîne un modèle depuis la page « Entrainement ».")
    st.stop()

df = pd.DataFrame(models)

df["Champion"] = df["is_current_champion"].apply(lambda x: "🏆" if x else "")
df["PR-AUC test"] = df["pr_auc_test"].apply(lambda v: f"{v:.3f}" if v is not None else "—")
df["F1 train/test"] = df.apply(
    lambda r: f"{r['f1_train']:.3f} / {r['f1_test']:.3f}"
    if r["f1_train"] is not None and r["f1_test"] is not None
    else "—",
    axis=1,
)
df["ROC-AUC test"] = df["roc_auc_test"].apply(lambda v: f"{v:.3f}" if v is not None else "—")

display_df = df[[
    "Champion", "run_name", "model_type", "horizon_years",
    "PR-AUC test", "F1 train/test", "ROC-AUC test", "status", "start_time", "run_id",
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
st.subheader("🗑️ Supprimer un modèle")

run_labels = {
    f"{m['run_name']} — {m['model_type']} ({m['run_id'][:8]}...)" +
    (" 🏆 CHAMPION ACTIF" if m["is_current_champion"] else ""): m["run_id"]
    for m in models
}

selected_label = st.selectbox("Choisir le run à supprimer", options=list(run_labels.keys()))
selected_run_id = run_labels[selected_label]
selected_model = next(m for m in models if m["run_id"] == selected_run_id)

if selected_model["is_current_champion"]:
    st.warning(
        "⚠️ Ce run est actuellement le **champion chargé en mémoire** par l'API. "
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
