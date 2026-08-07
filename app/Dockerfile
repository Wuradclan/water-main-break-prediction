FROM python:3.11-slim
WORKDIR /app

# --- Ajout : CLI Docker pour piloter le conteneur trainer depuis Streamlit ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends docker.io docker-compose-plugin && \
    rm -rf /var/lib/apt/lists/*
# -----------------------------------------------------------------------------

COPY requirements.txt .
#RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# On copie le nécessaire pour Streamlit
COPY app/ ./app/
# On copie src/ si Streamlit a besoin d'utilitaires
COPY src/ ./src/
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501"]
