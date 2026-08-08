# api/Dockerfile
FROM python:3.11-slim

# docker.io : l'API lance les entraînements via `docker exec` sur le trainer
# libomp-dev (XGBoost) + JRE (chargement H2O éventuel)
RUN apt-get update && \
    apt-get install -y --no-install-recommends docker.io docker-cli libomp-dev default-jre && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# --no-cache-dir pour ne pas stocker les fichiers d'installation lourds
RUN pip install --no-cache-dir -r requirements.txt

# On copie le code source
COPY api/ ./api/
COPY src/ ./src/
# Create the empty models directory instead of copying it
RUN mkdir -p ./models/

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]