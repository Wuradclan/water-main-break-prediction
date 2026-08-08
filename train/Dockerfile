# Dockerfile situé dans le dossier 'train/'
FROM python:3.11-slim

# Installation des dépendances système pour XGBoost ET H2O (Java)
# On utilise default-jre au lieu de default-jdk pour gagner énormément d'espace
RUN apt-get update && apt-get install -y \
    libomp-dev \
    default-jre \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# On copie les dépendances
COPY requirements.txt .

# L'OPTION MAGIQUE : --no-cache-dir empêche pip de stocker les fichiers d'installation
RUN pip install --no-cache-dir -r requirements.txt

# On copie le code source (le dossier src est au niveau supérieur)
COPY src/ ./src/
# On copie aussi les données si besoin
COPY data/ ./data/

# Maintient le conteneur en vie pour permettre d'utiliser 'docker compose exec'
CMD ["sleep", "infinity"]