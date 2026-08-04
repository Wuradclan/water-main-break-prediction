# api/Dockerfile
FROM python:3.11-slim

# Ensuite seulement, on peut mettre les autres instructions
#RUN apt-get update && apt-get install -y libomp-dev && rm -rf /var/lib/apt/lists/*
# Ajout de Java pour pouvoir charger les modèles H2O
RUN apt-get update && apt-get install -y default-jdk libomp-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# On copie le code source
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]