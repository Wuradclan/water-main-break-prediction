FROM python:3.11-slim

# On installe UNIQUEMENT mlflow.
# SQLite est inclus par défaut dans Python 3.11, donc aucun autre package n'est nécessaire.
RUN pip install --no-cache-dir mlflow

EXPOSE 5000