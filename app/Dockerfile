FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
#RUN pip install -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# On copie le nécessaire pour Streamlit
COPY app/ ./app/
# On copie src/ si Streamlit a besoin d'utilitaires
COPY src/ ./src/
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501"]