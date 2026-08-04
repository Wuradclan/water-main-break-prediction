# src/predict.py
import joblib
from pathlib import Path

MODEL_PATH = Path("models/model.pkl")

def get_prediction(input_data):
    # Charge le modèle
    model = joblib.load(MODEL_PATH)
    # Effectue la prédiction
    return model.predict(input_data)