"""Load the trained Iris model and its metadata, and make predictions."""
import json
from pathlib import Path

import joblib

from data import CLASS_NAMES

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "iris_model.pkl"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"

_TRAIN_HINT = "Run `python train.py` from the src/ folder first."


def load_model(model_path=MODEL_PATH):
    """Load the trained model from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. {_TRAIN_HINT}")
    return joblib.load(model_path)


def load_model_info(info_path=MODEL_INFO_PATH):
    """Load the model metadata written by train.py."""
    info_path = Path(info_path)
    if not info_path.exists():
        raise FileNotFoundError(f"Model metadata not found at {info_path}. {_TRAIN_HINT}")
    return json.loads(info_path.read_text())


def predict_data(model, X):
    """Predict classes and class probabilities.

    Args:
        model: A fitted scikit-learn classifier.
        X (array-like): 2D input of shape (n_samples, 4) in the order
            [sepal_length, sepal_width, petal_length, petal_width].

    Returns:
        list[dict]: One dict per sample with class_id, class_name,
            confidence and a probability for every class.
    """
    class_ids = model.predict(X)
    probas = model.predict_proba(X)

    results = []
    for class_id, proba in zip(class_ids, probas):
        probabilities = {
            CLASS_NAMES[int(c)]: round(float(p), 4) for c, p in zip(model.classes_, proba)
        }
        results.append({
            "class_id": int(class_id),
            "class_name": CLASS_NAMES[int(class_id)],
            "confidence": round(float(proba.max()), 4),
            "probabilities": probabilities,
        })
    return results
