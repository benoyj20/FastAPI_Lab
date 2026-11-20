"""Train a Decision Tree Classifier on the Iris dataset.

Saves two files to model/:
  - iris_model.pkl   the trained model (joblib)
  - model_info.json  metadata served by the API at GET /model/info
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from data import CLASS_NAMES, FEATURE_NAMES, load_data, split_data

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "iris_model.pkl"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"


def fit_model(X_train, y_train, model_path=MODEL_PATH):
    """Fit a Decision Tree Classifier and save it with joblib."""
    dt_classifier = DecisionTreeClassifier(max_depth=3, random_state=12)
    dt_classifier.fit(X_train, y_train)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dt_classifier, model_path)
    return dt_classifier


def save_model_info(model, accuracy, n_train, n_test, info_path=MODEL_INFO_PATH):
    """Write the model's metadata to a JSON file next to the model."""
    info = {
        "model_type": type(model).__name__,
        "hyperparameters": model.get_params(),
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "test_accuracy": round(float(accuracy), 4),
        "train_samples": int(n_train),
        "test_samples": int(n_test),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
    }
    info_path = Path(info_path)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(info, indent=2))
    return info


def train():
    """Run the full training pipeline and return the saved metadata."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return save_model_info(model, accuracy, len(X_train), len(X_test))


if __name__ == "__main__":
    info = train()
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metadata saved to {MODEL_INFO_PATH}")
    print(f"Test accuracy: {info['test_accuracy']:.4f}")
