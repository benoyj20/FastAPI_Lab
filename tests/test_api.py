"""Tests for the Iris FastAPI app. Run from the repo root with `pytest`."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import main  # noqa: E402
from train import MODEL_INFO_PATH, MODEL_PATH, train  # noqa: E402

SETOSA = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
VERSICOLOR = {"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, "petal_width": 1.5}
VIRGINICA = {"sepal_length": 6.9, "sepal_width": 3.1, "petal_length": 5.8, "petal_width": 2.3}


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Make sure the model and its metadata exist before the tests run."""
    if not (MODEL_PATH.exists() and MODEL_INFO_PATH.exists()):
        train()


@pytest.fixture(scope="session")
def client():
    # Using TestClient as a context manager runs the lifespan (model loading).
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    """A client whose server started with no model files on disk."""
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "missing.pkl")
    monkeypatch.setattr(main, "MODEL_INFO_PATH", tmp_path / "missing.json")
    with TestClient(main.app) as c:
        yield c


# --- Health and model info -------------------------------------------------

def test_health_reports_model_loaded(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": True}


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    info = response.json()
    assert info["model_type"] == "DecisionTreeClassifier"
    assert info["hyperparameters"]["max_depth"] == 3
    assert info["class_names"] == ["setosa", "versicolor", "virginica"]
    assert info["feature_names"] == ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    assert info["test_accuracy"] > 0.9
    assert info["train_samples"] + info["test_samples"] == 150


# --- Single prediction -----------------------------------------------------

@pytest.mark.parametrize(
    "features, class_id, class_name",
    [(SETOSA, 0, "setosa"), (VERSICOLOR, 1, "versicolor"), (VIRGINICA, 2, "virginica")],
)
def test_predict_returns_class_and_probabilities(client, features, class_id, class_name):
    response = client.post("/predict", json=features)
    assert response.status_code == 200
    body = response.json()
    assert body["class_id"] == class_id
    assert body["class_name"] == class_name
    assert set(body["probabilities"]) == {"setosa", "versicolor", "virginica"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["confidence"] == max(body["probabilities"].values())
    assert body["probabilities"][class_name] == body["confidence"]


def test_predict_converts_numeric_strings(client):
    features = {k: str(v) for k, v in SETOSA.items()}
    response = client.post("/predict", json=features)
    assert response.status_code == 200
    assert response.json()["class_name"] == "setosa"


# --- Input validation (422) ------------------------------------------------

def test_predict_missing_field_returns_422(client):
    response = client.post("/predict", json={"sepal_length": 5.1, "sepal_width": 3.5})
    assert response.status_code == 422


def test_predict_invalid_type_returns_422(client):
    response = client.post("/predict", json={**SETOSA, "sepal_length": "five point one"})
    assert response.status_code == 422


@pytest.mark.parametrize("bad_value", [-1.0, 0, 10.5])
def test_predict_out_of_range_returns_422(client, bad_value):
    response = client.post("/predict", json={**SETOSA, "petal_length": bad_value})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "petal_length"]


# --- Batch prediction ------------------------------------------------------

def test_predict_batch(client):
    response = client.post("/predict/batch", json={"samples": [SETOSA, VERSICOLOR, VIRGINICA]})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert [p["class_name"] for p in body["predictions"]] == ["setosa", "versicolor", "virginica"]


def test_predict_batch_empty_returns_400(client):
    response = client.post("/predict/batch", json={"samples": []})
    assert response.status_code == 400
    assert "at least one" in response.json()["detail"]


def test_predict_batch_too_large_returns_400(client):
    samples = [SETOSA] * (main.MAX_BATCH_SIZE + 1)
    response = client.post("/predict/batch", json={"samples": samples})
    assert response.status_code == 400
    assert "maximum" in response.json()["detail"]


def test_predict_batch_max_size_is_allowed(client):
    samples = [SETOSA] * main.MAX_BATCH_SIZE
    response = client.post("/predict/batch", json={"samples": samples})
    assert response.status_code == 200
    assert response.json()["count"] == main.MAX_BATCH_SIZE


def test_predict_batch_invalid_sample_returns_422(client):
    response = client.post("/predict/batch", json={"samples": [SETOSA, {**SETOSA, "sepal_width": -2}]})
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "samples", 1, "sepal_width"]


# --- Model not loaded (503) ------------------------------------------------

def test_health_when_model_missing(client_without_model):
    response = client_without_model.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model_loaded": False}


@pytest.mark.parametrize(
    "method, path, payload",
    [
        ("get", "/model/info", None),
        ("post", "/predict", SETOSA),
        ("post", "/predict/batch", {"samples": [SETOSA]}),
    ],
)
def test_endpoints_return_503_when_model_missing(client_without_model, method, path, payload):
    response = client_without_model.request(method.upper(), path, json=payload)
    assert response.status_code == 503
    assert "train.py" in response.json()["detail"]
