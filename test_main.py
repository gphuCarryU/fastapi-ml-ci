import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from main import METADATA_PATH, N_FEATURES, app

# Minimum quality bar: CI fails if a retrained model drops below these.
MIN_ACCURACY = 0.93
MIN_F1 = 0.93


@pytest.fixture(scope="module")
def client():
    # Using the context manager runs the lifespan handler, which loads the model.
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def dataset():
    return load_breast_cancer()


# ---------- Health & info ----------


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_reports_model_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_version"]


def test_model_info(client):
    body = client.get("/model/info").json()
    assert len(body["feature_names"]) == N_FEATURES
    assert body["target_names"] == ["malignant", "benign"]
    assert "metrics" in body


def test_process_time_header(client):
    response = client.get("/health")
    assert "x-process-time-ms" in response.headers


# ---------- Prediction ----------


def test_predict_returns_valid_schema(client, dataset):
    payload = {"features": dataset.data[0].tolist()}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("malignant", "benign")
    assert 0.5 <= body["confidence"] <= 1.0
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_confidence_is_not_constant(client, dataset):
    confidences = {
        client.post("/predict", json={"features": dataset.data[i].tolist()}).json()["confidence"]
        for i in range(20)
    }
    assert len(confidences) > 1


@pytest.mark.parametrize("label", [0, 1])
def test_predict_typical_samples(client, dataset, label):
    # The feature-wise mean of each class should be classified as that class.
    centroid = dataset.data[dataset.target == label].mean(axis=0)
    body = client.post("/predict", json={"features": centroid.tolist()}).json()
    assert body["prediction"] == label


def test_api_accuracy_on_full_dataset(client, dataset):
    instances = [{"features": row.tolist()} for row in dataset.data]
    predictions = []
    for start in range(0, len(instances), 100):
        response = client.post("/predict/batch", json={"instances": instances[start : start + 100]})
        assert response.status_code == 200
        predictions += [p["prediction"] for p in response.json()["predictions"]]
    accuracy = np.mean(np.array(predictions) == dataset.target)
    assert accuracy >= MIN_ACCURACY


def test_batch_predict(client, dataset):
    payload = {"instances": [{"features": dataset.data[i].tolist()} for i in range(5)]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 5
    assert body["model_version"]


# ---------- Input validation ----------


@pytest.mark.parametrize(
    "payload",
    [
        {"features": [0.6, 0.8, 0.4]},  # too few features
        {"features": [1.0] * (N_FEATURES + 1)},  # too many features
        {"features": []},  # empty
        {"features": ["abc"] * N_FEATURES},  # not numbers
        {"wrong_key": [1.0] * N_FEATURES},  # missing field
        {},  # empty body
    ],
)
def test_predict_rejects_invalid_input(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_rejects_empty_and_oversized(client):
    assert client.post("/predict/batch", json={"instances": []}).status_code == 422
    too_many = {"instances": [{"features": [1.0] * N_FEATURES}] * 101}
    assert client.post("/predict/batch", json=too_many).status_code == 422


# ---------- Model quality gate ----------


def test_model_metrics_meet_threshold():
    metrics = json.loads(METADATA_PATH.read_text(encoding="utf-8"))["metrics"]
    assert metrics["accuracy"] >= MIN_ACCURACY
    assert metrics["f1"] >= MIN_F1
