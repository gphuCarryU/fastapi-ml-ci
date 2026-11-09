"""ML Inference Service: serves a breast-cancer classifier over a REST API."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "model.joblib"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", BASE_DIR / "model" / "metadata.json"))

N_FEATURES = 30
MAX_BATCH_SIZE = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("inference-api")

# Holds the loaded model and its metadata for the lifetime of the app.
state: dict = {}


def load_model() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run `python train.py` first.")
    state["model"] = joblib.load(MODEL_PATH)
    state["metadata"] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    logger.info("Loaded model version %s from %s", state["metadata"]["model_version"], MODEL_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    state.clear()


app = FastAPI(
    title="ML Inference Service",
    description="Breast cancer tumor classification (malignant vs benign) using a scikit-learn model.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Schemas ----------

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class PredictionInput(BaseModel):
    features: list[FiniteFloat] = Field(
        ...,
        min_length=N_FEATURES,
        max_length=N_FEATURES,
        description=f"Exactly {N_FEATURES} numeric features, in the order listed by GET /model/info.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [
                        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                        1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                        25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189,
                    ]
                }
            ]
        }
    }


class BatchPredictionInput(BaseModel):
    instances: list[PredictionInput] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class PredictionOutput(BaseModel):
    prediction: int = Field(..., description="0 = malignant, 1 = benign")
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Probability of the predicted class")
    probabilities: dict[str, float]


class BatchPredictionOutput(BaseModel):
    predictions: list[PredictionOutput]
    model_version: str


# ---------- Helpers ----------


def get_model():
    model = state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return model


def run_inference(rows: list[list[float]]) -> list[PredictionOutput]:
    model = get_model()
    target_names: list[str] = state["metadata"]["target_names"]
    X = np.asarray(rows, dtype=float)
    probas = model.predict_proba(X)

    results = []
    for proba in probas:
        predicted = int(np.argmax(proba))
        results.append(
            PredictionOutput(
                prediction=predicted,
                label=target_names[predicted],
                confidence=round(float(proba[predicted]), 4),
                probabilities={name: round(float(p), 4) for name, p in zip(target_names, proba, strict=True)},
            )
        )
    return results


# ---------- Middleware ----------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info("%s %s -> %d (%.2f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


# ---------- Endpoints ----------


@app.get("/")
def read_root():
    return {"status": "healthy", "service": "inference-api"}


@app.get("/health")
def health():
    loaded = "model" in state
    return {
        "status": "healthy" if loaded else "unhealthy",
        "model_loaded": loaded,
        "model_version": state.get("metadata", {}).get("model_version"),
    }


@app.get("/model/info")
def model_info():
    get_model()
    return state["metadata"]


@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    return run_inference([data.features])[0]


@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(data: BatchPredictionInput):
    predictions = run_inference([item.features for item in data.instances])
    return BatchPredictionOutput(
        predictions=predictions,
        model_version=state["metadata"]["model_version"],
    )
