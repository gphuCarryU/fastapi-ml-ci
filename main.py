from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ML Inference Service")

class PredictionInput(BaseModel):
    features: list[float]

class PredictionOutput(BaseModel):
    prediction: int
    confidence: float

@app.get("/")
def read_root():
    return {"status": "healthy", "service": "inference-api"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    # Baseline inference stub
    score = sum(data.features) / max(len(data.features), 1)
    predicted_class = 1 if score > 0.5 else 0
    return PredictionOutput(prediction=predicted_class, confidence=0.92)