import joblib
import numpy as np
from fastapi import FastAPI, status
from pydantic import BaseModel, Field, field_validator

app = FastAPI(
    title="ML Inference Service",
    version="1.0.0",
    description="Microservice for ML inference on bioprocessing data",
)

model = joblib.load("model.joblib")


@app.get("/")
def home():
    return {"message": "Hello, FastAPI!"}


class InferenceRequest(BaseModel):
    features: list[float]


# Prediction endpoint
@app.post("/predict")
async def predict(request: InferenceRequest):
    data = np.array([request.features])
    prediction = model.predict(data)
    return {"prediction": prediction.tolist()}


# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
