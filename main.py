from __future__ import annotations

from pathlib import Path

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ValidationError

from ml.data import build_feature_table, request_to_exp_dataframe
from ml.model import inference, load_model

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "xgb_model.joblib"
MODEL = load_model(MODEL_PATH)


class ModelFeatures(BaseModel):
    timestamps: list[float]
    values: dict[str, list[float]]


app = FastAPI(
    title="ML Inference Service",
    version="1.0.0",
    description="API for ML inference on mAb Titer in bioprocessing",
)


@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def _parse_yaml_to_features(content: bytes) -> ModelFeatures:
    try:
        parsed = yaml.safe_load(content) or {}
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML format: {e}")

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="YAML content must be a dictionary")

    if "timestamps" in parsed and "values" in parsed:
        raw_data = parsed
    else:
        schemas = parsed.get("components", {}).get("schemas", {})
        properties = schemas.get("PredictRequest", {}).get(
            "properties", {}
        ) or parsed.get("properties", {})
        raw_data = {
            "timestamps": properties.get("timestamps", {}).get("example", []),
            "values": properties.get("values", {}).get("example", {}),
        }

    try:
        return ModelFeatures.model_validate(raw_data)
    except ValidationError as e:
        raise HTTPException(
            status_code=422, detail=f"Invalid request payload structure: {e}"
        )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):  # noqa: B008
    content = await file.read()
    features = _parse_yaml_to_features(content)

    request_df = request_to_exp_dataframe(features.timestamps, features.values)
    feature_df = build_feature_table(request_df).drop(
        columns=["n_days"], errors="ignore"
    )

    feature_matrix = feature_df.to_numpy(dtype=float)
    prediction = inference(MODEL, feature_matrix)

    return {"prediction": float(prediction[0])}
