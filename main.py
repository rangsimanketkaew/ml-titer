from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

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


@app.post("/predict")
async def predict(features: ModelFeatures):
    request_df = request_to_exp_dataframe(features.timestamps, features.values)
    feature_df = build_feature_table(request_df, filter_features=True)

    feature_matrix = feature_df.to_numpy(dtype=float)
    prediction = inference(MODEL, feature_matrix)

    return {"prediction": float(prediction[0])}
