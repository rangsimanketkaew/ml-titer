from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from ml.data import build_feature_table, request_to_exp_dataframe
from ml.model import get_model_metadata, inference_with_confidence, load_model

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "pls_model.joblib"
MODEL = load_model(MODEL_PATH)


class ModelFeatures(BaseModel):
    timestamps: list[float]
    values: dict[str, list[float]]


class ModelInfo(BaseModel):
    name: str
    version: str = "1.0.0"
    file: str


class ConfidenceInterval(BaseModel):
    lower_bound: float
    upper_bound: float


class PredictionResponse(BaseModel):
    status: str = "success"
    prediction: float = Field(..., description="Predicted Titer value")
    unit: str = "mg/L"
    uncertainty: float = Field(
        ..., description="Margin of error at 95% confidence level"
    )
    confidence_interval: ConfidenceInterval
    model_info: ModelInfo


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


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: ModelFeatures):
    request_df = request_to_exp_dataframe(features.timestamps, features.values)
    feature_df = build_feature_table(request_df, filter_features=True)

    pred_res = inference_with_confidence(MODEL, feature_df.to_numpy(dtype=float))
    model_meta = get_model_metadata(MODEL, MODEL_PATH)

    return PredictionResponse(
        status="success",
        prediction=pred_res["prediction"],
        unit="mg/L",
        uncertainty=pred_res["uncertainty"],
        confidence_interval=ConfidenceInterval(
            lower_bound=pred_res["lower_bound"],
            upper_bound=pred_res["upper_bound"],
        ),
        model_info=ModelInfo(**model_meta),
    )
