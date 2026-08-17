from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from ml.data import build_feature_table, request_to_exp_dataframe
from ml.model import get_model_metadata, inference_with_confidence, load_model

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
DEFAULT_MODEL_ID = "pls"

MODEL_REGISTRY = {
    "mlr": {
        "algorithm": "Multiple Linear Regression (MLR)",
        "file": "mlr_model.joblib",
        "path": MODELS_DIR / "mlr_model.joblib",
        "model": load_model(MODELS_DIR / "mlr_model.joblib"),
    },
    "pls": {
        "algorithm": "Partial Least Squares (PLS)",
        "file": "pls_model.joblib",
        "path": MODELS_DIR / "pls_model.joblib",
        "model": load_model(MODELS_DIR / "pls_model.joblib"),
    },
    "xgb": {
        "algorithm": "XGBoost Regressor",
        "file": "xgb_model.joblib",
        "path": MODELS_DIR / "xgb_model.joblib",
        "model": load_model(MODELS_DIR / "xgb_model.joblib"),
    },
}


class ModelFeatures(BaseModel):
    model: str = Field(
        "pls", description="Model ID to choose for inference: mlr, pls, xgb"
    )
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


class AvailableModelItem(BaseModel):
    id: str
    algorithm: str
    file: str


class AvailableModelsResponse(BaseModel):
    active_model: str
    available_models: list[AvailableModelItem]


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


@app.get("/models", response_model=AvailableModelsResponse)
async def get_models():
    available_models = [
        AvailableModelItem(id=k, algorithm=v["algorithm"], file=v["file"])
        for k, v in MODEL_REGISTRY.items()
    ]
    return AvailableModelsResponse(
        active_model=MODEL_REGISTRY[DEFAULT_MODEL_ID]["file"],
        available_models=available_models,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: ModelFeatures):
    selected_id = features.model.lower() if features.model else DEFAULT_MODEL_ID
    if selected_id not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{features.model}'. Available models: {list(MODEL_REGISTRY.keys())}",
        )

    model_entry = MODEL_REGISTRY[selected_id]
    model_path = model_entry["path"]
    model_obj = model_entry["model"]

    request_df = request_to_exp_dataframe(features.timestamps, features.values)
    feature_df = build_feature_table(request_df, filter_features=True)

    pred_res = inference_with_confidence(model_obj, feature_df.to_numpy(dtype=float))
    model_meta = get_model_metadata(model_obj, model_path)

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
