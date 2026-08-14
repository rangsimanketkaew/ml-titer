from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from ml.data import build_feature_table, spec_yml_to_dataframe
from ml.model import inference, load_model

ROOT_DIR = Path(__file__).resolve().parent
SPEC_PATH = ROOT_DIR / "inference_server_spec.yml"
MODEL_PATH = ROOT_DIR / "models" / "xgb_model.joblib"

app = FastAPI(
    title="ML Inference Service",
    version="1.0.0",
    description="Microservice for ML inference on bioprocessing data",
)


class ModelFeatures(BaseModel):
    timestamps: list[float]
    values: dict[str, list[float]]


@app.get("/")
def home():
    return {"message": "Hello, FastAPI!"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def _request_to_exp_dataframe(request: ModelFeatures) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, day in enumerate(request.timestamps):
        row: dict[str, object] = {"Exp": "inference", "Time[day]": day}
        for key, values in request.values.items():
            if key.startswith("Z:"):
                row[key] = values[0]
            elif i < len(values):
                row[key] = values[i]
            else:
                row[key] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


@app.post("/predict")
async def predict(features: ModelFeatures):
    spec_df = spec_yml_to_dataframe(SPEC_PATH)
    template_features = build_feature_table(spec_df).drop(
        columns=["n_days"], errors="ignore"
    )

    request_df = _request_to_exp_dataframe(features)
    feature_df = build_feature_table(request_df).drop(
        columns=["n_days"], errors="ignore"
    )

    feature_df = feature_df.reindex(
        columns=template_features.columns, fill_value=np.nan
    )

    model = load_model(MODEL_PATH)
    feature_matrix = feature_df.to_numpy(dtype=float)
    prediction = inference(model, feature_matrix)

    return {"prediction": float(prediction[0])}
