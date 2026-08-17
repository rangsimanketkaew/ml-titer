from pathlib import Path

import joblib
import numpy as np
from mapie.regression import SplitConformalRegressor
from pydantic import BaseModel, ConfigDict
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


class ModelMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    r2: float
    mae: float
    rmse: float
    mse: float


def _prepare_xy(X, y):
    """
    Convert input data to numpy arrays
    """
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()


def train_mlr_model(
    X_train: np.ndarray, y_train: np.ndarray, confidence_level: float = 0.95
) -> SplitConformalRegressor:
    """
    Train a scaled multiple linear regression model
    """
    X_train, y_train = _prepare_xy(X_train, y_train)

    base_model = Pipeline(
        [("scaler", StandardScaler()), ("regressor", LinearRegression())]
    )
    base_model.fit(X_train, y_train)

    mapie_model = SplitConformalRegressor(
        estimator=base_model, confidence_level=confidence_level, prefit=True
    )
    mapie_model.conformalize(X_train, y_train)

    return mapie_model


def train_pls_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int = 5,
    confidence_level: float = 0.95,
) -> SplitConformalRegressor:
    """
    Train a Partial Least Squares (PLS) regression model
    """
    X_train, y_train = _prepare_xy(X_train, y_train)

    base_model = PLSRegression(n_components=n_components)
    base_model.fit(X_train, y_train)

    mapie_model = SplitConformalRegressor(
        estimator=base_model, confidence_level=confidence_level, prefit=True
    )
    mapie_model.conformalize(X_train, y_train)

    return mapie_model


def train_xgb_model(
    X_train: np.ndarray, y_train: np.ndarray, confidence_level: float = 0.95
) -> SplitConformalRegressor:
    """
    Train an XGBoost regressor
    """
    X_train, y_train = _prepare_xy(X_train, y_train)

    base_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=0,
    )
    base_model.fit(X_train, y_train)

    mapie_model = SplitConformalRegressor(
        estimator=base_model, confidence_level=confidence_level, prefit=True
    )
    mapie_model.conformalize(X_train, y_train)

    return mapie_model


def compute_model_metrics(y: np.ndarray, pred: np.ndarray) -> ModelMetrics:
    """
    Return common regression metrics for observed and predicted values
    """
    y = np.asarray(y, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()

    return ModelMetrics(
        r2=float(r2_score(y, pred)),
        mae=float(mean_absolute_error(y, pred)),
        rmse=float(np.sqrt(mean_squared_error(y, pred))),
        mse=float(mean_squared_error(y, pred)),
    )


def inference(model, X: np.ndarray) -> np.ndarray:
    """
    Run model prediction on new feature data
    """
    X_array = np.asarray(X, dtype=float)
    return model.predict(X_array)


def save_model(model, path: str | Path) -> Path:
    """
    Save a trained model
    """
    model_path = Path(path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path


def load_model(path: str | Path):
    """
    Load a trained model
    """
    return joblib.load(path)


def performance_model(model, X: np.ndarray, y: np.ndarray) -> ModelMetrics:
    """
    Evaluate a trained model on supplied features and targets
    """
    pred = inference(model, X)
    return compute_model_metrics(y, pred)


def inference_with_confidence(model, X: np.ndarray) -> dict[str, float]:
    """
    Run model prediction and calculate
    - point estimate
    - MAPIE uncertainty
    - confidence bounds
    """
    X_array = np.asarray(X, dtype=float)
    if X_array.ndim == 1:
        X_array = X_array.reshape(1, -1)

    if hasattr(model, "predict_interval"):
        y_pred, y_pis = model.predict_interval(X_array)
        pred_val = round(float(y_pred[0]), 2)
        lower_bound = round(float(y_pis[0, 0, 0]), 2)
        upper_bound = round(float(y_pis[0, 1, 0]), 2)
        uncertainty = round((upper_bound - lower_bound) / 2.0, 2)
    else:
        pred = inference(model, X_array)
        pred_val = round(float(pred[0]), 2)
        uncertainty = 0.0
        lower_bound = pred_val
        upper_bound = pred_val

    return {
        "prediction": pred_val,
        "uncertainty": uncertainty,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }


def _get_base_estimator(model):
    """
    Make sure that it returns the actual base estimator name
    """
    if hasattr(model, "_estimator"):
        model = model._estimator
    if hasattr(model, "_mapie_regressor") and hasattr(
        model._mapie_regressor, "estimator"
    ):
        model = model._mapie_regressor.estimator
    if isinstance(model, Pipeline):
        model = model.steps[-1][1]
    return model


def get_model_metadata(model, model_path: str | Path, version: str = "1.0.0") -> dict:
    """
    Extract metadata details from a loaded model object and file path
    """
    path = Path(model_path)
    base_estimator = _get_base_estimator(model)
    return {
        "name": type(base_estimator).__name__,
        "version": version,
        "file": path.name,
    }
