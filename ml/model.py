from pathlib import Path

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict
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
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()


def train_mlr_model(X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """
    Train a scaled multiple linear regression model
    """
    X_train, y_train = _prepare_xy(X_train, y_train)

    model = Pipeline([("scaler", StandardScaler()), ("regressor", LinearRegression())])
    model.fit(X_train, y_train)
    return model


def train_xgb_model(X_train: np.ndarray, y_train: np.ndarray) -> XGBRegressor:
    """
    Train an XGBoost regressor for scalar titer prediction
    """
    X_train, y_train = _prepare_xy(X_train, y_train)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=0,
    )
    model.fit(X_train, y_train)
    return model


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
