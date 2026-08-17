import numpy as np
from mapie.regression import SplitConformalRegressor
from sklearn.cross_decomposition import PLSRegression

from ml.model import (
    get_model_metadata,
    inference_with_confidence,
    load_model,
    performance_model,
    save_model,
    train_pls_model,
)


def test_train_pls_model_creates_pls_regression_with_5_components():
    X = np.random.randn(20, 10)
    y = np.random.randn(20)

    model = train_pls_model(X, y, n_components=5)
    assert isinstance(model, SplitConformalRegressor)
    base = getattr(model, "_estimator", model)
    assert isinstance(base, PLSRegression)
    assert base.n_components == 5

    metrics = performance_model(model, X, y)
    assert hasattr(metrics, "r2")
    assert hasattr(metrics, "mae")
    assert hasattr(metrics, "rmse")
    assert hasattr(metrics, "mse")


def test_pls_model_save_and_load(tmp_path):
    X = np.random.randn(20, 10)
    y = np.random.randn(20)

    model = train_pls_model(X, y, n_components=5)
    save_path = tmp_path / "pls_model.joblib"
    save_model(model, save_path)

    assert save_path.exists()
    loaded_model = load_model(save_path)
    assert isinstance(loaded_model, SplitConformalRegressor)
    base = getattr(loaded_model, "_estimator", loaded_model)
    assert isinstance(base, PLSRegression)
    assert base.n_components == 5


def test_mapie_pls_uncertainty_calculation(tmp_path):
    np.random.seed(42)
    X = np.random.randn(30, 5)
    y = X[:, 0] * 2.0 + np.random.randn(30) * 0.5

    model = train_pls_model(X, y, n_components=3)

    res = inference_with_confidence(model, X[:1])
    assert "prediction" in res
    assert "uncertainty" in res
    assert "lower_bound" in res
    assert "upper_bound" in res
    assert res["uncertainty"] > 0
    assert abs(res["upper_bound"] - res["lower_bound"] - 2 * res["uncertainty"]) < 0.05

    save_path = tmp_path / "pls_model.joblib"
    meta = get_model_metadata(model, save_path)
    assert meta["name"] == "PLSRegression"
    assert meta["file"] == "pls_model.joblib"
