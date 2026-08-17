import numpy as np
from sklearn.cross_decomposition import PLSRegression

from ml.model import load_model, performance_model, save_model, train_pls_model


def test_train_pls_model_creates_pls_regression_with_5_components():
    X = np.random.randn(20, 10)
    y = np.random.randn(20)

    model = train_pls_model(X, y, n_components=5)
    assert isinstance(model, PLSRegression)
    assert model.n_components == 5

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
    assert isinstance(loaded_model, PLSRegression)
    assert loaded_model.n_components == 5
