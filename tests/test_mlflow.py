import numpy as np

from ml.mlflow_utils import log_model_run, setup_experiment
from ml.model import performance_model, train_pls_model


def test_mlflow_logging(tmp_path):
    # Set MLflow tracking URI to a temporary sqlite database
    db_path = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{db_path}"

    exp_id = setup_experiment("test_experiment", tracking_uri=tracking_uri)
    assert exp_id is not None

    X = np.random.rand(20, 5)
    y = np.random.rand(20)

    model = train_pls_model(X, y, n_components=2)
    metrics = performance_model(model, X, y).model_dump()

    run_id = log_model_run(
        run_name="test_pls_run",
        model=model,
        params={"n_components": 2},
        metrics=metrics,
    )

    assert run_id is not None
