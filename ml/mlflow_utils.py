import os
from typing import Any

import mlflow
import mlflow.sklearn

# Use sqlite:///mlflow.db as default database backend per MLflow 3.x recommendation
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def setup_experiment(
    experiment_name: str = "mAb_Titer_Prediction", tracking_uri: str | None = None
) -> str:
    """
    Set or create the MLflow experiment and return its ID
    """
    uri = tracking_uri or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_model_run(
    run_name: str,
    model: Any,
    params: dict[str, Any],
    metrics: dict[str, float],
    artifact_path: str = "model",
) -> str:
    """
    Log a training run and return the run_id
    """
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            serialization_format="cloudpickle",
        )
        return run.info.run_id
