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


def log_inference_run(
    model_id: str,
    prediction: float,
    uncertainty: float,
    lower_bound: float,
    upper_bound: float,
    num_timestamps: int,
    experiment_name: str = "Live_Inference",
    tracking_uri: str | None = None,
) -> str:
    """
    Log a live inference request run to MLflow and return the run_id.
    """
    exp_id = setup_experiment(
        experiment_name=experiment_name, tracking_uri=tracking_uri
    )
    with mlflow.start_run(
        experiment_id=exp_id, run_name=f"inference_{model_id}", nested=True
    ) as run:
        # Log parameters (visible in main table view by default)
        mlflow.log_params(
            {
                "model_id": model_id,
                "num_timestamps": num_timestamps,
                "predicted_titer": f"{prediction:.2f} mg/L",
                "uncertainty": f"±{uncertainty:.2f}",
            }
        )
        # Log numerical metrics (for charts/tracking)
        mlflow.log_metrics(
            {
                "prediction": float(prediction),
                "uncertainty": float(uncertainty),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            }
        )
        # Log tag
        mlflow.set_tag("source", "live_inference_api")

        # Log artifact JSON containing full inference payload
        inference_payload = {
            "model_id": model_id,
            "prediction": float(prediction),
            "unit": "mg/L",
            "uncertainty": float(uncertainty),
            "confidence_interval": {
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
            },
            "num_timestamps": num_timestamps,
        }
        mlflow.log_dict(inference_payload, "inference_result.json")

        return run.info.run_id
