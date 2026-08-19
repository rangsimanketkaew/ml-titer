from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from ml.mlflow_utils import log_model_run, setup_experiment
from ml.model import (
    performance_model,
    save_model,
    train_mlr_model,
    train_pls_model,
    train_xgb_model,
)


class TrainingConfig(BaseModel):
    data_path: Path = Path("data/train_exp_features.csv")
    models_dir: Path = Path("models")
    target_col: str = "Y:Titer"
    exclude_cols: tuple[str, ...] = ("Y:Titer",)
    pls_n_components: int = 5
    confidence_level: float = 0.95
    experiment_name: str = "mAb_Titer_Prediction"


def main():
    config = TrainingConfig()
    config.models_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(config.data_path, index_col="Exp")
    feature_cols = [col for col in train_df.columns if col not in config.exclude_cols]

    X = train_df[feature_cols].to_numpy()
    y = train_df[config.target_col].to_numpy()

    setup_experiment(config.experiment_name)
    print(f"Training on {len(feature_cols)} features...")

    models = {
        "MLR_Model": (
            train_mlr_model(X, y, confidence_level=config.confidence_level),
            "mlr_model.joblib",
            {
                "algorithm": "MLR",
                "n_features": len(feature_cols),
                "confidence_level": config.confidence_level,
            },
        ),
        "PLS_Model": (
            train_pls_model(
                X,
                y,
                n_components=config.pls_n_components,
                confidence_level=config.confidence_level,
            ),
            "pls_model.joblib",
            {
                "algorithm": "PLS",
                "n_components": config.pls_n_components,
                "n_features": len(feature_cols),
                "confidence_level": config.confidence_level,
            },
        ),
        "XGB_Model": (
            train_xgb_model(X, y, confidence_level=config.confidence_level),
            "xgb_model.joblib",
            {
                "algorithm": "XGBoost",
                "n_estimators": 200,
                "max_depth": 3,
                "learning_rate": 0.05,
                "n_features": len(feature_cols),
                "confidence_level": config.confidence_level,
            },
        ),
    }

    for name, (model, filename, params) in models.items():
        metrics = performance_model(model, X, y).model_dump()
        print(f"{name} metrics:", metrics)

        log_model_run(
            run_name=name,
            model=model,
            params=params,
            metrics=metrics,
        )

        save_model(model, config.models_dir / filename)

    print("Models saved and logged to MLflow successfully.")


if __name__ == "__main__":
    main()
