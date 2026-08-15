from pathlib import Path

import pandas as pd
from model import performance_model, save_model, train_mlr_model, train_xgb_model
from pydantic import BaseModel


class TrainingConfig(BaseModel):
    data_path: Path = Path("../data/train_exp_features.csv")
    models_dir: Path = Path("../models")
    target_col: str = "Y:Titer"
    exclude_cols: tuple[str, ...] = ("Y:Titer", "n_days")


def main():
    config = TrainingConfig()
    config.models_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(config.data_path, index_col="Exp")
    feature_cols = [col for col in train_df.columns if col not in config.exclude_cols]

    X = train_df[feature_cols].to_numpy()
    y = train_df[config.target_col].to_numpy()

    mlr_model = train_mlr_model(X, y)
    xgb_model = train_xgb_model(X, y)

    print("MLR metrics:", performance_model(mlr_model, X, y).model_dump())
    print("XGB metrics:", performance_model(xgb_model, X, y).model_dump())

    save_model(mlr_model, config.models_dir / "mlr_model.joblib")
    save_model(xgb_model, config.models_dir / "xgb_model.joblib")


if __name__ == "__main__":
    main()
