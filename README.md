# Predicting Titer of of a Simulated Upstream Bioprocess

Interview tasks for ML Engineer position at DataHow

## Task

Predict the final mAb product titer of a simulated fed-batch upstream bioprocess, from a mix of scalar process settings and daily time-series process data.

## How I Tackle the Challenge

1. Data analysis to understand the characteristics of the raw data
2. Cleaning and feature engineering
3. Develop baseline models based on the nature of the data
4. Hyperparameter optimization and fine-tune models
5. Design ML pipeline/architecture
6. Containerizing ML deployment with Docker and FastAPI
7. Maintenance, linting and format checking, documentation

---

## Dataset and Nomenclature

| File                                          | Content                                                                                  |
| --------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `datahow_interview_train_data.csv`            | Training inputs, long format (one row per experiment × day), 100 experiments / 990 rows. |
| `datahow_interview_train_targets.csv`         | Training targets, one row per experiment.                                                |
| `datahow_interview_test_data.csv`             | Test inputs (same schema), 20 experiments / 300 rows.                                    |
| `datahow_interview_test_targets-TEMPLATE.csv` | Submission placeholder (dummy `2000` values).                                            |

- `Z:` scalar parameters
- `W:` control profiles
- `X:` measured observations
- `Y:Titer` target (one scalar per experiment)

Each experiment runs for a different duration (7, 8, 9, 10 or 14 days, set by `Z:ExpDuration`). `Z:*` parameters are constant per experiment (only populated on day 0); `W:*` and `X:*` are daily time series recorded for the full duration of each run.

## Pipeline Architecture

1. **Raw data validation**: Validate data quality
2. **Data preparation**: Clean and preprocess data
3. **Data transformation**: Feature engineering
4. **Feature storage**: Manage engineered features
5. **Data versioning**: DVC-based version control
6. **Baseline model training**: Train baseline model
7. **Model deployment**: Containerize and deploy model with microservice
8. **Model versioning**: Track metrics and version models
9. **Log storage**: Log model performance

## Project Structure

| File | Purpose |
| --- | --- |
| [eda.ipynb](eda.ipynb) | Data structure exploration, cleaning & feature engineering, visualizations |
| [baseline.ipynb](baseline.ipynb) | Baseline model comparison (Ridge/PLS/Random Forest/Gradient Boosting) and feature importance |
| [xgboost.ipynb](xgboost.ipynb) | XGBoost model and fine-tuning |
| [test_pred.ipynb](test_pred.ipynb) | Titer prediction on the actual test set |
| [note.md](note.md) | Quick note |

## Get Started

Automated setup
```sh
git clone <repository-url>
cd datahow-titer-ml
python -m venv .venv
source .venv/bin/activate
pip install uv
uv sync
python main.py
```

## Development

#### Tech stack

- Environment: uv, ruff
- Data processing: NumPy, Pandas, Scikit-learn
- Model development: Scikit-learn, XGBoost
- Inference microservice: Docker, FastAPI, pyyaml
- DepOps: CI/CD, pydantic

#### Architecture

## Documentation

## Author

Rangsiman Ketkaew
