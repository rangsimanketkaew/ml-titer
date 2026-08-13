# Predicting Titer of of a Simulated Upstream Bioprocess

Interview tasks for ML Engineer position at DataHow

## Task

Predict the final mAb product titer (`Y:Titer`, one scalar per experiment) of a simulated fed-batch upstream bioprocess, from a mix of scalar process settings and daily time-series process data.

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

Each experiment runs for a different duration (7, 8, 9, 10 or 14 days, set by `Z:ExpDuration`). `Z:*` parameters are constant per experiment (only populated on day 0); `W:*` and `X:*` are daily time series recorded for the full duration of each run.

## Project Structure

| File | Purpose |
| --- | --- |
| [eda.ipynb](eda.ipynb) | Data structure exploration, cleaning & feature engineering, visualizations |
| [baseline.ipynb](baseline.ipynb) | Baseline model comparison (Ridge/PLS/Random Forest/Gradient Boosting) and feature importance |
| [xgboost.ipynb](xgboost.ipynb) | XGBoost model and fine-tuning |
| [test_pred.ipynb](test_pred.ipynb) | Titer prediction on the actual test set |
| [note.md](note.md) | Quick note |

## Development

#### Tech stack

- Environment: uv, ruff
- Data processing: NumPy, Pandas, Scikit-learn
- Model development: Scikit-learn, XGBoost
- DepOps: CI/CD

#### Architecture

## Author

Rangsiman Ketkaew
