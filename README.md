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

## Understanding the Dataset

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

### Obstacles

- Variables (input features) are provided in inference spec yaml file, which makes it difficult to use these variables for model inference. Variables should be provided at runtime API request, to increase maintainability and avoid server's heavy load.

### Improvement

- Create separate repository: (1) data processing and model training, and (2) ML pipeline deployment

## Model Selection

This guideline might be useful for further collaborator/team members.

1. Start with PLS regression on the engineered per-experiment feature table (as built in Section 2) as an interpretable benchmark, tuning the number of latent components by cross-validated R^2/MAE.

2. Use Gradient Boosting / Random Forest as the primary predictive model, because it captures nonlinearity without overfitting on 100 samples when combined with shallow trees, few boosting rounds/strong shrinkage, and rigorous CV (repeated K-fold given the small N).

3. Tune and select via nested/repeated cross-validation (not a single train/validation split) because with 99 samples a single split has high variance; report CV mean +- std, not a point estimate.

4. Keep the feature set compact and physically meaningful: prefer AUC/final-value summaries over dozens of redundant statistics, or feed the correlated features through PLS/PCA to reduce dimensionality before the tree model if the feature-to-sample ratio becomes a concern.

5. Consider a Gaussian Process: because it is a complementary uncertainty-aware model, 
   
6. Consider a hybrid mechanistic + ML model - it is the most scientific option for a process with known cell-growth kinetics and is the technique that most bioprocess-modeling teams take once black-box performance plateaus.

7. We should use deep learning when we have more experiment data available, or use a small model architecture e.g., a small 1D-CNN/GRU for feature extraction from the time series, feeding into a simple regressor head.

## Model Comparison

| Model | Fit for this dataset | Reasons |
|---|---|---|
| **PLS / PCR / regularized linear regression** | Recommended as the primary/benchmark model | PLS is the standard in bioprocess chemometrics for this setup: small N, collinear regressors, batch time-series unfolded into scalar summaries. It handles multicollinearity natively via latent variables, is highly interpretable (loadings show which process phases/variables drive titer), and is very hard to overfit with only a handful of latent components on 99 samples. |
| **Random Forest / Gradient Boosting / XGBoost** | Recommended as the primary predictive model | Tree ensembles capture the nonlinearities and interactions (e.g., feed rate x duration) that a linear/PLS model misses, remain fairly robust with small N (via bagging/shrinkage + shallow trees + strong CV), are insensitive to collinearity, and directly provide feature importance. |
| **Gaussian Process Regression** | Worth trying as a secondary model | Can be used with small-N regression; gives predictive uncertainty, which is valuable when the model will inform experiment design/process optimization decisions (e.g., "how confident are we in this titer prediction?"). |
| **Deep learning (LSTM/GRU/1D-CNN/Transformer)** | Not recommended given small amount of data volume | With 99 training experiments, a sequence model has far more parameters than data points and will overfit badly. But, it might be used within a *hybrid model* that couples a mechanistic ODE (Monod-type growth/substrate-consumption kinetics) with a small neural correction term. |

---

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


```
├── data
│   ├── *.csv               # Dataset files
├── Dockerfile              # Docker file
├── inference_server_spec.yml     # Example inference spec yaml
├── main.py                 # App microservice
├── ml
│   ├── data.py             # Helper functions for data processing
│   └── model.py            # Helper functions for ML
├── models
│   ├── *.joblib            # Pretrained models
├── notebook
│   ├── baseline.ipynb      # Baseline model
│   ├── eda.ipynb           # Data exploration, cleaning & feature engineering, visualizations
│   └── test_template.ipynb     # Test model prediction
├── pyproject.toml          # Project configuration
├── README.md               # This file
├── spec_yml_to_json.py     # Convert inference server yml to JSON file
├── tests
│   └── test_pipeline.py    # Pytest functions
├── train_model.py          # Train models
└── uv.lock                 # uv configuration
```

## Get Started

**1. Environment setup**
```sh
git clone <repository-url>
cd datahow-titer-ml
python -m venv .venv
source .venv/bin/activate
pip install uv
uv sync
```

**2.1 Deploy model with FastAPI (native)**
```sh
# Start microservice
uv run uvicorn main:app --host 0.0.0.0 --port 8000

# Check status
curl -X GET http://0.0.0.0:8000/health

# Call inference endpoint
uv run python spec_yml_to_json.py > payload.json
curl -X POST http://0.0.0.0:8000/predict \
    -H "Content-Type: application/json" \
    --data @payload.json
```

**2.2 Deploy model with Docker**
```sh
# Build image
docker build -t datahow-titer-ml .

# Run container
docker run --rm -p 8000:8000 datahow-titer-ml

# Health check
curl -X GET http://localhost:8000/health
```


## Development

#### Tech stack

- Environment: uv, ruff
- Data processing: NumPy, Pandas, Scikit-learn
- Model development: Scikit-learn, XGBoost
- Inference microservice: Docker, FastAPI, pyyaml
- DepOps: CI/CD, pydantic

#### Architecture

## Author

Rangsiman Ketkaew
