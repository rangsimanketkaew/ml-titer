# Predicting Titer of of a Simulated Upstream Bioprocess

Interview tasks for ML Engineer position at DataHow

## Task

Predict the final mAb product titer of a simulated fed-batch upstream bioprocess, from a mix of scalar process settings and daily time-series process data.


## Pipeline Architecture

1. **Raw data validation**: Validate data quality
2. **Data preparation**: Clean and preprocess data
3. **Data transformation**: Feature engineering
4. **Feature storage**: Manage engineered features
5. **Data versioning**: DVC-based version control
6. **Baseline model training**: Train baseline model
7. **Model deployment**: Containerize and deploy model with Uvicorn/Docker
8. **Model versioning**: Track metrics and version models
9. **Log storage**: Log model performance

## Project Structure

```
├── data
│   ├── *.csv                     # Dataset files
├── Dockerfile                    # Docker file
├── inference_server_spec.yml     # Example inference spec yaml
├── main.py                       # App microservice
├── ml
│   ├── data.py                   # Helper functions for data processing
│   ├── model.py                  # Helper functions for ML
│   └── train_model.py            # Train ML models
├── models
│   ├── *.joblib                  # Pretrained models
├── notebook
│   ├── baseline.ipynb            # Baseline model
│   ├── eda.ipynb                 # Data exploration, cleaning & feature engineering, visualizations
│   └── test_template.ipynb       # Test model prediction
├── pyproject.toml                # Project configuration
├── README.md                     # This file
├── spec_yml_to_json.py           # Convert inference server yml to JSON file
├── tests
│   └── test_pipeline.py          # Pytest functions
└── uv.lock                       # uv configuration
```

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

**See [eda.ipynb](./notebook/eda.ipynb) for exploratory data analysis (EDA)**

1. Very small sample size: 
   - 100 experiments for training (20 held out as the final test set).
2. Mixed data types: 
   - static scalar process settings (`Z:*`), time-varying control profiles (`W:*`), and time-varying measurements (`X:*`) of different, ragged lengths (7-14 days).
3. Strong multicollinearity: 
   - `Z:*` parameters are generated from a designed experiment (feed start/end, pH/temp start-end-shift are linked)
   - Time-series summary features (final/max/mean/AUC of the same variable) are highly correlated with each other.
4. Single scalar target per experiment: 
   - Titer is measured only at the end, i.e., this is a batch-to-scalar regression problem, not a sequence-to-sequence forecasting problem.
5. Underlying process is *nonlinear* 
   - E.g. microbial/cell growth kinetics, feed-limited dynamics, saturation effects

## How I Tackle the Challenge

1. Data analysis to understand the characteristics of the raw data
2. Cleaning and feature engineering
3. Develop baseline models based on the nature of the data
4. Hyperparameter optimization and fine-tune models
5. Design ML pipeline/architecture
6. Containerizing ML deployment with Docker and FastAPI
7. Maintenance, linting and format checking, documentation

## Challenges of the Training and Test sets

- Experiment runs for a different duration (7, 8, 9, or 14 days, set by `Z:ExpDuration`, starting from day 0), while in test set the experiment is recorded for 14 days. This is the experiment duration mismatch. Fortunetely, the goal of this task is to **predict only the titer at the final/day-14 timepoint**, with full days 0–14 already given in test set, and it is not a titer-per-day prediction.
- Even though zero-padding technique to make experiment 14-day rows is tempting here because it will make training set shape matches the test set shape, but it can cause more problems; e.g. it creates spurious features and zeroes have no meaningful physical meaning. Therefore, I decided to go for **one row per experiment** by compressing an entire experiment feature into a single summary.
- Small amount of training samples (N=100) could make model prone to overfitting. I frist start with different models and evaluating them using cross-validation technique: linear/optimization-based models like PLS/regularized linear models, multi-linear regression (MLR), tree-based models like Random Forest/Gradient Boosting, XGBoost, and probabilistic-based models like Gaussian Process (GP).

**The guideline on selecting models**

1. Start with PLS regression on the engineered per-experiment feature table as an interpretable benchmark, tuning the number of latent components by cross-validated R^2/MAE.
2. Use Gradient Boosting / Random Forest as the primary predictive model, because it captures nonlinearity without overfitting on 100 samples when combined with shallow trees, few boosting rounds/strong shrinkage, and rigorous CV (repeated K-fold given the small N).
3. Use physically meaningful features: AUC/final-value summaries and don't use dozens of redundant statistics.
4. Use PCA to reduce dimensionality of correlated features and give the most important features.
5. Consider a Gaussian Process because it is a uncertainty-aware model.  
6. Consider a hybrid mechanistic + ML model - it is the most scientific option for a process (espeically for bioprocessing) with known cell-growth kinetics.
7. We should use deep learning (e.g. LSTM or GRU) when we have more training data, or at least use a small model architecture.

See [baseline.ipynb](./notebook/baseline.ipynb) for baseline model and [test_template.ipynb](./notebook/test_template.ipynb) for test template.

## What Baseline Models Actually Showed

- With the comparison of 7 regressors, cross-validated R2 now spans 0.45-0.83. MLR (unregularized OLS on all 67 engineered features) is the best performer (R2 = 0.83), followed by Gradient Boosting (0.79), Ridge (0.78), PLS (0.75), Random Forest (0.74), XGBoost (0.72), MLR with top 5 features (0.48) and LR (0.45).
- The LR (using only `VCD_time_to_peak` feature) underperforms the multivariate models (R2 = 0.45, std = 0.31) - confirming that no single summary statistic is sufficient and that combining multiple engineered features helps.
- Despite the strong feature collinearity (see the heatmap in [eda.ipynb](./notebook/eda.ipynb)), unregularized OLS still comes out on top on cross-validated R2. Collinearity mainly inflates the variance/instability of individual coefficient estimates rather than necessarily hurting held-out predictive accuracy.
- The domain-informed feature `VCD_time_to_peak` (day at which viable cell density peaks) turned out to be the single strongest correlate (r = 0.75), which tracks titer better than any generic summary statistic. Cumulative glucose feed (`FeedGlc_auc`) and cumulative viable cell density (`VCD_auc`) remain strong predictors, followed by lactate AUC.
- `Z:ExpDuration` alone already explains a large share of variance (0.62). That means longer runs simply accumulate more product (aMb Titer), but it does not guarantee the best product titer (longer duration is not always => higher titer").

## App for Inference

- **OpenAPI YAML vs. JSON DTO**: Sample inference input data is provided as an example in an OpenAPI spec file (`inference_server_spec.yml`), whereas the FastAPI server requires a JSON payload matching Pydantic DTO. Parsing YAML file on every server request introduces unnecessary disk I/O and heavy parsing overhead on the API server.
- To solve this problem, I use a separate script `spec_yml_to_json.py` to extract the sample experiment payload into a JSON file (`payload.json`). So clients can send standard JSON POST requests at runtime directly to `/predict`, keeping the microservice fast. In addition, I also implemented a `/predict/file` endpoint in `feat/yaml-file-prediction` branch as alternative, which allows clients to upload `.yml` files directly. This endpoint uses FastAPI's `UploadFile` to receive `.yml` files and parse them through the Pydantic DTO.

## Get Started

### Environment setup
```sh
git clone <repository-url>
cd datahow-titer-ml
python -m venv .venv
source .venv/bin/activate
pip install uv
uv sync
```

### Launching API Service

An endpoint on the App server using FastAPI framework handles the prediction requests and returns the value predicted by the deployed ML pipeline. The endpoint is server/predict with a **POST** operation.

Uvicorn Server uses the API to serve the prediction requests. 

#### Serve model server with Uvicorn (native)
```sh
# Start microservice
uv run uvicorn main:app --host 0.0.0.0 --port 8000

# Check if server is healthy
curl -X GET http://0.0.0.0:8000/health

# Make inference request
uv run python spec_yml_to_json.py > payload.json
curl -X POST http://0.0.0.0:8000/predict \
    -H "Content-Type: application/json" \
    --data @payload.json
```

We can also dockerize this server, and final predictions will be served by the Docker container. The file `Dockerfile` contains all the instructions required to build the Docker image.

#### Serve model server with Docker
```sh
# Build image
docker build -t datahow-titer-ml .

# Run container
docker run --rm -p 8000:8000 datahow-titer-ml

# Health check
curl -X GET http://localhost:8000/health
```

## Development

**Tech stack**
- **Environment**: Python 3.11-3.13, uv, ruff
- **Data processing**: NumPy, Pandas, Scikit-learn
- **Model development**: Scikit-learn, XGBoost
- **Inference microservice**: Docker, FastAPI, pyyaml
- **DepOps**: CI/CD, pydantic

## Author

Rangsiman Ketkaew
