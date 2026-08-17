import pathlib

import pandas as pd
from fastapi.testclient import TestClient

from main import app
from ml.data import build_feature_table, spec_yml_to_dataframe
from spec_yml_to_json import load_runtime_request

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "inference_server_spec.yml"


def test_read_infer_spec_dataframe_creates_long_format_table() -> None:
    df = spec_yml_to_dataframe(SPEC_PATH)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Exp" in df.columns
    assert "Time[day]" in df.columns
    assert "X:VCD" in df.columns
    assert len(df) == 15


def test_build_feature_table_produces_expected_feature_values() -> None:
    raw = spec_yml_to_dataframe(SPEC_PATH)
    features = build_feature_table(raw)

    assert features.shape[0] == 1
    assert "VCD_final" in features.columns
    assert "temp_final" in features.columns
    assert "Lysed_slope" in features.columns

    expected = {
        "VCD_final": 27.40298708,
        "temp_final": 36.94736842,
    }

    for key, expected_value in expected.items():
        actual = float(features.iloc[0][key])
        assert abs(actual - expected_value) < 1e-4, (
            f"{key} mismatch: {actual} != {expected_value}"
        )


def test_predict_endpoint_returns_numeric_prediction() -> None:
    request_body = load_runtime_request(SPEC_PATH)

    client = TestClient(app)
    response = client.post("/predict", json=request_body)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert isinstance(payload["prediction"], (int, float))


def test_predict_endpoint_with_model_in_request_body() -> None:
    request_body = load_runtime_request(SPEC_PATH)
    client = TestClient(app)

    for model_id in ["mlr", "pls", "xgb"]:
        body = dict(request_body, model=model_id)
        response = client.post("/predict", json=body)
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["status"] == "success"
        assert "prediction" in payload
        assert "model_info" in payload


def test_predict_endpoint_with_invalid_model_returns_400() -> None:
    request_body = load_runtime_request(SPEC_PATH)
    client = TestClient(app)

    body = dict(request_body, model="invalid")
    response = client.post("/predict", json=body)
    assert response.status_code == 400
