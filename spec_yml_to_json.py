from __future__ import annotations

import json
import pathlib
import yaml

SPEC_PATH = pathlib.Path("inference_server_spec.yml")


def load_runtime_request(spec_path: str | pathlib.Path = SPEC_PATH) -> dict:
    """
    Read OpenAPI yaml and return the PredictRequest payload in JSON format
    """
    with pathlib.Path(spec_path).open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}

    request_schema = (
        spec.get("components", {}).get("schemas", {}).get("PredictRequest", {})
    )
    properties = request_schema.get("properties", {})

    return {
        "timestamps": properties.get("timestamps", {}).get("example", []),
        "values": properties.get("values", {}).get("example", {}),
    }


if __name__ == "__main__":
    payload = load_runtime_request(SPEC_PATH)
    print(json.dumps(payload, indent=2))
