from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]


def request_to_exp_dataframe(
    timestamps: list[float],
    values: dict[str, list[float]],
    exp_name: str = "inference",
) -> pd.DataFrame:
    """
    Convert raw timestamps and variable values into a long-format experiment DataFrame
    """
    rows: list[dict[str, object]] = []
    for i, day in enumerate(timestamps):
        row: dict[str, object] = {"Exp": exp_name, "Time[day]": day}
        for col, arr in values.items():
            if col.startswith("Z:") and i == 0:
                row[col] = arr[0] if arr else np.nan
            elif col.startswith("Z:"):
                row[col] = np.nan
            elif i < len(arr):
                row[col] = arr[i]
            else:
                row[col] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def spec_yml_to_dataframe(
    yaml_path: str | Path, exp_name: str = "inf_example"
) -> pd.DataFrame:
    """
    Parse the inference spec payload (yaml) into a long-format dataframe
    """
    spec_path = Path(yaml_path)
    if not spec_path.is_absolute():
        spec_path = (ROOT_DIR / spec_path).resolve()

    with spec_path.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}

    properties = (
        spec.get("components", {})
        .get("schemas", {})
        .get("PredictRequest", {})
        .get("properties", {})
    )
    timestamps = properties.get("timestamps", {}).get("example", [])
    values = properties.get("values", {}).get("example", {})

    return request_to_exp_dataframe(timestamps, values, exp_name=exp_name)


def _get_feature_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    return [col for col in df.columns if col.startswith(prefix)]


def clean_timeseries(df: pd.DataFrame, x_cols: list[str] | None = None) -> pd.DataFrame:
    """
    Clip physically impossible negative measurements to zero
    """
    df = df.copy()
    x_cols = x_cols or _get_feature_columns(df, "X:")

    if not x_cols:
        return df

    neg_counts = (df[x_cols] < 0).sum()
    negative_cols = neg_counts[neg_counts > 0].to_dict()
    if negative_cols:
        print(f"Clipping negative values found in: {negative_cols}")

    df[x_cols] = df[x_cols].clip(lower=0)
    return df


def featurize_timeseries(
    group: pd.DataFrame, cols: list[str] | None = None
) -> dict[str, float]:
    """
    Summarize one experiment's time-series columns into scalar features
    """
    t = group["Time[day]"].to_numpy(dtype=float)
    cols = cols or [col for col in group.columns if col.startswith(("W:", "X:"))]
    feats: dict[str, float] = {}

    for c in cols:
        y = group[c].to_numpy(dtype=float)
        name = c.split(":", 1)[1]
        feats[f"{name}_final"] = float(y[-1])
        feats[f"{name}_max"] = float(np.nanmax(y))
        feats[f"{name}_mean"] = float(np.nanmean(y))
        feats[f"{name}_auc"] = float(np.trapezoid(y, t)) if len(t) > 1 else 0.0
        slope = float(np.polyfit(t, y, 1)[0]) if len(t) > 1 else 0.0
        feats[f"{name}_slope"] = slope

    vcd_col = "X:VCD"
    if vcd_col not in group.columns:
        raise KeyError(f"Missing required column: {vcd_col}")

    vcd = group[vcd_col].to_numpy(dtype=float)
    vcd_initial = vcd[0] if vcd[0] > 0 else 1e-6
    duration = t[-1] if t[-1] > 0 else 1.0
    feats["VCD_growth_rate"] = float(
        np.log(max(vcd[-1], 1e-6) / vcd_initial) / duration
    )
    feats["VCD_time_to_peak"] = float(t[np.argmax(vcd)])
    return feats


def build_feature_table(
    df: pd.DataFrame,
    z_cols: list[str] | None = None,
    w_cols: list[str] | None = None,
    x_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Collapse long (Exp, day) data into one row per experiment
    """
    z_cols = z_cols or _get_feature_columns(df, "Z:")
    w_cols = w_cols or _get_feature_columns(df, "W:")
    x_cols = x_cols or _get_feature_columns(df, "X:")

    df = clean_timeseries(df, x_cols=x_cols)
    rows: list[dict[str, object]] = []

    for exp, group in df.groupby("Exp", sort=False):
        group = group.sort_values("Time[day]")
        n_days = float(group["Time[day]"].max())
        z_row = group.loc[group["Time[day]"] == 0, z_cols].iloc[0].to_dict()
        ts_feats = featurize_timeseries(group, cols=w_cols + x_cols)

        rows.append(
            {
                "Exp": exp,
                "n_days": n_days,
                **z_row,
                **ts_feats,
                "temp_shift_reached": float(z_row.get("Z:tempShift", np.nan)) <= n_days,
                "ph_shift_reached": float(z_row.get("Z:phShift", np.nan)) <= n_days,
            }
        )

    return pd.DataFrame(rows).set_index("Exp")
