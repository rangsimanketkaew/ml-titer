import numpy as np
import pandas as pd
import yaml


def read_spec_example_as_dataframe(
    yaml_path: str, exp_name: str = "inf_example"
) -> pd.DataFrame:
    """Parse data example from the inference server spec into a long-format dataframe
    (Exp, Time[day], Z:*/W:*/X:*) matching train_data's schema.
    """
    with open(yaml_path) as f:
        spec = yaml.safe_load(f)

    props = spec["components"]["schemas"]["PredictRequest"]["properties"]
    timestamps = props["timestamps"]["example"]
    values = props["values"]["example"]

    rows = []
    for i, day in enumerate(timestamps):
        row = {"Exp": exp_name, "Time[day]": day}
        for col, arr in values.items():
            # Z: scalars only populated on day 0, matching train_data's ragged format
            row[col] = (
                arr[0]
                if (col.startswith("Z:") and i == 0)
                else (np.nan if col.startswith("Z:") else arr[i])
            )
        rows.append(row)

    return pd.DataFrame(rows)


inference_data = read_spec_example_as_dataframe("inference_server_spec.yml")
print(inference_data.head(20))

print("inference_data:", inference_data.shape)

duration_train = inference_data.groupby("Exp")["Time[day]"].max()

# print(duration_train)

Z_COLS = [c for c in inference_data.columns if c.startswith("Z:")]
W_COLS = [c for c in inference_data.columns if c.startswith("W:")]
X_COLS = [c for c in inference_data.columns if c.startswith("X:")]


def clean_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """Clip physically-impossible negative concentrations/counts to 0 (plan step B)."""
    df = df.copy()
    neg_counts = (df[X_COLS] < 0).sum()
    if neg_counts.any():
        print(
            "Clipping negative values found in:",
            neg_counts[neg_counts > 0].to_dict(),
        )
    df[X_COLS] = df[X_COLS].clip(lower=0)
    return df


def featurize_timeseries(group: pd.DataFrame, cols: list[str]) -> dict:
    """Summarize one experiment's time-series columns into scalar features."""
    t = group["Time[day]"].to_numpy(dtype=float)
    feats = {}
    for c in cols:
        y = group[c].to_numpy(dtype=float)
        name = c.split(":", 1)[1]
        feats[f"{name}_final"] = y[-1]
        feats[f"{name}_max"] = np.nanmax(y)
        feats[f"{name}_mean"] = np.nanmean(y)
        feats[f"{name}_auc"] = np.trapezoid(y, t) if len(t) > 1 else 0.0
        slope = np.polyfit(t, y, 1)[0] if len(t) > 1 else 0.0
        feats[f"{name}_slope"] = slope

    # domain-informed features (plan step C.3), derived from VCD growth dynamics
    vcd = group["X:VCD"].to_numpy(dtype=float)
    vcd_initial = vcd[0] if vcd[0] > 0 else 1e-6
    duration = t[-1] if t[-1] > 0 else 1.0
    feats["VCD_growth_rate"] = np.log(max(vcd[-1], 1e-6) / vcd_initial) / duration
    feats["VCD_time_to_peak"] = t[np.argmax(vcd)]
    return feats


def build_experiment_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse long (Exp, day) data into one row per experiment."""
    df = clean_timeseries(df)
    rows = []
    for exp, group in df.groupby("Exp", sort=False):
        group = group.sort_values("Time[day]")
        n_days = group["Time[day]"].max()
        z_row = group.loc[group["Time[day]"] == 0, Z_COLS].iloc[0].to_dict()
        ts_feats = featurize_timeseries(group, W_COLS + X_COLS)
        rows.append(
            {
                "Exp": exp,
                "n_days": n_days,
                **z_row,
                **ts_feats,
                # whether the scheduled Z:*Shift day was actually reached, so Z:*End is realized rather than just a target setpoint
                "temp_shift_reached": z_row["Z:tempShift"] <= n_days,
                "ph_shift_reached": z_row["Z:phShift"] <= n_days,
            }
        )
    return pd.DataFrame(rows).set_index("Exp")


inference_exp = build_experiment_table(inference_data)

print(inference_exp.head(20))
