from pathlib import Path
import io
import json
import zipfile
import joblib
import numpy as np
import pandas as pd
import h5py

BASE = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE / "models"

FEATURE_FILES = {
    "features": "dffn_feature_columns.pkl",
    "rf_features": "rf_feature_columns.pkl",
    "scaler": "final_dffn_scaler.pkl",
    "rf": "final_rf.pkl",
    "model": "final_dffn.keras",
}

EXPECTED_FEATURES = [
    "Lagging_Current_Reactive_Power_kVarh_lag1",
    "Leading_Current_Reactive_Power_kVarh_lag4",
    "CO2_tCO2_lag1",
    "Leading_Current_Power_Factor_lag4",
    "NSM_lag4",
    "Target_Month",
    "Target_Hour",
    "Target_Minute",
    "Target_WeekStatus_Weekend",
    "Target_Day_of_week_Monday",
    "Target_Day_of_week_Saturday",
    "Target_Day_of_week_Sunday",
    "Target_Day_of_week_Thursday",
    "Target_Day_of_week_Tuesday",
    "Target_Day_of_week_Wednesday",
    "Target_Load_Type_Maximum_Load",
    "Target_Load_Type_Medium_Load",
]


def load_artifacts():
    found = {}
    for key, name in FEATURE_FILES.items():
        p = MODEL_DIR / name
        if not p.exists():
            raise RuntimeError(f"Missing model artifact: {name}")
        found[key] = p
    features = joblib.load(found["features"])
    rf_features = joblib.load(found["rf_features"])
    if list(features) != EXPECTED_FEATURES:
        raise RuntimeError("DFFN feature schema does not match the finalized 17-feature schema.")
    if list(rf_features) != EXPECTED_FEATURES:
        raise RuntimeError("RF feature schema does not match the finalized 17-feature schema.")
    return found


def build_input(v, feature_columns):
    x = pd.DataFrame(0.0, index=[0], columns=feature_columns, dtype=float)
    mapping = {
        "lagging_reactive_power": "Lagging_Current_Reactive_Power_kVarh_lag1",
        "leading_reactive_power": "Leading_Current_Reactive_Power_kVarh_lag4",
        "co2": "CO2_tCO2_lag1",
        "power_factor": "Leading_Current_Power_Factor_lag4",
    }
    for source, col in mapping.items():
        if source not in v:
            raise ValueError(f"Missing input: {source}")
        if col not in x.columns:
            raise RuntimeError(f"Required trained feature missing: {col}")
        x.loc[0, col] = float(v[source])

    target_dt = pd.Timestamp(v["target_datetime"])
    lag4_dt = target_dt - pd.Timedelta(minutes=60)
    x.loc[0, "NSM_lag4"] = lag4_dt.hour * 3600 + lag4_dt.minute * 60 + lag4_dt.second
    x.loc[0, "Target_Month"] = target_dt.month
    x.loc[0, "Target_Hour"] = target_dt.hour
    x.loc[0, "Target_Minute"] = target_dt.minute

    if target_dt.weekday() >= 5 and "Target_WeekStatus_Weekend" in x.columns:
        x.loc[0, "Target_WeekStatus_Weekend"] = 1.0
    day = f"Target_Day_of_week_{target_dt.day_name()}"
    if day in x.columns:
        x.loc[0, day] = 1.0

    load = str(v["load_type"])
    if load == "Medium Load":
        x.loc[0, "Target_Load_Type_Medium_Load"] = 1.0
    elif load == "Maximum Load":
        x.loc[0, "Target_Load_Type_Maximum_Load"] = 1.0
    elif load != "Light Load":
        raise ValueError("load_type must be Light Load, Medium Load, or Maximum Load")

    if list(feature_columns) != EXPECTED_FEATURES:
        raise RuntimeError("Unexpected feature order")
    return x.loc[:, feature_columns]


def _numpy_dffn(model_path, x_scaled):
    with zipfile.ZipFile(model_path) as z:
        config = json.loads(z.read("config.json").decode("utf-8"))
        weights_blob = z.read("model.weights.h5")
    dense = [l for l in config["config"]["layers"] if l.get("class_name") == "Dense"]
    if len(dense) != 4:
        raise RuntimeError("Saved DFFN architecture is not the expected 4-Dense network.")
    with h5py.File(io.BytesIO(weights_blob), "r") as f:
        W, b = [], []
        for idx in range(4):
            group = "layers/dense" if idx == 0 else f"layers/dense_{idx}"
            W.append(np.asarray(f[f"{group}/vars/0"][:], dtype=np.float32))
            b.append(np.asarray(f[f"{group}/vars/1"][:], dtype=np.float32))
    a = np.asarray(x_scaled, dtype=np.float32)
    for i in range(3):
        a = np.maximum(a @ W[i] + b[i], 0)
    return float((a @ W[3] + b[3]).reshape(-1)[0])


def predict_dffn(v, artifacts):
    features = joblib.load(artifacts["features"])
    scaler = joblib.load(artifacts["scaler"])
    X = build_input(v, features)
    value = _numpy_dffn(artifacts["model"], scaler.transform(X))
    if not np.isfinite(value) or value < 0:
        raise RuntimeError(f"DFFN returned invalid prediction: {value}")
    return value


def predict_rf(v, artifacts):
    features = joblib.load(artifacts["rf_features"])
    model = joblib.load(artifacts["rf"])
    X = build_input(v, features)
    value = float(np.asarray(model.predict(X)).reshape(-1)[0])
    if not np.isfinite(value) or value < 0:
        raise RuntimeError(f"RF returned invalid prediction: {value}")
    return value


def compare_models(dffn, rf):
    diff = float(rf - dffn)
    pct = abs(diff) / max(abs(dffn), 1e-9) * 100
    if pct <= 5:
        status = "Models closely aligned"
    elif pct <= 15:
        status = "Models show moderate difference"
    else:
        status = "Models show material difference"
    return {"difference": diff, "difference_pct": pct, "status": status}


def rf_supporting_perspective(artifacts, inputs, dffn_rows, top_n=4):
    """Return local RF sensitivity for the drivers highlighted by the DFFN.

    The DFFN determines which current-case drivers are highlighted. For those
    same drivers, the RF model is evaluated locally by perturbing the current
    input and measuring the resulting change in the RF prediction. The values
    are normalized only across the highlighted drivers. This is a local model
    sensitivity view, not global RF feature importance, a causal effect, or a
    second energy forecast.
    """
    rf_features = list(joblib.load(artifacts["rf_features"]))
    model = joblib.load(artifacts["rf"])

    def rf_value(v):
        X = build_input(v, rf_features)
        value = float(np.asarray(model.predict(X)).reshape(-1)[0])
        if not np.isfinite(value):
            raise RuntimeError("RF returned an invalid local sensitivity prediction.")
        return value

    base = rf_value(inputs)

    feature_map = {
        "Lagging Reactive Power": "lagging_reactive_power",
        "Leading Reactive Power": "leading_reactive_power",
        "CO₂ Emission": "co2",
        "Power Factor": "power_factor",
    }

    selected = []
    seen = set()
    for row in (dffn_rows or [])[:top_n]:
        label = str(row.get("feature", ""))
        key = label.replace(" · lag1", "").replace(" · lag4", "")
        if key in seen:
            continue

        # Continuous drivers use the same perturbation scale as the DFFN
        # local explanation so the two model perspectives are comparable.
        input_key = feature_map.get(key)
        if input_key is not None:
            old = float(inputs[input_key])
            frac = {
                "lagging_reactive_power": 0.05,
                "leading_reactive_power": 0.05,
                "co2": 0.10,
                "power_factor": 0.02,
            }[input_key]
            delta = max(abs(old) * frac, 0.01)
            plus = dict(inputs)
            minus = dict(inputs)
            plus[input_key] = old + delta
            minus[input_key] = max(0.0, old - delta)
            p = rf_value(plus)
            m = rf_value(minus)
            sensitivity = (p - m) / 2.0
            selected.append({
                "feature": label,
                "rf_local_sensitivity": float(sensitivity),
                "rf_importance": float(abs(sensitivity)),
                "rf_importance_pct": 0.0,
                "dffn_contribution": float(row.get("contribution", 0.0)),
                "baseline_rf_prediction": base,
            })
            seen.add(key)
            continue

        # Load Type is categorical. Mirror the DFFN local counterfactual:
        # compare the current load with the mean prediction of the two other
        # load states. This keeps the RF view local to the current prediction.
        if key == "Load Type":
            load = str(inputs["load_type"])
            alternatives = [
                x for x in ["Light Load", "Medium Load", "Maximum Load"]
                if x != load
            ]
            alt_preds = []
            for alt in alternatives:
                v = dict(inputs)
                v["load_type"] = alt
                alt_preds.append(rf_value(v))
            if alt_preds:
                sensitivity = base - float(np.mean(alt_preds))
                selected.append({
                    "feature": label,
                    "rf_local_sensitivity": float(sensitivity),
                    "rf_importance": float(abs(sensitivity)),
                    "rf_importance_pct": 0.0,
                    "dffn_contribution": float(row.get("contribution", 0.0)),
                    "baseline_rf_prediction": base,
                })
                seen.add(key)

    total = float(sum(x["rf_importance"] for x in selected))
    if total > 0:
        for item in selected:
            item["rf_importance_pct"] = float(item["rf_importance"] / total * 100.0)
    return selected

