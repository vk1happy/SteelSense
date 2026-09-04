import numpy as np
from services.prediction import build_input, _numpy_dffn
import joblib


def _dffn_value(v, artifacts):
    features = joblib.load(artifacts["features"])
    scaler = joblib.load(artifacts["scaler"])
    X = build_input(v, features)
    return _numpy_dffn(artifacts["model"], scaler.transform(X))


def local_driver_profile(inputs, artifacts):
    # Local sensitivity is deliberately model-based, not causal.
    specs = [
        ("Lagging Reactive Power", "lagging_reactive_power", "numeric", 0.05),
        ("Leading Reactive Power", "leading_reactive_power", "numeric", 0.05),
        ("CO₂ Emission", "co2", "numeric", 0.10),
        ("Power Factor", "power_factor", "numeric", 0.02),
    ]
    rows = []
    base = _dffn_value(inputs, artifacts)
    for label, key, kind, frac in specs:
        old = float(inputs[key])
        delta = max(abs(old) * frac, 0.01)
        plus = dict(inputs); minus = dict(inputs)
        plus[key] = old + delta
        minus[key] = max(0.0, old - delta) if key != "power_factor" else max(0.0, old - delta)
        p = _dffn_value(plus, artifacts)
        m = _dffn_value(minus, artifacts)
        sensitivity = (p - m) / 2.0
        rows.append({
            "feature": label,
            "contribution": float(sensitivity),
            "absolute_contribution": float(abs(sensitivity)),
            "baseline": old,
            "delta": delta,
            "plus_prediction": p,
            "minus_prediction": m,
        })

    # Add current load effect as a model counterfactual against the two other
    # load states, averaged to avoid pretending it is a causal effect.
    load = inputs["load_type"]
    alternatives = [x for x in ["Light Load", "Medium Load", "Maximum Load"] if x != load]
    alt_preds = []
    for alt in alternatives:
        v = dict(inputs); v["load_type"] = alt
        alt_preds.append(_dffn_value(v, artifacts))
    if alt_preds:
        load_effect = base - float(np.mean(alt_preds))
        rows.append({
            "feature": "Load Type",
            "contribution": float(load_effect),
            "absolute_contribution": float(abs(load_effect)),
            "baseline": load,
            "delta": None,
            "plus_prediction": None,
            "minus_prediction": None,
        })

    rows.sort(key=lambda x: x["absolute_contribution"], reverse=True)
    return rows
