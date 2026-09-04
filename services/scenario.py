from services.prediction import predict_dffn
from services.cost import estimate_cost

SCENARIO_FEATURES = {"lagging_reactive_power", "leading_reactive_power", "co2", "power_factor", "load_type"}
LOAD_TYPES = {"Light Load", "Medium Load", "Maximum Load"}


def _validate_inputs(v):
    if v["load_type"] not in LOAD_TYPES:
        raise ValueError("Invalid load type.")
    if not 0 <= float(v["power_factor"]) <= 100:
        raise ValueError("Power Factor must be between 0 and 100.")
    for key in ("lagging_reactive_power", "leading_reactive_power", "co2"):
        if float(v[key]) < 0:
            raise ValueError(f"{key} cannot be negative.")


def _public(v):
    return {k: v[k] for k in ("lagging_reactive_power", "leading_reactive_power", "co2", "power_factor", "load_type")}


def verify_scenario(scenario_inputs, baseline_inputs, baseline_dffn, tariff, artifacts):
    """Evaluate one user-defined What-If condition with the primary DFFN model."""
    _validate_inputs(scenario_inputs)
    dffn = float(predict_dffn(scenario_inputs, artifacts))
    baseline_cost = float(estimate_cost(baseline_dffn, tariff))
    scenario_cost = float(estimate_cost(dffn, tariff))
    energy_change = dffn - float(baseline_dffn)
    energy_change_pct = energy_change / max(abs(float(baseline_dffn)), 1e-9) * 100.0
    return {
        "type": "dffn_what_if",
        "dffn": dffn,
        "baseline_dffn": float(baseline_dffn),
        "energy_change": float(energy_change),
        "energy_change_pct": float(energy_change_pct),
        "baseline_cost": baseline_cost,
        "scenario_cost": scenario_cost,
        "cost_change": float(scenario_cost - baseline_cost),
        "scenario_inputs": _public(scenario_inputs),
    }
