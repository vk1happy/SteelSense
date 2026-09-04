from datetime import datetime
import time
from flask import Flask, jsonify, request, session, send_from_directory
from pathlib import Path

from services.prediction import load_artifacts, predict_dffn, rf_supporting_perspective
from services.cost import estimate_cost
from services.explainability import local_driver_profile
from services.scenario import verify_scenario
from services.decision_support import decision_summary

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
app = Flask(__name__, static_folder=str(FRONTEND), static_url_path="")
app.secret_key = "steelsense-final-local-session-key"
ARTIFACTS = load_artifacts()


def locked_context():
    value = session.get("forecast_context")
    if value:
        return datetime.fromisoformat(value)
    dt = datetime(2026, 8, 24, 0, 15)
    session["forecast_context"] = dt.isoformat()
    return dt


def parse_inputs(payload, target_dt):
    required = ["lagging_reactive_power", "leading_reactive_power", "co2", "power_factor", "load_type"]
    for key in required:
        if key not in payload:
            raise ValueError(f"Missing required input: {key}")
    pf = float(payload["power_factor"])
    if not 0 <= pf <= 100:
        raise ValueError("Power Factor must be between 0 and 100.")
    load = str(payload["load_type"])
    if load == "Low Load":
        load = "Light Load"
    if load not in {"Light Load", "Medium Load", "Maximum Load"}:
        raise ValueError("Invalid load type.")
    values = {
        "lagging_reactive_power": float(payload["lagging_reactive_power"]),
        "leading_reactive_power": float(payload["leading_reactive_power"]),
        "co2": float(payload["co2"]),
        "power_factor": pf,
        "load_type": load,
        "target_datetime": target_dt,
    }
    for key in ("lagging_reactive_power", "leading_reactive_power", "co2"):
        if values[key] < 0:
            raise ValueError(f"{key} cannot be negative.")
    return values


def public_inputs(v):
    return {k: v for k, v in v.items() if k != "target_datetime"}


@app.get("/")
def index():
    return send_from_directory(FRONTEND, "SteelSense_Explainability_Visual_FINAL.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "application": "SteelSense", "status": "backend-online",
                    "artifacts": sorted(ARTIFACTS.keys()),
                    "workflow": ["Forecast", "Explainability", "What-If Analysis", "Decision Support", "Final Summary"]})


@app.post("/api/forecast")
def forecast():
    try:
        payload = request.get_json(force=True) or {}
        dt = locked_context()
        inputs = parse_inputs(payload, dt)
        tariff = float(payload.get("tariff", 0))
        if tariff <= 0:
            raise ValueError("Tariff must be greater than zero.")
        dffn = float(predict_dffn(inputs, ARTIFACTS))
        explanation = local_driver_profile(inputs, ARTIFACTS)
        rf_support = rf_supporting_perspective(ARTIFACTS, inputs, explanation, top_n=4)
        analysis = {
            "dffn": dffn,
            "tariff": tariff,
            "rf_supporting_perspective": rf_support,
            "cost": estimate_cost(dffn, tariff),
            "inputs": public_inputs(inputs),
            "target_datetime": dt.isoformat(),
            "context_locked": True,
            "calculation_id": str(time.time_ns()),
        }
        session["analysis"] = analysis
        session["inputs"] = public_inputs(inputs)
        session["explanation"] = explanation
        session.pop("scenario", None)
        session.pop("ranking", None)
        session.modified = True
        public_analysis = {k: v for k, v in analysis.items() if k != "_rf_internal"}
        return jsonify({"ok": True, "analysis": public_analysis, "explainability": explanation})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.post("/api/what-if")
def what_if():
    try:
        analysis = session.get("analysis")
        raw = session.get("inputs")
        if not analysis or not raw:
            raise ValueError("Run the 8-hour forecast before What-If Analysis.")
        payload = request.get_json(force=True) or {}
        dt = datetime.fromisoformat(analysis["target_datetime"])
        tariff = float(analysis["tariff"])

        if "inputs" not in payload:
            raise ValueError("Provide the What-If operating inputs.")

        scenario_inputs = parse_inputs(payload["inputs"], dt)
        scenario = verify_scenario(
            scenario_inputs,
            dict(raw, target_datetime=dt),
            float(analysis["dffn"]),
            tariff,
            ARTIFACTS,
        )
        session["scenario"] = scenario
        session.modified = True
        return jsonify({"ok": True, "scenario": scenario})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.get("/api/state")
def state():
    analysis = session.get("analysis")
    public_analysis = None if not analysis else {k: v for k, v in analysis.items() if not k.startswith("_")}
    return jsonify({"ok": True, "analysis": public_analysis,
                    "explainability": session.get("explanation"),
                    "ranking": session.get("ranking"),
                    "scenario": session.get("scenario"),
                    "decision": decision_summary(analysis, session.get("scenario"))})


@app.get("/api/decision-support")
def decision():
    analysis = session.get("analysis")
    return jsonify({"ok": True,
                    "analysis": None if not analysis else {k: v for k, v in analysis.items() if not k.startswith("_")},
                    "explainability": session.get("explanation"),
                    "ranking": session.get("ranking"),
                    "scenario": session.get("scenario"),
                    "decision": decision_summary(analysis, session.get("scenario"))})


@app.get("/api/final-summary")
def final_summary():
    analysis = session.get("analysis")
    return jsonify({"ok": True,
                    "analysis": None if not analysis else {k: v for k, v in analysis.items() if not k.startswith("_")},
                    "explainability": session.get("explanation"),
                    "ranking": session.get("ranking"),
                    "scenario": session.get("scenario"),
                    "decision": decision_summary(analysis, session.get("scenario"))})


@app.get("/api/model-performance")
def model_performance():
    # Fixed test-set evaluation from the finalized chronological evaluation.
    # These are evaluation results, not a new user-facing forecast.
    return jsonify({
        "ok": True,
        "evaluation": {
            "test_rows": 7001,
            "dffn": {"mae": 248.8292, "rmse": 368.1558, "r2": 0.7322},
            "rf": {"mae": 244.6106, "rmse": 393.2960, "r2": 0.6944},
            "selection": "DFFN selected as the primary forecasting model based on stronger overall RMSE and R² performance."
        }
    })


@app.post("/api/reset")
def reset():
    session.clear()
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
