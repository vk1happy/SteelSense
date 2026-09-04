def decision_summary(analysis, scenario):
    if not analysis:
        return {
            "status": "Awaiting forecast",
            "title": "Run the 8-hour forecast first.",
            "message": "Decision Support uses the latest DFFN forecast and the latest evaluated What-If scenario."
        }
    if scenario:
        change = float(scenario["energy_change_pct"])
        cost_change = float(scenario["cost_change"])
        if change >= 10:
            status, title = "High pressure", "Scenario indicates materially higher energy demand."
        elif change >= 5:
            status, title = "Review required", "Scenario indicates moderately higher energy demand."
        elif change <= -5:
            status, title = "Potential reduction", "Scenario indicates lower predicted energy demand."
        else:
            status, title = "Within control", "Scenario impact is relatively limited."
        message = (
            f"Baseline DFFN forecast {analysis['dffn']:.2f} kWh; "
            f"What-If DFFN forecast {scenario['dffn']:.2f} kWh "
            f"({change:+.2f}%). Estimated DFFN cost change: ₹{cost_change:,.2f}. "
            "This is a model-level scenario estimate, not a guaranteed production effect. "
            "Validate operational feasibility and actual energy measurements before implementation."
        )
    else:
        status, title = "Forecast ready", "Review the forecast before scenario action."
        message = (
            f"DFFN predicts {analysis['dffn']:.2f} kWh for the current operating condition. "
            "Use What-If Analysis to evaluate an alternative operating condition."
        )
    return {"status": status, "title": title, "message": message}
