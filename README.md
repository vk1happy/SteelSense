# SteelSense — Final Integrated Build

**Research topic:** Steel Manufacturing Energy Consumption Forecasting for Decision Support Using Initial Operational Data

## Approved workflow
1. **Forecast** — DFFN produces the final 8-hour energy forecast from the initial operational inputs.
2. **Explainability** — shows local DFFN feature influence for the current forecast, followed by an RF Supporting Perspective. RF uses the variables highlighted by the current DFFN explanation and reports normalized RF learned-importance percentages for those same drivers; it does not produce or alter the DFFN forecast.
3. **What-If Analysis** — accepts a combined operating condition (lagging reactive power, leading reactive power, CO₂, power factor, and load type). DFFN evaluates the user-defined scenario and compares it directly with the current DFFN baseline.
4. **Decision Support** — interprets the verified DFFN scenario impact and estimated cost change.
5. **Final Summary** — the Dashboard reflects the same stored analysis state.

## Important conventions
- Power Factor is entered on the **0–100 percentage scale** (for example, `98.0` means 98%).
- DAEWOO data is sampled every 15 minutes, so `lag4` means 60 minutes.
- The trained model uses the exact finalized 17-feature schema. Time features and one-hot load features are reconstructed by the backend.
- Light Load is the one-hot baseline: Maximum=0, Medium=0; Medium Load: Maximum=0, Medium=1; Maximum Load: Maximum=1, Medium=0.
- The frontend visual design is preserved from the approved preview. Backend integration changes behavior only; it does not redesign the UI.
- Random Forest remains a comparative research model and also provides the Explainability-tab RF Supporting Perspective described above. RF is not used for the displayed forecast, What-If result, or Decision Support prediction, and it never replaces, averages, or modifies the DFFN forecast.
- Forecast context is locked per session for deterministic repeated calculations.

## Model artifacts
- `backend/models/final_dffn.keras`
- `backend/models/final_dffn_scaler.pkl`
- `backend/models/dffn_feature_columns.pkl`
- `backend/models/final_rf.pkl`
- `backend/models/rf_feature_columns.pkl`

## Windows run
From the extracted project root:

```powershell
py -3.11 -m pip install -r backend\requirements.txt
py -3.11 backend\app.py
```

Then open `http://127.0.0.1:5000`.

Do not double-click the HTML file for the integrated application.

## Validation
Run:

```powershell
py -3.11 tests\test_core_backend.py
py -3.11 tests\test_backend_workflow.py
py -3.11 tests\test_frontend_static.py
py -3.11 tests\test_final_integration_static.py
```

The core model uses the supplied saved artifacts; no retraining or model architecture changes are performed by this integration.
