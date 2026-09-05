# SteelSense — Final Integrated Build

**Research Topic:** Steel Manufacturing Energy Consumption Forecasting for Decision Support Using Initial Operational Data

## Project Overview

SteelSense is an integrated forecasting and decision-support web application developed for forecasting future energy consumption in a steel manufacturing environment using initial operational data.

The system focuses on predicting **future 8-hour energy consumption** using machine learning and deep learning techniques. It combines forecasting, explainability, what-if analysis, decision support, and a consolidated final summary to support operational planning and energy-management decisions.

## Project Links

- **Live Application:** https://steelsense2026.pythonanywhere.com/
- **GitHub Repository:** https://github.com/vk1happy/SteelSense

## Approved Workflow

1. **Forecast** — DFFN produces the final 8-hour energy forecast from the initial operational inputs.

2. **Explainability** — shows local DFFN feature influence for the current forecast, followed by an RF Supporting Perspective. RF evaluates the same current-case drivers highlighted by the DFFN explanation and reports normalized percentages based on RF's local sensitivity around the current input. It does not produce or alter the DFFN forecast.

3. **What-If Analysis** — accepts a combined operating condition consisting of lagging reactive power, leading reactive power, CO₂, power factor, and load type. DFFN evaluates the user-defined scenario and compares it directly with the current DFFN baseline.

4. **Decision Support** — interprets the verified DFFN scenario impact and estimated electricity cost change.

5. **Final Summary** — the Dashboard reflects the same stored analysis state and consolidates the forecasting, explainability, scenario, and decision-support results.

## Important Conventions

- Power Factor is entered on the **0–100 percentage scale**. For example, `98.0` means 98%.
- The DAEWOO Steel Industry dataset is sampled every 15 minutes, so `lag4` represents 60 minutes (1 hour).
- The trained model uses the exact finalized 17-feature schema. Time features and one-hot encoded load features are reconstructed by the backend.
- Light Load is the one-hot baseline: Maximum=0, Medium=0.
- Medium Load: Maximum=0, Medium=1.
- Maximum Load: Maximum=1, Medium=0.
- The frontend visual design is preserved from the approved preview. Backend integration changes application behavior without redesigning the UI.
- Random Forest remains a comparative research model and also provides the Explainability-tab RF Supporting Perspective. RF is not used for the displayed forecast, What-If result, or Decision Support prediction.
- RF never replaces, averages, or modifies the DFFN forecast.
- The RF Supporting Perspective uses local RF sensitivity around the current input. The percentages are normalized only across the current DFFN-highlighted drivers and provide an independent model perspective.
- Forecast context is locked per session to provide deterministic repeated calculations.

## Models

### Primary Forecasting Model — DFFN

The primary forecasting model is a Deep Feedforward Neural Network (DFFN).

Architecture:

```text
17 Input Features
       ↓
Dense 64
       ↓
Dense 32
       ↓
Dense 16
       ↓
Output: Future 8-Hour Energy Consumption
