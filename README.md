# SteelSense

## Steel Manufacturing Energy Consumption Forecasting for Decision Support using initial operational data.

SteelSense is a machine learning and deep learning based forecasting system designed to predict future energy consumption in a steel manufacturing environment using initial operational data.

The system focuses on forecasting energy requirements over the next 8 hours and provides model comparison, local explainability, What-If analysis, and decision-support insights for operational planning and energy management.

## Research Objective

The objective of this project is to develop an effective energy consumption forecasting approach using initial production and operational information, while providing interpretable model outputs that can support informed operational decision-making.

## Key Features

- 8-hour future energy consumption forecasting
- Deep Feedforward Neural Network (DFFN) forecasting model
- Random Forest supporting model
- Decision Tree Regressor baseline
- Local driver ranking for model explainability
- Random Forest local supporting perspective
- What-If scenario analysis
- Energy cost estimation
- Decision-support recommendations
- Final consolidated summary

## Models

### Primary Model

Deep Feedforward Neural Network (DFFN)

Architecture:

17 → 64 → 32 → 16 → 1

### Supporting Model

Random Forest Regressor

### Baseline

Decision Tree Regressor (DTR)

## Dataset

The project uses the DAEWOO Steel Industry Energy Consumption dataset.

The original data is recorded at 15-minute intervals. Feature engineering and temporal transformations are applied to construct the forecasting inputs.

## Selected Features

The final forecasting models use 17 selected operational and temporal features, including:

- Lagging Reactive Power
- Leading Reactive Power
- CO₂ Emission
- Leading Power Factor
- Number of Seconds from Midnight (NSM)
- Target Month
- Target Hour
- Target Minute
- Weekend status
- Day-of-week indicators
- Load-type indicators

## Model Performance

Evaluation is performed using a chronological 80/20 train-test split.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| DFFN | 248.8292 | 368.1558 | 0.7322 |
| Random Forest | 244.6106 | 393.2960 | 0.6944 |

The DFFN is selected as the primary model because it provides the stronger overall RMSE and R² performance on the held-out test set.

## Explainability

SteelSense provides local driver analysis for individual forecasts.

The explainability module identifies which current input variables have the strongest local influence on the DFFN prediction around the selected operating condition.

The Random Forest supporting perspective independently evaluates the same highlighted drivers using local RF sensitivity.

These explanations represent model-level behavior and should not be interpreted as causal effects.

## What-If Analysis

Users can modify operational inputs and evaluate the resulting change in predicted energy consumption.

The system compares:

- Baseline forecast
- Scenario forecast
- Energy change
- Percentage change
- Estimated electricity cost impact

## Decision Support

The system translates forecast and What-If results into operational decision-support categories such as:

- High pressure
- Review required
- Within control
- Potential reduction

These recommendations are model-level scenario estimates and should be validated against actual operational conditions and energy measurements.

## Hosted Application

SteelSense is deployed as a web application.

**Live Application:**

https://steelsense2026.pythonanywhere.com/

## Project Structure

```text
SteelSense/
├── app.py
├── requirements.txt
│
├── frontend/
│   └── SteelSense_Explainability_Visual_FINAL.html
│
├── models/
│   ├── dffn_feature_columns.pkl
│   ├── final_dffn.keras
│   ├── final_dffn_scaler.pkl
│   └── rf_feature_columns.pkl
│
└── services/
    ├── __init__.py
    ├── cost.py
    ├── decision_support.py
    ├── explainability.py
    ├── prediction.py
    └── scenario.py
