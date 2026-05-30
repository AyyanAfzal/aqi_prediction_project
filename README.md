# Serverless AQI Predictor 🌫️

A production-style AQI forecasting project for **Hyderabad, Pakistan** using **Open-Meteo**, **Hopsworks Feature Store**, **GitHub Actions**, and **Streamlit**.

The system collects historical and forecast air-quality/weather data, engineers features, trains machine learning models, stores the best model in Hopsworks Model Registry, and serves a clean Streamlit dashboard that predicts AQI for the next 3 days.

---

## Project Overview

This project predicts the next **72 hours / 3 days AQI** using forecast features from Open-Meteo and a trained machine learning model stored in Hopsworks.

The current model setup uses **19 features**:

- 12 air-quality/weather features
- 7 time-based features

No month-based features are used because the project is not yet trained on full yearly/seasonal data.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Requests
- Scikit-learn
- XGBoost
- Hopsworks Feature Store
- Hopsworks Model Registry
- GitHub Actions
- Streamlit
- Plotly
- Open-Meteo APIs

---

## Project Architecture

```text
Open-Meteo APIs
     |
     |-- Historical AQI + Weather Data
     |       |
     |       v
     |   Backfill Pipeline
     |       |
     |       v
     |   Hopsworks Training Feature Group
     |       |
     |       v
     |   Training Pipeline
     |       |
     |       v
     |   Hopsworks Model Registry
     |
     |-- Future Forecast AQI + Weather Data
             |
             v
       Feature Pipeline
             |
             v
       Hopsworks Forecast Feature Group
             |
             v
       Streamlit AQI Dashboard
```

---

## Current Hopsworks Resources

### Training Feature Group

```text
aqi_openmeteo_19f_training_fg
```

Used by the training pipeline.

### Forecast Feature Group

```text
aqi_openmeteo_19f_forecast_fg
```

Used by the hourly feature pipeline and Streamlit dashboard.

### Model Registry Name

```text
aqi_openmeteo_19f_best_model
```

Used by the Streamlit dashboard to load the latest registered model.

---

## Feature Set

### Base AQI/Weather Features

```text
pm25_24h
pm10_24h
o3_8h_ppb
co_8h_ppm
no2_1h_ppb
temperature_2m
relative_humidity_2m
precipitation
windspeed_10m
surface_pressure
shortwave_radiation
et0_fao_evapotranspiration
```

### Time-Based Features

```text
hour
day_of_week
is_weekend
hour_sin
hour_cos
day_of_week_sin
day_of_week_cos
```

### Not Used

```text
month
month_sin
month_cos
```

These are intentionally not included because the project is currently trained on limited historical data, not a full yearly dataset.

---

## Repository Structure

```text
aqi-predictor/
│
├── app/
│   └── app.py
│
├── pipelines/
│   ├── backfill_pipeline.py
│   ├── feature_pipeline.py
│   ├── training_pipeline.py
│   ├── test_future_aqi_predictions.py
│   └── experiments.py
│
├── reports/
│   └── generated reports and CSV outputs
│
├── models/
│   └── local model artifacts
│
├── data/
│   └── local cache and training CSV files
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml
│       └── training_pipeline.yml
│
├── requirements.txt
└── README.md
```

---

## Environment Variables

Create a `.env` file locally:

```env
CITY_NAME=Hyderabad
COUNTRY_NAME=Pakistan
COUNTRY_CODE=PK
TIMEZONE=Asia/Karachi

LATITUDE=25.3960
LONGITUDE=68.3578

HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
HOPSWORKS_PROJECT=your_hopsworks_project_name
HOPSWORKS_API_KEY=your_hopsworks_api_key

FEATURE_GROUP_NAME=aqi_openmeteo_19f_training_fg
FEATURE_GROUP_VERSION=1

FORECAST_FEATURE_GROUP_NAME=aqi_openmeteo_19f_forecast_fg
FORECAST_FEATURE_GROUP_VERSION=1

MODEL_NAME=aqi_openmeteo_19f_best_model
TARGET_COLUMN=us_aqi

BACKFILL_DAYS=180
BACKFILL_CHUNK_DAYS=30

FORECAST_PAST_DAYS=2
FORECAST_DAYS=5
PREDICTION_HOURS=72

HOPSWORKS_ONLINE_ENABLED=false
```

Do not commit `.env` to GitHub.

---

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Recommended Hopsworks version:

```text
hopsworks[python]==4.7.*
```

---

## Run Pipelines Locally

### 1. Run Backfill Pipeline

Fetches historical data, creates 19-feature training rows, and uploads to Hopsworks.

```powershell
python pipelines/backfill_pipeline.py
```

### 2. Run Training Pipeline

Trains multiple models, selects the best model, and uploads it to Hopsworks Model Registry.

```powershell
python pipelines/training_pipeline.py
```

### 3. Run Feature Pipeline

Fetches latest forecast data, creates 19-feature future rows, and uploads to Hopsworks.

```powershell
python pipelines/feature_pipeline.py
```

### 4. Run Streamlit Dashboard

```powershell
streamlit run app/app.py
```

---

## Streamlit Dashboard

The Streamlit app loads:

1. Latest future forecast features from:

```text
aqi_openmeteo_19f_forecast_fg
```

2. Latest trained model from:

```text
aqi_openmeteo_19f_best_model
```

Then it predicts the next 72 hours AQI and displays:

- Current AQI
- 3-day average AQI
- Peak AQI
- AQI category
- 3-day AQI cards
- Hourly AQI trend
- Pollutant levels
- Forecast tables

---

## GitHub Actions

This project uses two automation workflows.

### Hourly Feature Pipeline

Runs every hour and updates future forecast features.

```text
.github/workflows/feature_pipeline.yml
```

Main job:

```text
python pipelines/feature_pipeline.py
```

Updates:

```text
aqi_openmeteo_19f_forecast_fg
```

### Nightly Backfill and Model Training

Runs once daily and updates the training data and best model.

```text
.github/workflows/training_pipeline.yml
```

Main jobs:

```text
python pipelines/backfill_pipeline.py
python pipelines/training_pipeline.py
```

Updates:

```text
aqi_openmeteo_19f_training_fg
aqi_openmeteo_19f_best_model
```

The daily workflow does not run the feature pipeline because forecast feature generation is handled by the hourly workflow.

---

## GitHub Secrets

Add these secrets in GitHub:

```text
HOPSWORKS_HOST
HOPSWORKS_PROJECT
HOPSWORKS_API_KEY
```

Go to:

```text
GitHub Repository → Settings → Secrets and variables → Actions
```

---

## Streamlit Cloud Deployment

Deploy the app using Streamlit Community Cloud.

Main file path:

```text
app/app.py
```

Python version:

```text
3.11
```

Add these secrets in Streamlit Cloud:

```toml
HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
HOPSWORKS_PROJECT = "your_hopsworks_project_name"
HOPSWORKS_API_KEY = "your_hopsworks_api_key"

CITY_NAME = "Hyderabad"
COUNTRY_NAME = "Pakistan"
COUNTRY_CODE = "PK"
TIMEZONE = "Asia/Karachi"

FEATURE_GROUP_NAME = "aqi_openmeteo_19f_training_fg"
FEATURE_GROUP_VERSION = "1"

FORECAST_FEATURE_GROUP_NAME = "aqi_openmeteo_19f_forecast_fg"
FORECAST_FEATURE_GROUP_VERSION = "1"

MODEL_NAME = "aqi_openmeteo_19f_best_model"
TARGET_COLUMN = "us_aqi"

PREDICTION_HOURS = "72"
```

---

## Important Notes

- The training target is:

```text
us_aqi
```

- The forecast comparison/reference column is:

```text
openmeteo_us_aqi_reference
```

- The model predicts AQI using forecast features.
- The Streamlit app performs prediction at runtime by loading the model and forecast rows from Hopsworks.
- Generated folders like `reports/`, `data/`, and `models/` should not be committed unless needed.

---

## Recommended Git Ignore

Add these to `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc

reports/
data/
models/

.DS_Store
```

---

## Manual Test Order

Before deploying or pushing major changes, run:

```powershell
python pipelines/backfill_pipeline.py
python pipelines/training_pipeline.py
python pipelines/feature_pipeline.py
streamlit run app/app.py
```

---

## Status

Current version:

```text
19-feature AQI forecasting system
```

Feature setup:

```text
12 base features + 7 time-based features
```

Deployment target:

```text
Streamlit Cloud
```
