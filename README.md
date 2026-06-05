# Serverless AQI Predictor

## Hyderabad AQI Forecasting System

A complete end-to-end air quality forecasting system that predicts the **Air Quality Index (AQI) for Hyderabad, Pakistan for the next 72 hours** using Open-Meteo data, Hopsworks Feature Store, GitHub Actions automation, FastAPI backend, and a Streamlit dashboard.

---

## Project Overview

This project forecasts AQI for the next 3 days by combining future weather forecasts, pollutant forecasts, engineered time-based features, and a trained machine learning model.

The system is designed as a serverless-style ML pipeline:

* Open-Meteo provides weather and air-quality data.
* Feature pipelines generate training and forecast features.
* Hopsworks stores features and the trained model.
* GitHub Actions automates hourly and daily workflows.
* FastAPI serves predictions through an API.
* Streamlit displays the AQI dashboard.

The main goal is not just to train a model, but to build a working production-style AQI forecasting pipeline.

---

## Live Demo

### Streamlit Dashboard

```text
https://hyderabadaqi046.streamlit.app/
```

### FastAPI Backend

```text
https://ayyan22bscs046-hyderabad-aqi-api.hf.space
```

### API Health Check

```text
https://ayyan22bscs046-hyderabad-aqi-api.hf.space/health
```

### Prediction Endpoint

```text
https://ayyan22bscs046-hyderabad-aqi-api.hf.space/predictions?hours=72
```

---

## Tech Stack

| Component            | Technology                                 |
| -------------------- | ------------------------------------------ |
| Data Source          | Open-Meteo Weather API and Air Quality API |
| Programming Language | Python                                     |
| Data Processing      | Pandas, NumPy                              |
| Machine Learning     | Scikit-learn, XGBoost                      |
| Feature Store        | Hopsworks Feature Store                    |
| Model Registry       | Hopsworks Model Registry                   |
| Automation           | GitHub Actions                             |
| Backend API          | FastAPI                                    |
| Frontend Dashboard   | Streamlit                                  |
| Backend Deployment   | Hugging Face Docker Space                  |
| Frontend Deployment  | Streamlit Cloud                            |

---

## System Architecture

```text
Open-Meteo APIs
    |
    |-- Weather Forecast Data
    |-- Air Quality Forecast Data
    |
Feature Engineering Pipelines
    |
    |-- Historical Training Features
    |-- Future Forecast Features
    |
Hopsworks Feature Store
    |
    |-- Training Feature Group
    |-- Forecast Feature Group
    |
Training Pipeline
    |
    |-- Trains ML Models
    |-- Evaluates Metrics
    |-- Registers Best Model
    |
Hopsworks Model Registry
    |
FastAPI Backend
    |
    |-- Loads Latest Model
    |-- Reads Latest Forecast Features
    |-- Generates 72-hour AQI Predictions
    |
Streamlit Dashboard
    |
    |-- Current AQI
    |-- 3-Day Forecast
    |-- Hourly Trend
    |-- Pollutant Breakdown
    |-- AQI Alerts
```

---

## Project Structure

```text
aqi-predictor/
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml
│       └── training_pipeline.yml
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── app/
│   └── app.py
│
├── data/
│   └── openmeteo_19f_training_features.csv
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   └── eda.ipynb
│
├── pipelines/
│   ├── feature_pipeline.py
│   ├── backfill_pipeline.py
│   ├── training_pipeline.py
│   └── test_future_aqi_predictions.py
│
├── reports/
│   ├── model_metrics.csv
│   ├── selected_features.csv
│   ├── feature_importance.csv
│   └── figures/
│
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── .gitignore
```

---

## Dataset Details

| Item                 | Details                                    |
| -------------------- | ------------------------------------------ |
| City                 | Hyderabad, Pakistan                        |
| Data Source          | Open-Meteo Weather API and Air Quality API |
| Data Type            | Hourly weather, pollutant, and AQI data    |
| Historical Period    | Last 180 days                              |
| Approximate Duration | Around 6 months                            |
| Time Granularity     | Hourly                                     |
| Approximate Rows     | Around 4,320 hourly records                |
| Target Variable      | `us_aqi`                                   |
| Forecast Horizon     | Next 72 hours / 3 days                     |
| Final Feature Count  | 19 features                                |

---

## Final Feature Set

The final production model uses 19 forecast-compatible features.

### Pollutant Features

| Feature      | Description                                                |
| ------------ | ---------------------------------------------------------- |
| `pm25_24h`   | PM2.5 24-hour rolling average                              |
| `pm10_24h`   | PM10 24-hour rolling average                               |
| `o3_8h_ppb`  | Ozone converted to ppb and averaged over 8 hours           |
| `co_8h_ppm`  | Carbon monoxide converted to ppm and averaged over 8 hours |
| `no2_1h_ppb` | Nitrogen dioxide converted to ppb                          |

### Weather Features

| Feature                      | Description                 |
| ---------------------------- | --------------------------- |
| `temperature_2m`             | Temperature forecast        |
| `relative_humidity_2m`       | Relative humidity forecast  |
| `precipitation`              | Precipitation forecast      |
| `windspeed_10m`              | Wind speed forecast         |
| `surface_pressure`           | Surface pressure forecast   |
| `shortwave_radiation`        | Solar radiation forecast    |
| `et0_fao_evapotranspiration` | Evapotranspiration forecast |

### Time-Based Features

| Feature           | Description                        |
| ----------------- | ---------------------------------- |
| `hour`            | Hour of the day                    |
| `day_of_week`     | Day of week                        |
| `is_weekend`      | Weekend indicator                  |
| `hour_sin`        | Cyclic hour sine encoding          |
| `hour_cos`        | Cyclic hour cosine encoding        |
| `day_of_week_sin` | Cyclic day-of-week sine encoding   |
| `day_of_week_cos` | Cyclic day-of-week cosine encoding |

---

## Why AQI Features Were Not Used as Inputs

Open-Meteo also provides future AQI forecasts. However, future AQI values were not used as model input features.

Using AQI as an input while predicting AQI would create a proxy form of target leakage. The model would become dependent on another AQI forecast instead of learning AQI from pollutant and weather conditions.

Therefore:

* Open-Meteo `us_aqi` was used as the training target/reference.
* Future Open-Meteo AQI was used only for comparison and validation.
* Production features were limited to pollutant, weather, and deterministic time-based variables.

This makes the model more independent, interpretable, and closer to a real forecasting system.

---

## Pollutant Unit Conversion

Pollutant features were engineered to match AQI-style measurement conventions.

| Pollutant | Raw Input Unit | Engineering Applied                                              | Final Feature |
| --------- | -------------- | ---------------------------------------------------------------- | ------------- |
| PM2.5     | µg/m³          | 24-hour rolling average                                          | `pm25_24h`    |
| PM10      | µg/m³          | 24-hour rolling average                                          | `pm10_24h`    |
| O₃        | µg/m³ → ppb    | `ppb = (µg/m³ × 24.45) / 48.00`; 8-hour rolling average          | `o3_8h_ppb`   |
| CO        | µg/m³ → ppm    | `ppm = (µg/m³ × 24.45) / (28.01 × 1000)`; 8-hour rolling average | `co_8h_ppm`   |
| NO₂       | µg/m³ → ppb    | `ppb = (µg/m³ × 24.45) / 46.0055`; 1-hour feature                | `no2_1h_ppb`  |

These conversions were not used to manually calculate AQI. The model target remained Open-Meteo’s `us_aqi`. The conversions made the input features more scientifically meaningful and AQI-aligned.

---

## Model Training

The training pipeline compares three models:

1. Ridge Regression
2. Random Forest
3. XGBoost

A time-based train-test split was used instead of a random split. The dataset was sorted by timestamp, with older records used for training and newer records used for testing. This better simulates real forecasting because the model learns from the past and predicts future values.

### Final Model Results

| Model            | Train MAE | Train RMSE | Train R² | Test MAE | Test RMSE | Test R² |
| ---------------- | --------: | ---------: | -------: | -------: | --------: | ------: |
| XGBoost          |    1.1049 |     1.5423 |   0.9973 |   2.0749 |    3.9196 |  0.9619 |
| Random Forest    |    0.8088 |     1.4747 |   0.9975 |   2.0045 |    4.3100 |  0.9539 |
| Ridge Regression |    5.7771 |     8.7000 |   0.9144 |   5.3275 |    7.6916 |  0.8532 |

XGBoost was selected as the final model because it achieved the best overall balance, with the lowest test RMSE and highest test R².

---

## Automation

GitHub Actions manages the scheduled pipelines.

### Hourly Feature Pipeline

Runs every hour at minute 10.

```yaml
cron: "10 * * * *"
```

Responsibilities:

* Fetch future Open-Meteo weather and air-quality data.
* Generate 72-hour forecast features.
* Store features in the Hopsworks forecast Feature Group.

### Daily Training Pipeline

Runs daily at 12:10 AM Pakistan time.

```yaml
cron: "10 19 * * *"
```

GitHub Actions uses UTC, so `19:10 UTC` equals `12:10 AM PKT`.

Responsibilities:

* Backfill recent historical data.
* Generate training features.
* Train and evaluate models.
* Register the best model in Hopsworks Model Registry.

---

## FastAPI Backend

The FastAPI backend is responsible for inference.

It:

* Loads the best model from Hopsworks Model Registry.
* Reads the latest forecast features from Hopsworks Feature Store.
* Generates AQI predictions for the next 72 hours.
* Returns results as JSON.

### API Endpoints

| Endpoint                | Purpose                                   |
| ----------------------- | ----------------------------------------- |
| `/health`               | Checks if the API is running              |
| `/predictions?hours=72` | Returns 72-hour AQI predictions           |
| `/refresh-cache`        | Clears cached model and feature resources |

Run locally:

```bash
uvicorn api.main:app --reload --port 8000
```

Test locally:

```bash
http://127.0.0.1:8000/health
http://127.0.0.1:8000/predictions?hours=72
```

---

## Streamlit Dashboard

The Streamlit dashboard is the UI layer.

It displays:

* Current forecast AQI
* AQI category
* 3-day average AQI
* Peak AQI
* Daily forecast cards
* Hourly AQI trend
* Pollutant breakdown
* AQI warning alerts
* Full forecast table

The dashboard does not directly load the model. It calls the FastAPI backend using:

```env
FASTAPI_URL=https://ayyan22bscs046-hyderabad-aqi-api.hf.space
```

Run locally:

```bash
streamlit run app/app.py
```

---

## Environment Variables

Create a `.env` file locally:

```env
CITY_NAME=Hyderabad
COUNTRY_CODE=PK
COUNTRY_NAME=Pakistan
TIMEZONE=Asia/Karachi

LATITUDE=25.3960
LONGITUDE=68.3578

FORECAST_PAST_DAYS=2
FORECAST_DAYS=5
PREDICTION_HOURS=72

HOPSWORKS_HOST=your_hopsworks_host
HOPSWORKS_PROJECT=your_hopsworks_project
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_ONLINE_ENABLED=false

FEATURE_GROUP_NAME=aqi_openmeteo_19f_training_fg
FEATURE_GROUP_VERSION=1

FORECAST_FEATURE_GROUP_NAME=aqi_openmeteo_19f_forecast_fg
FORECAST_FEATURE_GROUP_VERSION=1

MODEL_NAME=aqi_openmeteo_19f_best_model
MODEL_OUTPUT_PATH=models/best_model.pkl

FASTAPI_URL=http://127.0.0.1:8000
```

Never commit `.env` to GitHub.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/AyyanAfzal/aqi_prediction_project.git
cd aqi_prediction_project
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate virtual environment:

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

For notebook/EDA work:

```bash
pip install -r requirements-dev.txt
```

---

## Running the Project Locally

### 1. Run Backfill Pipeline

```bash
python pipelines/backfill_pipeline.py
```

### 2. Run Training Pipeline

```bash
python pipelines/training_pipeline.py
```

### 3. Run Feature Pipeline

```bash
python pipelines/feature_pipeline.py
```

### 4. Start FastAPI Backend

```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Start Streamlit Dashboard

Open a second terminal:

```bash
streamlit run app/app.py
```

---

## Deployment

### FastAPI Backend

The FastAPI backend is deployed on Hugging Face Spaces using Docker.

Required files:

```text
Dockerfile
requirements.txt
api/main.py
api/__init__.py
```

The Dockerfile starts the backend using:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 7860
```

### Streamlit Frontend

The dashboard is deployed on Streamlit Cloud.

Streamlit Cloud secrets:

```toml
FASTAPI_URL = "https://ayyan22bscs046-hyderabad-aqi-api.hf.space"
PREDICTION_HOURS = "72"
```

---

## EDA and Explainability

The project includes exploratory data analysis and explainability:

* Feature distribution analysis
* Bivariate pollutant/weather vs AQI analysis
* Feature drift analysis
* SHAP feature importance
* SHAP beeswarm-style plot
* Actual vs predicted AQI plot

SHAP confirmed that `pm25_24h` was the strongest model driver, followed by ozone and PM10-related features. This is scientifically reasonable because particulate matter is a major contributor to AQI.

---

## Key Design Decisions

| Decision                                   | Reason                                                                    |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| Removed AQI lag features                   | They would not be available for future timestamps and could cause leakage |
| Did not use future Open-Meteo AQI as input | It would act as proxy target leakage                                      |
| Used pollutant/weather forecast features   | These are genuinely available for future hours                            |
| Used time-based split                      | Better reflects real forecasting                                          |
| Excluded month features                    | 180-day training window does not cover full yearly seasonality            |
| Used FastAPI separately from Streamlit     | Cleaner architecture and easier deployment                                |
| Kept Hopsworks offline feature groups      | Hourly dashboard does not need millisecond-level online serving           |

---

## Known Limitations

* The model depends on the quality of Open-Meteo forecast data.
* The training dataset covers around 6 months, not multiple years.
* The model is currently built for Hyderabad only.
* Hugging Face free Spaces may sleep after inactivity, causing slower first requests.
* Deep learning models were not included due to dataset size and deployment constraints.

---

## Future Improvements

* Extend the system to multiple cities.
* Increase historical training data to multiple years.
* Add real-time alert notifications.
* Add online Hopsworks feature serving for lower-latency API predictions.
* Add model drift monitoring to the dashboard.
* Compare against ground-truth local sensor data.
* Experiment with temporal deep learning models when enough data is available.

---

## Final Validation

The deployed FastAPI endpoint successfully returned a full 72-hour prediction payload.

Final validation values:

| Metric               |    Value | Category         |
| -------------------- | -------: | ---------------- |
| Current forecast AQI |     75.1 | Moderate         |
| 72-hour average AQI  |     73.4 | Moderate         |
| Peak AQI             |     77.5 | Moderate         |
| Forecast window      | 72 hours | 3 days           |
| Deployed model       |  XGBoost | 19-feature model |

---

## Author

**Ayyan Afzal**

Computer Science Student
Project: Serverless AQI Predictor for Hyderabad, Pakistan

---

## References

* Open-Meteo Weather API
* Open-Meteo Air Quality API
* Hopsworks Feature Store Documentation
* Hopsworks Model Registry Documentation
* GitHub Actions Documentation
* FastAPI Documentation
* Streamlit Documentation
* Hugging Face Spaces Docker Documentation
* Scikit-learn Documentation
* XGBoost Documentation
* SHAP Documentation
