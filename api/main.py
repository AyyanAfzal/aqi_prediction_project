from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import hopsworks
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# Environment
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="Hyderabad AQI Prediction API",
    description="FastAPI backend for AQI prediction using Hopsworks model and forecast features.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Constants
# ============================================================

DEFAULT_FEATURE_COLUMNS = [
    "pm25_24h",
    "pm10_24h",
    "o3_8h_ppb",
    "co_8h_ppm",
    "no2_1h_ppb",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "windspeed_10m",
    "surface_pressure",
    "shortwave_radiation",
    "et0_fao_evapotranspiration",
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]

POLLUTANT_COLUMNS = [
    ("PM2.5", "pm25_24h", "µg/m³", "Fine particles"),
    ("PM10", "pm10_24h", "µg/m³", "Coarse particles"),
    ("Ozone O₃", "o3_8h_ppb", "ppb", "Ground-level ozone"),
    ("CO", "co_8h_ppm", "ppm", "Carbon monoxide"),
    ("NO₂", "no2_1h_ppb", "ppb", "Nitrogen dioxide"),
]


# ============================================================
# Helpers
# ============================================================

def get_setting(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value not in [None, ""]:
        return str(value)
    return default


def get_aqi_category(aqi: float | int | None) -> str:
    if aqi is None or pd.isna(aqi):
        return "Unknown"

    aqi = float(aqi)

    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def get_aqi_advice(category: str) -> str:
    advice = {
        "Good": "Air quality is clean. Normal outdoor activity is fine.",
        "Moderate": "Air quality is acceptable. Sensitive people should stay aware.",
        "Unhealthy for Sensitive Groups": "Sensitive groups should reduce long outdoor exposure.",
        "Unhealthy": "Everyone should limit prolonged outdoor activity.",
        "Very Unhealthy": "Avoid outdoor exertion. Keep windows closed if possible.",
        "Hazardous": "Health warning. Avoid outdoor activity and use protection indoors.",
        "Unknown": "AQI status unavailable.",
    }
    return advice.get(category, "AQI status unavailable.")


def get_pollutant_level(column: str, value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "Unknown"

    value = float(value)

    breakpoints = {
        "pm25_24h": [
            (9.0, "Good"),
            (35.4, "Moderate"),
            (55.4, "Unhealthy for Sensitive Groups"),
            (125.4, "Unhealthy"),
            (225.4, "Very Unhealthy"),
            (float("inf"), "Hazardous"),
        ],
        "pm10_24h": [
            (54, "Good"),
            (154, "Moderate"),
            (254, "Unhealthy for Sensitive Groups"),
            (354, "Unhealthy"),
            (424, "Very Unhealthy"),
            (float("inf"), "Hazardous"),
        ],
        "o3_8h_ppb": [
            (54, "Good"),
            (70, "Moderate"),
            (85, "Unhealthy for Sensitive Groups"),
            (105, "Unhealthy"),
            (200, "Very Unhealthy"),
            (float("inf"), "Hazardous"),
        ],
        "co_8h_ppm": [
            (4.4, "Good"),
            (9.4, "Moderate"),
            (12.4, "Unhealthy for Sensitive Groups"),
            (15.4, "Unhealthy"),
            (30.4, "Very Unhealthy"),
            (float("inf"), "Hazardous"),
        ],
        "no2_1h_ppb": [
            (53, "Good"),
            (100, "Moderate"),
            (360, "Unhealthy for Sensitive Groups"),
            (649, "Unhealthy"),
            (1249, "Very Unhealthy"),
            (float("inf"), "Hazardous"),
        ],
    }

    if column not in breakpoints:
        return "Measured"

    for upper, label in breakpoints[column]:
        if value <= upper:
            return label

    return "Unknown"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def parse_datetime_column(series: pd.Series, timezone_name: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")

    try:
        if parsed.dt.tz is not None:
            return parsed.dt.tz_convert(timezone_name).dt.tz_localize(None)
        return parsed
    except Exception:
        return (
            pd.to_datetime(series.astype(str), errors="coerce", utc=True)
            .dt.tz_convert(timezone_name)
            .dt.tz_localize(None)
        )


def find_file(root: Path, preferred_names: list[str], fallback_patterns: list[str]) -> Path:
    for name in preferred_names:
        matches = list(root.rglob(name))
        if matches:
            return matches[0]

    for pattern in fallback_patterns:
        matches = list(root.rglob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Could not find model file inside downloaded folder: {root}")


def find_optional_file(root: Path, filename: str) -> Path | None:
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


# ============================================================
# Hopsworks + model loading
# ============================================================

@lru_cache(maxsize=1)
def get_hopsworks_project():
    host = get_setting("HOPSWORKS_HOST")
    project_name = get_setting("HOPSWORKS_PROJECT")
    api_key = get_setting("HOPSWORKS_API_KEY")

    missing = []
    if not host:
        missing.append("HOPSWORKS_HOST")
    if not project_name:
        missing.append("HOPSWORKS_PROJECT")
    if not api_key:
        missing.append("HOPSWORKS_API_KEY")

    if missing:
        raise ValueError("Missing Hopsworks env values: " + ", ".join(missing))

    return hopsworks.login(
        host=host,
        project=project_name,
        api_key_value=api_key,
    )


def get_latest_model_metadata(mr, model_name: str, model_version: str | None):
    if model_version not in [None, "", "None", "none"]:
        return mr.get_model(model_name, version=int(model_version))

    try:
        return mr.get_model(model_name)
    except Exception:
        models = mr.get_models(model_name)
        if not models:
            raise ValueError(f"No model versions found for {model_name}")

        return sorted(
            models,
            key=lambda model_obj: int(getattr(model_obj, "version", 0)),
            reverse=True,
        )[0]


@lru_cache(maxsize=1)
def load_model_bundle():
    project = get_hopsworks_project()
    mr = project.get_model_registry()

    model_name = get_setting("MODEL_NAME", "aqi_openmeteo_19f_best_model")
    model_version = get_setting("MODEL_VERSION", None)

    model_meta = get_latest_model_metadata(mr, model_name, model_version)
    model_dir = Path(model_meta.download())

    model_path = find_file(
        root=model_dir,
        preferred_names=["model.pkl", "model.joblib", "best_model.pkl"],
        fallback_patterns=["*.pkl", "*.joblib"],
    )

    loaded = joblib.load(model_path)

    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        feature_columns = loaded.get("feature_columns", DEFAULT_FEATURE_COLUMNS)
        best_model_name = loaded.get("best_model_name", "Best AQI Model")
    else:
        model = loaded
        feature_columns = getattr(model, "feature_names_in_", DEFAULT_FEATURE_COLUMNS)
        best_model_name = "Best AQI Model"

    feature_json_path = find_optional_file(model_dir, "feature_columns.json")
    if feature_json_path is not None:
        with open(feature_json_path, "r", encoding="utf-8") as file:
            feature_json = json.load(file)

        if isinstance(feature_json, list):
            feature_columns = feature_json
        elif isinstance(feature_json, dict):
            feature_columns = feature_json.get("feature_columns", feature_columns)

    metadata = {}

    for metadata_filename in ["model_metadata.json", "metadata.json", "metrics.json"]:
        metadata_path = find_optional_file(model_dir, metadata_filename)
        if metadata_path is not None:
            with open(metadata_path, "r", encoding="utf-8") as file:
                loaded_metadata = json.load(file)
            if isinstance(loaded_metadata, dict):
                metadata.update(loaded_metadata)

    best_model_name = metadata.get("best_model_name", metadata.get("model_name", best_model_name))
    feature_columns = [str(col).strip().lower() for col in list(feature_columns)]

    return {
        "model": model,
        "feature_columns": feature_columns,
        "metadata": metadata,
        "model_name": model_name,
        "model_version": getattr(model_meta, "version", None),
        "best_model_name": best_model_name,
        "model_path": str(model_path),
    }


def load_forecast_features() -> pd.DataFrame:
    project = get_hopsworks_project()
    fs = project.get_feature_store()

    fg_name = get_setting("FORECAST_FEATURE_GROUP_NAME", "aqi_openmeteo_19f_forecast_fg")
    fg_version = int(get_setting("FORECAST_FEATURE_GROUP_VERSION", "1"))

    fg = fs.get_feature_group(name=fg_name, version=fg_version)

    try:
        df = fg.read(dataframe_type="pandas", read_options={"use_hive": True})
    except Exception:
        try:
            df = fg.read(dataframe_type="pandas")
        except Exception:
            df = fg.read()

    if df.empty:
        raise ValueError("Forecast Feature Group is empty.")

    return df


# ============================================================
# Prediction logic
# ============================================================

def prepare_prediction_dataframe(
    raw_df: pd.DataFrame,
    feature_columns: list[str],
    prediction_hours: int,
    timezone_name: str,
) -> pd.DataFrame:
    df = normalize_columns(raw_df)

    if "timestamp" not in df.columns:
        raise ValueError("Forecast Feature Group must contain a timestamp column.")

    df["timestamp"] = parse_datetime_column(df["timestamp"], timezone_name)
    df = df.dropna(subset=["timestamp"]).copy()

    for col in ["forecast_run_timestamp", "ingestion_timestamp"]:
        if col in df.columns:
            df[col] = parse_datetime_column(df[col], timezone_name)

    run_col = None
    if "forecast_run_timestamp" in df.columns:
        run_col = "forecast_run_timestamp"
    elif "ingestion_timestamp" in df.columns:
        run_col = "ingestion_timestamp"

    if run_col is not None and df[run_col].notna().any():
        latest_run = df[run_col].max()
        df = df[df[run_col] == latest_run].copy()

    if "is_future" in df.columns:
        df["is_future"] = pd.to_numeric(df["is_future"], errors="coerce").fillna(0).astype(int)
        future_df = df[df["is_future"] == 1].copy()
    else:
        now_local = pd.Timestamp.now(tz=ZoneInfo(timezone_name)).tz_localize(None).floor("h")
        future_df = df[df["timestamp"] > now_local].copy()

    if future_df.empty:
        raise ValueError(
            "No future forecast rows found. Run the hourly feature pipeline first."
        )

    subset_cols = ["timestamp"]
    if "city" in future_df.columns:
        subset_cols = ["city", "timestamp"]

    sort_cols = ["timestamp"]
    if run_col is not None:
        sort_cols = [run_col, "timestamp"]

    future_df = (
        future_df.sort_values(sort_cols)
        .drop_duplicates(subset=subset_cols, keep="last")
        .sort_values("timestamp")
        .head(prediction_hours)
        .reset_index(drop=True)
    )

    missing_features = [col for col in feature_columns if col not in future_df.columns]
    if missing_features:
        raise ValueError(
            "Forecast Feature Group is missing model features: "
            + ", ".join(missing_features)
        )

    for col in feature_columns:
        future_df[col] = pd.to_numeric(future_df[col], errors="coerce")
        median_value = future_df[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        future_df[col] = future_df[col].fillna(median_value)

    return future_df


def predict_aqi(forecast_df: pd.DataFrame, model, feature_columns: list[str]) -> pd.DataFrame:
    df = forecast_df.copy()

    predictions = model.predict(df[feature_columns])
    predictions = np.clip(predictions, 0, 500)

    df["predicted_aqi"] = predictions
    df["predicted_aqi"] = df["predicted_aqi"].round(1)
    df["aqi_category"] = df["predicted_aqi"].apply(get_aqi_category)

    return df


def make_daily_summary(hourly_df: pd.DataFrame) -> pd.DataFrame:
    df = hourly_df.copy().sort_values("timestamp").reset_index(drop=True)
    df["forecast_day"] = (np.arange(len(df)) // 24) + 1

    agg_dict = {
        "predicted_aqi": "mean",
        "timestamp": ["min", "max"],
    }

    if "openmeteo_us_aqi_reference" in df.columns:
        df["openmeteo_us_aqi_reference"] = pd.to_numeric(
            df["openmeteo_us_aqi_reference"],
            errors="coerce",
        )
        agg_dict["openmeteo_us_aqi_reference"] = "mean"

    for _, col, _, _ in POLLUTANT_COLUMNS:
        if col in df.columns:
            agg_dict[col] = "mean"

    daily = df.groupby("forecast_day").agg(agg_dict).head(3)

    daily.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in daily.columns
    ]

    daily = daily.reset_index()

    daily = daily.rename(
        columns={
            "predicted_aqi_mean": "predicted_aqi",
            "timestamp_min": "start_time",
            "timestamp_max": "end_time",
            "openmeteo_us_aqi_reference_mean": "openmeteo_us_aqi_reference",
            "pm25_24h_mean": "pm25_24h",
            "pm10_24h_mean": "pm10_24h",
            "o3_8h_ppb_mean": "o3_8h_ppb",
            "co_8h_ppm_mean": "co_8h_ppm",
            "no2_1h_ppb_mean": "no2_1h_ppb",
        }
    )

    daily["predicted_aqi"] = daily["predicted_aqi"].round(1)
    daily["aqi_category"] = daily["predicted_aqi"].apply(get_aqi_category)
    daily["advice"] = daily["aqi_category"].apply(get_aqi_advice)

    return daily


def make_pollutant_summary(hourly_df: pd.DataFrame) -> list[dict]:
    rows = []

    for name, col, unit, desc in POLLUTANT_COLUMNS:
        if col not in hourly_df.columns:
            continue

        value = float(pd.to_numeric(hourly_df[col], errors="coerce").mean())
        level = get_pollutant_level(col, value)

        rows.append(
            {
                "name": name,
                "column": col,
                "description": desc,
                "value": round(value, 3),
                "unit": unit,
                "level": level,
            }
        )

    return rows


def dataframe_to_hourly_records(hourly_df: pd.DataFrame) -> list[dict]:
    cols = [
        "timestamp",
        "predicted_aqi",
        "aqi_category",
        "pm25_24h",
        "pm10_24h",
        "o3_8h_ppb",
        "co_8h_ppm",
        "no2_1h_ppb",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "windspeed_10m",
        "surface_pressure",
    ]

    if "openmeteo_us_aqi_reference" in hourly_df.columns:
        cols.insert(3, "openmeteo_us_aqi_reference")

    cols = [col for col in cols if col in hourly_df.columns]
    df = hourly_df[cols].copy()

    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df.to_dict(orient="records")


def dataframe_to_daily_records(daily_df: pd.DataFrame) -> list[dict]:
    df = daily_df.copy()

    for col in ["start_time", "end_time"]:
        if col in df.columns:
            df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df.to_dict(orient="records")


def build_prediction_payload(prediction_hours: int = 72) -> dict:
    timezone_name = get_setting("TIMEZONE", "Asia/Karachi")
    city_name = get_setting("CITY_NAME", "Hyderabad")
    country_name = get_setting("COUNTRY_NAME", "Pakistan")

    model_bundle = load_model_bundle()
    raw_forecast_df = load_forecast_features()

    forecast_df = prepare_prediction_dataframe(
        raw_df=raw_forecast_df,
        feature_columns=model_bundle["feature_columns"],
        prediction_hours=prediction_hours,
        timezone_name=timezone_name,
    )

    hourly_predictions = predict_aqi(
        forecast_df=forecast_df,
        model=model_bundle["model"],
        feature_columns=model_bundle["feature_columns"],
    )

    daily_summary = make_daily_summary(hourly_predictions)
    pollutant_summary = make_pollutant_summary(hourly_predictions)

    current_aqi = float(hourly_predictions.iloc[0]["predicted_aqi"])
    avg_aqi = float(hourly_predictions["predicted_aqi"].mean())
    peak_aqi = float(hourly_predictions["predicted_aqi"].max())

    current_category = get_aqi_category(current_aqi)
    avg_category = get_aqi_category(avg_aqi)
    peak_category = get_aqi_category(peak_aqi)

    return {
        "status": "success",
        "location": {
            "city": city_name,
            "country": country_name,
            "timezone": timezone_name,
        },
        "summary": {
            "current_aqi": round(current_aqi, 1),
            "current_category": current_category,
            "current_advice": get_aqi_advice(current_category),
            "average_aqi_72h": round(avg_aqi, 1),
            "average_category": avg_category,
            "peak_aqi_72h": round(peak_aqi, 1),
            "peak_category": peak_category,
            "forecast_hours": int(len(hourly_predictions)),
            "start_time": hourly_predictions["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": hourly_predictions["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "model": {
            "model_name": model_bundle["model_name"],
            "model_version": model_bundle["model_version"],
            "best_model_name": model_bundle["best_model_name"],
            "feature_count": len(model_bundle["feature_columns"]),
            "feature_columns": model_bundle["feature_columns"],
        },
        "daily": dataframe_to_daily_records(daily_summary),
        "hourly": dataframe_to_hourly_records(hourly_predictions),
        "pollutants": pollutant_summary,
    }


# ============================================================
# API routes
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Hyderabad AQI Prediction API is running.",
        "docs": "/docs",
        "health": "/health",
        "predictions": "/predictions",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Hyderabad AQI Prediction API",
        "model_name": get_setting("MODEL_NAME", "aqi_openmeteo_19f_best_model"),
        "forecast_feature_group": get_setting(
            "FORECAST_FEATURE_GROUP_NAME",
            "aqi_openmeteo_19f_forecast_fg",
        ),
    }


@app.get("/predictions")
def predictions(
    hours: int = Query(default=72, ge=1, le=168, description="Number of future hours to predict")
):
    try:
        return build_prediction_payload(prediction_hours=hours)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/refresh-cache")
def refresh_cache():
    try:
        get_hopsworks_project.cache_clear()
        load_model_bundle.cache_clear()
        return {"status": "success", "message": "API cache cleared."}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))