
"""
Feature Pipeline — Raw + 12 Derived Open-Meteo Features
======================================================

Purpose:
- Fetch recent past + future Open-Meteo air-quality and weather forecast data.
- Keep raw pollutant/weather features for EDA and dashboard analysis.
- Derive the final 12 model features used for AQI prediction.
- Keep Open-Meteo us_aqi as reference/target column, but never as an input feature.
- Select next 72 future hours.
- Save locally and optionally upload to Hopsworks.

Final ML logic:
    predicted_aqi(t) = f(12 derived pollutant/weather features at time t)

Important:
- Raw features are stored for EDA/reporting.
- Model should use only MODEL_FEATURE_COLUMNS.
- openmeteo_us_aqi_reference is target/reference, not input.
- Rolling windows are calculated on recent past + future forecast together.
- Only after rolling features are created do we filter the next 72 future rows.

Run:
    python pipelines/feature_pipeline.py --no-upload
    python pipelines/feature_pipeline.py
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import hopsworks


# ─────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

REFERENCE_COLUMN = "openmeteo_us_aqi_reference"
TARGET_COLUMN = "us_aqi"

# Gas conversion constants.
# Open-Meteo gas values are in µg/m³.
O3_TO_PPB = 24.465 / 48
NO2_TO_PPB = 24.465 / 46
CO_TO_PPM = 24.465 / (28 * 1000)


# ─────────────────────────────────────────────────────────────
# Columns
# ─────────────────────────────────────────────────────────────

RAW_AIR_QUALITY_COLUMNS = [
    "pm25",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "ozone",
]

RAW_WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "windspeed_10m",
    "surface_pressure",
    "shortwave_radiation",
    "et0_fao_evapotranspiration",
]

RAW_FEATURE_COLUMNS = RAW_AIR_QUALITY_COLUMNS + RAW_WEATHER_COLUMNS

MODEL_FEATURE_COLUMNS = [
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
]

DERIVED_DIAGNOSTIC_COLUMNS = [
    "o3_8h",
    "co_8h",
    "no2_1h",
]

CLIP_BOUNDS = {
    # Raw pollutants
    "pm25": (0, 500),
    "pm10": (0, 600),
    "carbon_monoxide": (0, 60000),
    "nitrogen_dioxide": (0, 4000),
    "ozone": (0, 1000),

    # Raw weather
    "temperature_2m": (-10, 55),
    "relative_humidity_2m": (0, 100),
    "precipitation": (0, 200),
    "windspeed_10m": (0, 150),
    "surface_pressure": (900, 1100),
    "shortwave_radiation": (0, 1200),
    "et0_fao_evapotranspiration": (0, 20),

    # Derived features
    "pm25_24h": (0, 500),
    "pm10_24h": (0, 600),
    "o3_8h_ppb": (0, 300),
    "co_8h_ppm": (0, 50),
    "no2_1h_ppb": (0, 2049),

    # Reference
    REFERENCE_COLUMN: (0, 500),
}

# Keep output columns unique.
# NOTE:
# The 12 model features already include 7 raw weather columns:
# temperature_2m, relative_humidity_2m, precipitation, windspeed_10m,
# surface_pressure, shortwave_radiation, et0_fao_evapotranspiration.
# If we add RAW_FEATURE_COLUMNS and MODEL_FEATURE_COLUMNS directly,
# those weather columns appear twice. Hopsworks then sees duplicate
# column names and can crash with: DataFrame object has no attribute dtype.
FORECAST_OUTPUT_COLUMNS = list(dict.fromkeys([
    "city",
    "timestamp",
    "ingestion_timestamp",
    "forecast_run_timestamp",
    "is_future",

    # Raw Open-Meteo features for EDA/dashboard/debugging
    *RAW_FEATURE_COLUMNS,

    # Intermediate derived features for EDA/debugging
    *DERIVED_DIAGNOSTIC_COLUMNS,

    # Final model features
    *MODEL_FEATURE_COLUMNS,

    # Reference/target from Open-Meteo; do not use as model input
    REFERENCE_COLUMN,
]))


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

def str_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"true", "1", "yes", "y"}


def load_config() -> dict[str, Any]:
    load_dotenv(ENV_PATH)

    return {
        "city_name": os.getenv("CITY_NAME", os.getenv("CITY", "Hyderabad")),
        "country_code": os.getenv("COUNTRY_CODE", "PK"),
        "timezone": os.getenv("TIMEZONE", "Asia/Karachi"),

        "latitude": os.getenv("LATITUDE"),
        "longitude": os.getenv("LONGITUDE"),

        "past_days": int(os.getenv("FORECAST_PAST_DAYS", os.getenv("OPENMETEO_PAST_DAYS", "2"))),
        "forecast_days": int(os.getenv("FORECAST_DAYS", os.getenv("OPENMETEO_FORECAST_DAYS", "5"))),
        "prediction_hours": int(os.getenv("PREDICTION_HOURS", "72")),

        "hopsworks_host": os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
        "hopsworks_project": os.getenv("HOPSWORKS_PROJECT", os.getenv("HOPSWORKS_PROJECT_NAME")),
        "hopsworks_api_key": os.getenv("HOPSWORKS_API_KEY"),

        "forecast_feature_group_name": os.getenv(
            "FORECAST_FEATURE_GROUP_NAME",
            os.getenv("HOPSWORKS_FORECAST_FEATURE_GROUP", "aqi_openmeteo_12f_forecast_fg"),
        ),
        "forecast_feature_group_version": int(
            os.getenv("FORECAST_FEATURE_GROUP_VERSION", os.getenv("HOPSWORKS_FORECAST_FEATURE_GROUP_VERSION", "1"))
        ),

        "online_enabled": str_to_bool(os.getenv("HOPSWORKS_ONLINE_ENABLED"), default=False),

        "local_forecast_output_path": PROJECT_ROOT / os.getenv(
            "FORECAST_FEATURE_OUTPUT_PATH",
            "reports/latest_72h_forecast_features.csv",
        ),
    }


def validate_config(cfg: dict[str, Any], upload: bool) -> None:
    if cfg["past_days"] < 2:
        raise ValueError("FORECAST_PAST_DAYS must be at least 2 for 24-hour rolling windows.")

    if cfg["forecast_days"] < 5:
        logger.warning("FORECAST_DAYS < 5 may return fewer than 72 future hours late in the day.")

    if upload:
        required = [
            "hopsworks_host",
            "hopsworks_project",
            "hopsworks_api_key",
            "forecast_feature_group_name",
            "forecast_feature_group_version",
        ]
        missing = [key for key in required if not cfg.get(key)]
        if missing:
            raise ValueError(f"Missing .env values for upload: {missing}")

    logger.info("City: %s", cfg["city_name"])
    logger.info("Timezone: %s", cfg["timezone"])
    logger.info("Past days: %s", cfg["past_days"])
    logger.info("Forecast days: %s", cfg["forecast_days"])
    logger.info("Prediction hours: %s", cfg["prediction_hours"])
    logger.info("Raw feature count: %s", len(RAW_FEATURE_COLUMNS))
    logger.info("Model feature count: %s", len(MODEL_FEATURE_COLUMNS))
    logger.info("Total output columns: %s", len(FORECAST_OUTPUT_COLUMNS))
    logger.info("Forecast FG: %s v%s", cfg["forecast_feature_group_name"], cfg["forecast_feature_group_version"])


# ─────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────

def get_json(url: str, params: dict[str, Any], retries: int = 5, timeout: int = 120) -> dict[str, Any]:
    last_error = None

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as error:
            last_error = error
            logger.warning("API failed attempt %s/%s: %s", attempt, retries, error)
            if attempt < retries:
                time.sleep(attempt * 5)

    raise RuntimeError(f"API request failed after {retries} attempts: {last_error}")


def get_city_coordinates(cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg.get("latitude") and cfg.get("longitude"):
        return {
            "city": str(cfg["city_name"]).lower().replace(" ", "_"),
            "country": "pakistan",
            "latitude": float(cfg["latitude"]),
            "longitude": float(cfg["longitude"]),
        }

    params = {
        "name": cfg["city_name"],
        "count": 10,
        "language": "en",
        "format": "json",
        "country_code": cfg["country_code"],
    }

    data = get_json(GEOCODING_URL, params)
    results = data.get("results", [])

    if not results:
        raise ValueError("No location found from Open-Meteo Geocoding API.")

    locations = pd.DataFrame(results)

    mask = (
        locations["name"].str.lower().eq("hyderabad")
        & locations["country"].str.lower().eq("pakistan")
    )

    if "admin1" in locations.columns:
        mask = mask & locations["admin1"].str.lower().eq("sindh")

    selected = locations[mask]
    if selected.empty:
        selected = locations.head(1)

    row = selected.iloc[0]

    return {
        "city": "hyderabad_sindh",
        "country": "pakistan",
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
    }


def fetch_air_quality_data(location: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "timezone": cfg["timezone"],
        "past_days": cfg["past_days"],
        "forecast_days": cfg["forecast_days"],
        "hourly": [
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "ozone",
            "us_aqi",
        ],
    }

    logger.info("Fetching Open-Meteo air-quality forecast data...")
    return get_json(AIR_QUALITY_URL, params)


def fetch_weather_data(location: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "timezone": cfg["timezone"],
        "past_days": cfg["past_days"],
        "forecast_days": cfg["forecast_days"],
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "windspeed_10m",
            "surface_pressure",
            "shortwave_radiation",
            "et0_fao_evapotranspiration",
        ],
    }

    logger.info("Fetching Open-Meteo weather forecast data...")
    return get_json(WEATHER_FORECAST_URL, params)


def air_quality_to_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    hourly = data.get("hourly")
    if not hourly:
        raise ValueError("Air-quality response has no hourly data.")

    return pd.DataFrame(hourly).rename(
        columns={
            "time": "timestamp",
            "pm2_5": "pm25",
            "us_aqi": REFERENCE_COLUMN,
        }
    )


def weather_to_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    hourly = data.get("hourly")
    if not hourly:
        raise ValueError("Weather response has no hourly data.")

    return pd.DataFrame(hourly).rename(columns={"time": "timestamp"})


# ─────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────

def merge_dataframes(
    air_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    location: dict[str, Any],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    df = pd.merge(air_df, weather_df, on="timestamp", how="inner")

    if df.empty:
        raise ValueError("Merged forecast dataframe is empty.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    current_local_hour = pd.Timestamp.now(tz=cfg["timezone"]).floor("h").tz_localize(None)
    run_time = datetime.now(timezone.utc).replace(tzinfo=None)

    df["city"] = location["city"]
    df["country"] = location["country"]
    df["latitude"] = location["latitude"]
    df["longitude"] = location["longitude"]
    df["ingestion_timestamp"] = run_time
    df["forecast_run_timestamp"] = run_time
    df["is_future"] = (df["timestamp"] > current_local_hour).astype(int)

    logger.info("Current local hour: %s", current_local_hour)
    logger.info("Merged forecast shape before features: %s", df.shape)

    return df.sort_values(["city", "timestamp"]).reset_index(drop=True)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["city", "timestamp"]).reset_index(drop=True)

    numeric_columns = RAW_FEATURE_COLUMNS + [REFERENCE_COLUMN]

    missing = [col for col in numeric_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_columns] = df[numeric_columns].ffill().bfill()

    for col, (lower, upper) in CLIP_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lower, upper=upper)

    return df.drop_duplicates(["city", "timestamp"], keep="last").reset_index(drop=True)


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive final 12 model features.

    Critical:
    Rolling windows are computed on recent past + future forecast together.
    Only after this step do we filter future rows.
    """

    df = df.copy().sort_values(["city", "timestamp"]).reset_index(drop=True)

    logger.info("Deriving final 12 model features...")

    df["pm25_24h"] = df.groupby("city")["pm25"].transform(
        lambda s: s.rolling(window=24, min_periods=1).mean()
    )
    df["pm10_24h"] = df.groupby("city")["pm10"].transform(
        lambda s: s.rolling(window=24, min_periods=1).mean()
    )

    df["o3_8h"] = df.groupby("city")["ozone"].transform(
        lambda s: s.rolling(window=8, min_periods=1).mean()
    )
    df["co_8h"] = df.groupby("city")["carbon_monoxide"].transform(
        lambda s: s.rolling(window=8, min_periods=1).mean()
    )

    df["no2_1h"] = df["nitrogen_dioxide"]

    df["o3_8h_ppb"] = df["o3_8h"] * O3_TO_PPB
    df["co_8h_ppm"] = df["co_8h"] * CO_TO_PPM
    df["no2_1h_ppb"] = df["no2_1h"] * NO2_TO_PPB

    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_columns = RAW_FEATURE_COLUMNS + DERIVED_DIAGNOSTIC_COLUMNS + MODEL_FEATURE_COLUMNS + [REFERENCE_COLUMN]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing after feature engineering: {missing}")

    for col in required_columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    for col, (lower, upper) in CLIP_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lower, upper=upper)

    df = df.dropna(subset=required_columns).reset_index(drop=True)

    return df


def build_forecast_feature_dataframe(cfg: dict[str, Any]) -> pd.DataFrame:
    location = get_city_coordinates(cfg)
    logger.info("Selected location: %s", location)

    air_data = fetch_air_quality_data(location, cfg)
    weather_data = fetch_weather_data(location, cfg)

    air_df = air_quality_to_dataframe(air_data)
    weather_df = weather_to_dataframe(weather_data)

    raw_df = merge_dataframes(air_df, weather_df, location, cfg)
    clean_df = clean_raw_data(raw_df)
    feature_df = derive_features(clean_df)
    feature_df = preprocess_features(feature_df)

    # Save all rows as a useful debug/context file.
    all_rows_path = REPORTS_DIR / "latest_forecast_context_raw_plus_derived.csv"
    feature_df.to_csv(all_rows_path, index=False)
    logger.info("Saved full past+future context features: %s", all_rows_path)

    future_df = feature_df[feature_df["is_future"] == 1].copy()
    future_df = future_df.sort_values("timestamp").head(cfg["prediction_hours"]).reset_index(drop=True)

    if len(future_df) < cfg["prediction_hours"]:
        raise ValueError(
            f"Only {len(future_df)} future rows available. Need {cfg['prediction_hours']}. "
            "Increase FORECAST_DAYS."
        )

    final_df = future_df[FORECAST_OUTPUT_COLUMNS].copy()

    missing_feature_values = int(final_df[MODEL_FEATURE_COLUMNS].isna().sum().sum())
    if missing_feature_values > 0:
        raise ValueError(f"Model feature columns contain {missing_feature_values} missing values.")

    logger.info("Final forecast shape: %s", final_df.shape)
    logger.info("Future range: %s → %s", final_df["timestamp"].min(), final_df["timestamp"].max())
    logger.info("Columns: %s", len(final_df.columns))

    return final_df


# ─────────────────────────────────────────────────────────────
# Hopsworks / output
# ─────────────────────────────────────────────────────────────

def connect_to_hopsworks(cfg: dict[str, Any]):
    project = hopsworks.login(
        host=cfg["hopsworks_host"],
        project=cfg["hopsworks_project"],
        api_key_value=cfg["hopsworks_api_key"],
        engine="python",
    )
    return project.get_feature_store()


def prepare_for_hopsworks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["ingestion_timestamp"] = pd.to_datetime(df["ingestion_timestamp"], errors="coerce")
    df["forecast_run_timestamp"] = pd.to_datetime(df["forecast_run_timestamp"], errors="coerce")
    return df.dropna(subset=["timestamp", "ingestion_timestamp", "forecast_run_timestamp"])


def write_to_hopsworks(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    fs = connect_to_hopsworks(cfg)

    fg = fs.get_or_create_feature_group(
        name=cfg["forecast_feature_group_name"],
        version=cfg["forecast_feature_group_version"],
        description=(
            "Forecast feature group for AQI model. Includes raw Open-Meteo "
            "air-quality/weather features, 12 derived model features, and "
            "Open-Meteo us_aqi reference for comparison."
        ),
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=cfg["online_enabled"],
    )

    df_to_insert = prepare_for_hopsworks(df)

    logger.info("Rows to insert: %s", len(df_to_insert))
    logger.info("Columns to insert: %s", len(df_to_insert.columns))

# Fix Hopsworks schema type mismatch
    if "openmeteo_us_aqi_reference" in df_to_insert.columns:
        df_to_insert["openmeteo_us_aqi_reference"] = pd.to_numeric(
            df_to_insert["openmeteo_us_aqi_reference"],
            errors="coerce"
        ).astype("float64")        
    # Avoid waiting for Hopsworks materialization logs, because the wait step can
    # randomly fail with connection drops even when upload has started correctly.
    logging.info("Insert dataframe dtypes:\n%s", df_to_insert.dtypes)
    fg.insert(df_to_insert, write_options={"wait_for_job": False})

    logger.info("Forecast features submitted to Hopsworks successfully.")
    logger.info("Check Hopsworks UI for materialization job status.")


def save_local_copy(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    output_path = Path(cfg["local_forecast_output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved local forecast features: %s", output_path)


def run_feature_pipeline(upload: bool = True) -> pd.DataFrame:
    cfg = load_config()
    validate_config(cfg, upload=upload)

    df = build_forecast_feature_dataframe(cfg)
    save_local_copy(df, cfg)

    if upload:
        write_to_hopsworks(df, cfg)
    else:
        logger.info("Upload skipped because --no-upload was used.")

    logger.info("Feature pipeline completed successfully.")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    run_feature_pipeline(upload=not args.no_upload)


if __name__ == "__main__":
    main()
