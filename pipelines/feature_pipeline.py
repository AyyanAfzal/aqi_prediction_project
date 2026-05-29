"""
Feature Pipeline for Serverless AQI Predictor

City: Hyderabad, Sindh, Pakistan
Data source: Open-Meteo
Feature store: Hopsworks

Pipeline:
1. Load config from .env
2. Get Hyderabad, Sindh coordinates
3. Fetch hourly air-quality data from Open-Meteo
4. Fetch hourly weather data from Open-Meteo
5. Merge AQI + weather data on event_time
6. Clean and preprocess dataframe
7. Add time, interaction, lag, and rolling features
8. Insert/upsert features into Hopsworks Feature Group

No ML training happens in this file.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import hopsworks


# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


# ============================================================
# Columns
# ============================================================

AIR_QUALITY_COLUMNS = [
    "us_aqi",
    "pm10",
    "pm2_5",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
    "dust",
    "uv_index",
]

WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]

METADATA_COLUMNS = [
    "city",
    "country",
    "latitude",
    "longitude",
    "event_time",
    "ingestion_time",
    "source",
]

TARGET_COLUMN = "us_aqi"


# ============================================================
# Config
# ============================================================

def str_to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default

    return value.strip().lower() in {"true", "1", "yes", "y"}


def load_config() -> dict[str, Any]:
    """
    Load environment variables from .env.
    """

    load_dotenv(ENV_PATH)

    config = {
        "city_name": os.getenv("CITY_NAME", "Hyderabad"),
        "country_code": os.getenv("COUNTRY_CODE", "PK"),
        "timezone": os.getenv("TIMEZONE", "Asia/Karachi"),

        "past_days": int(os.getenv("OPENMETEO_PAST_DAYS", "2")),
        "forecast_days": int(os.getenv("OPENMETEO_FORECAST_DAYS", "3")),

        "hopsworks_host": os.getenv("HOPSWORKS_HOST"),
        "hopsworks_project": os.getenv("HOPSWORKS_PROJECT"),
        "hopsworks_api_key": os.getenv("HOPSWORKS_API_KEY"),

        "feature_group_name": os.getenv("HOPSWORKS_FEATURE_GROUP", "aqi_features"),
        "feature_group_version": int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "1")),
        "online_enabled": str_to_bool(os.getenv("HOPSWORKS_ONLINE_ENABLED"), default=False),
    }

    return config


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate all required configuration values.
    """

    required_keys = [
        "city_name",
        "country_code",
        "timezone",
        "hopsworks_host",
        "hopsworks_project",
        "hopsworks_api_key",
        "feature_group_name",
        "feature_group_version",
    ]

    missing = [key for key in required_keys if not config.get(key)]

    if missing:
        raise ValueError(f"Missing required config values: {missing}")

    logger.info("Config loaded successfully.")
    logger.info("City: %s", config["city_name"])
    logger.info("Country code: %s", config["country_code"])
    logger.info("Timezone: %s", config["timezone"])
    logger.info("Past days: %s", config["past_days"])
    logger.info("Forecast days: %s", config["forecast_days"])
    logger.info("Hopsworks host: %s", config["hopsworks_host"])
    logger.info("Hopsworks project: %s", config["hopsworks_project"])
    logger.info(
        "Feature Group: %s v%s",
        config["feature_group_name"],
        config["feature_group_version"],
    )


# ============================================================
# API helpers
# ============================================================

def get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """
    Make GET request and return JSON.
    """

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_city_coordinates(config: dict[str, Any]) -> dict[str, Any]:
    """
    Get coordinates for Hyderabad, Sindh, Pakistan using Open-Meteo Geocoding API.
    """

    logger.info("Fetching coordinates for Hyderabad, Sindh...")

    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": config["city_name"],
        "count": 10,
        "language": "en",
        "format": "json",
        "country_code": config["country_code"],
    }

    data = get_json(url, params)
    results = data.get("results", [])

    if not results:
        raise ValueError("No city results found from Open-Meteo Geocoding API.")

    locations_df = pd.DataFrame(results)

    required_columns = {"name", "country", "admin1", "latitude", "longitude"}
    missing = required_columns - set(locations_df.columns)

    if missing:
        raise ValueError(f"Geocoding response missing columns: {missing}")

    mask = (
        locations_df["name"].str.lower().eq("hyderabad")
        & locations_df["country"].str.lower().eq("pakistan")
        & locations_df["admin1"].str.lower().eq("sindh")
    )

    selected = locations_df[mask]

    if selected.empty:
        raise ValueError("Could not find Hyderabad, Sindh, Pakistan.")

    row = selected.iloc[0]

    location = {
        "city": "hyderabad_sindh",
        "country": "pakistan",
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
    }

    logger.info("Selected location: %s", location)

    return location


def fetch_air_quality_data(
    latitude: float,
    longitude: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Fetch hourly air-quality data from Open-Meteo Air Quality API.
    """

    logger.info("Fetching air-quality data...")

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(AIR_QUALITY_COLUMNS),
        "timezone": config["timezone"],
        "past_days": config["past_days"],
        "forecast_days": config["forecast_days"],
    }

    return get_json(url, params)


def fetch_weather_data(
    latitude: float,
    longitude: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Fetch hourly weather data from Open-Meteo Forecast API.
    """

    logger.info("Fetching weather data...")

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(WEATHER_COLUMNS),
        "timezone": config["timezone"],
        "past_days": config["past_days"],
        "forecast_days": config["forecast_days"],
    }

    return get_json(url, params)


# ============================================================
# DataFrame conversion
# ============================================================

def air_quality_to_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    """
    Convert Open-Meteo air-quality response to pandas DataFrame.
    """

    hourly = data.get("hourly")

    if not hourly:
        raise ValueError("Air-quality response does not contain hourly data.")

    df = pd.DataFrame(hourly)

    if "time" not in df.columns:
        raise ValueError("Air-quality dataframe is missing the time column.")

    df["event_time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.drop(columns=["time"])

    logger.info("Air-quality rows: %s", len(df))

    return df


def weather_to_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    """
    Convert Open-Meteo weather response to pandas DataFrame.
    """

    hourly = data.get("hourly")

    if not hourly:
        raise ValueError("Weather response does not contain hourly data.")

    df = pd.DataFrame(hourly)

    if "time" not in df.columns:
        raise ValueError("Weather dataframe is missing the time column.")

    df["event_time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.drop(columns=["time"])

    logger.info("Weather rows: %s", len(df))

    return df


def merge_dataframes(
    air_quality_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    location: dict[str, Any],
) -> pd.DataFrame:
    """
    Merge air-quality and weather data on event_time.
    """

    logger.info("Merging air-quality and weather data...")

    df = pd.merge(
        air_quality_df,
        weather_df,
        on="event_time",
        how="inner",
    )

    df["city"] = location["city"]
    df["country"] = location["country"]
    df["latitude"] = location["latitude"]
    df["longitude"] = location["longitude"]
    df["ingestion_time"] = datetime.now(timezone.utc)
    df["source"] = "open_meteo"

    logger.info("Merged rows: %s", len(df))

    return df


# ============================================================
# Cleaning and preprocessing
# ============================================================

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase snake_case.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean dataframe safely before feature engineering.
    """

    logger.info("Cleaning data...")

    df = df.copy()
    df = standardize_column_names(df)

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["ingestion_time"] = pd.to_datetime(
        df["ingestion_time"],
        errors="coerce",
        utc=True,
    )

    before = len(df)
    df = df.dropna(subset=["event_time"])
    logger.info("Dropped rows with missing event_time: %s", before - len(df))

    numeric_columns = [
        "latitude",
        "longitude",
        *AIR_QUALITY_COLUMNS,
        *WEATHER_COLUMNS,
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.drop_duplicates(subset=["city", "event_time"], keep="last")
    logger.info("Dropped duplicate rows: %s", before - len(df))

    df = df.sort_values(["city", "event_time"]).reset_index(drop=True)

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        logger.info("Missing values after cleaning:\n%s", missing_counts)

    return df


# ============================================================
# Feature engineering
# ============================================================

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Safely divide two series.
    """

    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and cyclical features.
    """

    df = df.copy()

    df["hour"] = df["event_time"].dt.hour
    df["day"] = df["event_time"].dt.day
    df["month"] = df["event_time"].dt.month
    df["day_of_week"] = df["event_time"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add deterministic weather and pollutant interaction features.
    """

    df = df.copy()

    if {"temperature_2m", "relative_humidity_2m"}.issubset(df.columns):
        df["temp_humidity_interaction"] = (
            df["temperature_2m"] * df["relative_humidity_2m"]
        )
    else:
        df["temp_humidity_interaction"] = np.nan

    if {"wind_speed_10m", "wind_direction_10m"}.issubset(df.columns):
        wind_direction_radians = np.deg2rad(df["wind_direction_10m"])
        df["wind_x"] = df["wind_speed_10m"] * np.cos(wind_direction_radians)
        df["wind_y"] = df["wind_speed_10m"] * np.sin(wind_direction_radians)
    else:
        df["wind_x"] = np.nan
        df["wind_y"] = np.nan

    if {"pm2_5", "pm10"}.issubset(df.columns):
        df["pm_ratio"] = safe_divide(df["pm2_5"], df["pm10"])
    else:
        df["pm_ratio"] = np.nan

    if {"nitrogen_dioxide", "ozone"}.issubset(df.columns):
        df["no2_o3_ratio"] = safe_divide(df["nitrogen_dioxide"], df["ozone"])
    else:
        df["no2_o3_ratio"] = np.nan

    return df


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling features using only past timestamps.

    Current us_aqi is the target.
    Past us_aqi lag/rolling values are safe because they use previous rows only.
    """

    logger.info("Adding lag and rolling features...")

    df = df.copy()
    df = df.sort_values(["city", "event_time"]).reset_index(drop=True)

    lag_config = {
        "pm2_5": [1, 3, 6],
        "pm10": [1, 3, 6],
        "nitrogen_dioxide": [1],
        "ozone": [1],
        "temperature_2m": [1],
        "relative_humidity_2m": [1],
        "wind_speed_10m": [1],
        "surface_pressure": [1],
        "us_aqi": [1, 3, 6],
    }

    for col, lags in lag_config.items():
        if col not in df.columns:
            logger.warning("Skipping missing lag column: %s", col)
            continue

        for lag in lags:
            df[f"{col}_lag_{lag}h"] = df.groupby("city")[col].shift(lag)

    rolling_config = {
        "us_aqi": {"mean": [3, 6, 12], "std": [6]},
        "pm2_5": {"mean": [3, 6], "std": [6]},
        "pm10": {"mean": [3, 6]},
        "temperature_2m": {"mean": [6]},
        "relative_humidity_2m": {"mean": [6]},
        "wind_speed_10m": {"mean": [6]},
    }

    for col, operations in rolling_config.items():
        if col not in df.columns:
            logger.warning("Skipping missing rolling column: %s", col)
            continue

        shifted = df.groupby("city")[col].shift(1)

        for operation, windows in operations.items():
            for window in windows:
                new_col = f"{col}_rolling_{operation}_{window}h"

                if operation == "mean":
                    df[new_col] = (
                        shifted
                        .groupby(df["city"])
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True)
                    )

                elif operation == "std":
                    df[new_col] = (
                        shifted
                        .groupby(df["city"])
                        .rolling(window=window, min_periods=2)
                        .std()
                        .reset_index(level=0, drop=True)
                    )

    lag_rolling_cols = [
        col for col in df.columns
        if "_lag_" in col or "_rolling_" in col
    ]

    if lag_rolling_cols:
        logger.info("Lag/rolling missing values:\n%s", df[lag_rolling_cols].isna().sum())

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all feature engineering steps.
    """

    logger.info("Engineering features...")

    df = add_time_features(df)
    df = add_interaction_features(df)
    df = add_lag_and_rolling_features(df)

    return df


# ============================================================
# Validation and column ordering
# ============================================================

def validate_final_dataframe(df: pd.DataFrame) -> None:
    """
    Validate final dataframe before inserting into Hopsworks.
    """

    required_columns = [
        "city",
        "country",
        "latitude",
        "longitude",
        "event_time",
        "ingestion_time",
        "source",
        "us_aqi",
    ]

    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Final dataframe missing required columns: {missing}")

    if df.empty:
        raise ValueError("Final dataframe is empty.")

    logger.info("Final dataframe shape: %s", df.shape)


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder columns cleanly:
    metadata → target → raw AQI → weather → engineered features.
    """

    engineered_columns = [
        col for col in df.columns
        if col not in METADATA_COLUMNS
        and col not in AIR_QUALITY_COLUMNS
        and col not in WEATHER_COLUMNS
    ]

    ordered_columns = (
        METADATA_COLUMNS
        + [TARGET_COLUMN]
        + [col for col in AIR_QUALITY_COLUMNS if col != TARGET_COLUMN]
        + WEATHER_COLUMNS
        + engineered_columns
    )

    ordered_columns = [col for col in ordered_columns if col in df.columns]

    return df[ordered_columns]


def prepare_for_hopsworks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare dataframe for Hopsworks insertion.

    Hopsworks supports timestamp/date/bigint for event_time.
    We keep event_time and ingestion_time as pandas datetime.
    """

    df = df.copy()

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["ingestion_time"] = pd.to_datetime(df["ingestion_time"], errors="coerce", utc=True)

    # Hopsworks/Arrow can be sensitive to timezone-aware timestamps.
    # Convert to timezone-naive UTC timestamps for stable insertion.
    if getattr(df["event_time"].dt, "tz", None) is not None:
        df["event_time"] = df["event_time"].dt.tz_convert("UTC").dt.tz_localize(None)

    if getattr(df["ingestion_time"].dt, "tz", None) is not None:
        df["ingestion_time"] = df["ingestion_time"].dt.tz_convert("UTC").dt.tz_localize(None)

    # Replace inf/-inf with nulls
    df = df.replace([np.inf, -np.inf], np.nan)

    # Ensure text columns are strings
    for col in ["city", "country", "source"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


# ============================================================
# Hopsworks writer
# ============================================================

def connect_to_hopsworks(config: dict[str, Any]):
    """
    Connect to Hopsworks and return the project feature store.
    """

    logger.info("Connecting to Hopsworks...")

    project = hopsworks.login(
        host=config["hopsworks_host"],
        project=config["hopsworks_project"],
        api_key_value=config["hopsworks_api_key"],
        engine="python",
    )

    fs = project.get_feature_store()

    logger.info("Connected to Hopsworks project: %s", project.name)
    logger.info("Connected to Feature Store: %s", fs.name)

    return fs


def write_to_hopsworks(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """
    Insert/upsert final dataframe into Hopsworks Feature Group.
    """

    logger.info("Writing features to Hopsworks Feature Group...")

    fs = connect_to_hopsworks(config)

    fg = fs.get_or_create_feature_group(
        name=config["feature_group_name"],
        version=config["feature_group_version"],
        description=(
            "Hourly AQI, pollutant, weather, time, lag, and rolling features "
            "for Hyderabad, Sindh, Pakistan from Open-Meteo."
        ),
        primary_key=["city", "event_time"],
        event_time="event_time",
        online_enabled=config["online_enabled"],
    )

    logger.info(
        "Feature Group ready: %s v%s",
        config["feature_group_name"],
        config["feature_group_version"],
    )

    df_to_insert = prepare_for_hopsworks(df)

    logger.info("Rows to insert/upsert: %s", len(df_to_insert))
    logger.info("Columns to insert/upsert: %s", len(df_to_insert.columns))

    # operation="upsert" prevents duplicates for same primary key when using HUDI.
    # wait=True makes the script wait until the ingestion job finishes.
    fg.insert(
    df_to_insert,
    operation="upsert",
    write_options={"wait_for_job": True},
)

    logger.info("Hopsworks insert/upsert completed successfully.")


# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    """
    Run complete feature pipeline.
    """

    logger.info("Starting Hyderabad AQI feature pipeline...")

    config = load_config()
    validate_config(config)

    location = get_city_coordinates(config)

    air_quality_data = fetch_air_quality_data(
        latitude=location["latitude"],
        longitude=location["longitude"],
        config=config,
    )

    weather_data = fetch_weather_data(
        latitude=location["latitude"],
        longitude=location["longitude"],
        config=config,
    )

    air_quality_df = air_quality_to_dataframe(air_quality_data)
    weather_df = weather_to_dataframe(weather_data)

    merged_df = merge_dataframes(
        air_quality_df=air_quality_df,
        weather_df=weather_df,
        location=location,
    )

    clean_df = clean_data(merged_df)

    final_df = engineer_features(clean_df)
    final_df = reorder_columns(final_df)

    validate_final_dataframe(final_df)

    write_to_hopsworks(final_df, config)

    logger.info("Feature pipeline completed successfully.")


if __name__ == "__main__":
    main()