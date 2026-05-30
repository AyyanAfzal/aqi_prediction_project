from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
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
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

TARGET_COLUMN = "us_aqi"

# Backward-compatible path used by training_pipeline.py imports.
LOCAL_CACHE_PATH = DATA_DIR / os.getenv(
    "TRAINING_CACHE_PATH",
    "openmeteo_19f_training_cache.pkl",
)

# Gas conversion constants.
# Open-Meteo gas values are usually returned in µg/m³.
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

BASE_MODEL_FEATURE_COLUMNS = [
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

# IMPORTANT: No month/month_sin/month_cos for now.
TIME_FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]

MODEL_FEATURE_COLUMNS = BASE_MODEL_FEATURE_COLUMNS + TIME_FEATURE_COLUMNS

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

    # Derived base model features
    "pm25_24h": (0, 500),
    "pm10_24h": (0, 600),
    "o3_8h_ppb": (0, 300),
    "co_8h_ppm": (0, 50),
    "no2_1h_ppb": (0, 2049),

    # Time features
    "hour": (0, 23),
    "day_of_week": (0, 6),
    "is_weekend": (0, 1),
    "hour_sin": (-1, 1),
    "hour_cos": (-1, 1),
    "day_of_week_sin": (-1, 1),
    "day_of_week_cos": (-1, 1),

    # Target
    TARGET_COLUMN: (0, 500),
}

TRAINING_OUTPUT_COLUMNS = list(dict.fromkeys([
    "city",
    "timestamp",
    "ingestion_timestamp",

    # Raw features for EDA/debugging
    *RAW_FEATURE_COLUMNS,

    # Intermediate derived features for EDA/debugging
    *DERIVED_DIAGNOSTIC_COLUMNS,

    # Final model features: 12 base + 7 time = 19
    *MODEL_FEATURE_COLUMNS,

    # ML target from Open-Meteo historical AQI
    TARGET_COLUMN,
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

        # Keep this controlled through .env/GitHub env.
        "backfill_days": int(os.getenv("BACKFILL_DAYS", "180")),
        "backfill_chunk_days": int(os.getenv("BACKFILL_CHUNK_DAYS", "30")),

        "hopsworks_host": os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
        "hopsworks_project": os.getenv("HOPSWORKS_PROJECT", os.getenv("HOPSWORKS_PROJECT_NAME")),
        "hopsworks_api_key": os.getenv("HOPSWORKS_API_KEY"),

        # NEW training FG for 19-feature schema.
        # Do not reuse the old aqi_openmeteo_12f_training_fg because Hopsworks schema is strict.
        "feature_group_name": os.getenv(
            "FEATURE_GROUP_NAME",
            os.getenv("HOPSWORKS_FEATURE_GROUP", "aqi_openmeteo_19f_training_fg"),
        ),
        "feature_group_version": int(
            os.getenv("FEATURE_GROUP_VERSION", os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "1"))
        ),

        "online_enabled": str_to_bool(os.getenv("HOPSWORKS_ONLINE_ENABLED"), default=False),

        "training_cache_path": PROJECT_ROOT / os.getenv(
            "TRAINING_CACHE_PATH",
            "data/openmeteo_19f_training_cache.pkl",
        ),
        "training_csv_path": PROJECT_ROOT / os.getenv(
            "TRAINING_CSV_PATH",
            "data/openmeteo_19f_training_features.csv",
        ),
    }


def validate_config(cfg: dict[str, Any], upload: bool) -> None:
    if cfg["backfill_days"] < 30:
        logger.warning("BACKFILL_DAYS < 30 may be too small for stable model training.")

    if cfg["backfill_chunk_days"] <= 0:
        raise ValueError("BACKFILL_CHUNK_DAYS must be greater than 0.")

    if upload:
        required = [
            "hopsworks_host",
            "hopsworks_project",
            "hopsworks_api_key",
            "feature_group_name",
            "feature_group_version",
        ]
        missing = [key for key in required if not cfg.get(key)]
        if missing:
            raise ValueError(f"Missing .env values for upload: {missing}")

        fg_name = str(cfg["feature_group_name"]).lower()
        if "12f" in fg_name:
            raise ValueError(
                "You are trying to upload 19-feature rows into a 12f training Feature Group. "
                "Set FEATURE_GROUP_NAME=aqi_openmeteo_19f_training_fg in .env/GitHub Actions."
            )

    logger.info("City: %s", cfg["city_name"])
    logger.info("Timezone: %s", cfg["timezone"])
    logger.info("Backfill days: %s", cfg["backfill_days"])
    logger.info("Backfill chunk days: %s", cfg["backfill_chunk_days"])
    logger.info("Raw feature count: %s", len(RAW_FEATURE_COLUMNS))
    logger.info("Base model feature count: %s", len(BASE_MODEL_FEATURE_COLUMNS))
    logger.info("Time feature count: %s", len(TIME_FEATURE_COLUMNS))
    logger.info("Model feature count: %s", len(MODEL_FEATURE_COLUMNS))
    logger.info("Total output columns: %s", len(TRAINING_OUTPUT_COLUMNS))
    logger.info("Training FG: %s v%s", cfg["feature_group_name"], cfg["feature_group_version"])


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


def date_chunks(start_date: date, end_date: date, chunk_days: int):
    current = start_date
    while current <= end_date:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_date)
        yield current, chunk_end
        current = chunk_end + timedelta(days=1)


def get_backfill_range(days: int) -> tuple[date, date]:
    # Yesterday is safer because historical archive/current day can be incomplete.
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)
    return start_date, end_date


# ─────────────────────────────────────────────────────────────
# Fetch historical data
# ─────────────────────────────────────────────────────────────

def fetch_air_quality_archive(
    location: dict[str, Any],
    cfg: dict[str, Any],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "timezone": cfg["timezone"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": [
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "ozone",
            "us_aqi",
        ],
    }

    data = get_json(AIR_QUALITY_URL, params)
    return air_quality_to_dataframe(data)


def fetch_weather_archive(
    location: dict[str, Any],
    cfg: dict[str, Any],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "timezone": cfg["timezone"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
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

    data = get_json(WEATHER_ARCHIVE_URL, params)
    return weather_to_dataframe(data)


def air_quality_to_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    hourly = data.get("hourly")
    if not hourly:
        raise ValueError("Air-quality response has no hourly data.")

    return pd.DataFrame(hourly).rename(
        columns={
            "time": "timestamp",
            "pm2_5": "pm25",
        }
    )


def weather_to_dataframe(data: dict[str, Any]) -> pd.DataFrame:
    hourly = data.get("hourly")
    if not hourly:
        raise ValueError("Weather response has no hourly data.")

    return pd.DataFrame(hourly).rename(columns={"time": "timestamp"})


def fetch_historical_raw_dataframe(cfg: dict[str, Any]) -> pd.DataFrame:
    location = get_city_coordinates(cfg)
    start_date, end_date = get_backfill_range(cfg["backfill_days"])

    logger.info("Selected location: %s", location)
    logger.info("Historical range: %s to %s", start_date, end_date)

    frames = []
    chunks = list(date_chunks(start_date, end_date, cfg["backfill_chunk_days"]))
    logger.info("Total chunks: %s", len(chunks))

    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        logger.info("Fetching chunk %s/%s: %s to %s", index, len(chunks), chunk_start, chunk_end)

        air_df = fetch_air_quality_archive(location, cfg, chunk_start, chunk_end)
        weather_df = fetch_weather_archive(location, cfg, chunk_start, chunk_end)

        chunk_df = pd.merge(air_df, weather_df, on="timestamp", how="inner")

        if chunk_df.empty:
            logger.warning("Empty merged chunk: %s to %s", chunk_start, chunk_end)
            continue

        frames.append(chunk_df)

    if not frames:
        raise ValueError("No historical data fetched.")

    raw_df = pd.concat(frames, ignore_index=True)
    raw_df = raw_df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce")
    raw_df = raw_df.dropna(subset=["timestamp"]).copy()

    raw_df["city"] = location["city"]
    raw_df["country"] = location["country"]
    raw_df["latitude"] = location["latitude"]
    raw_df["longitude"] = location["longitude"]
    raw_df["ingestion_timestamp"] = datetime.now(timezone.utc).replace(tzinfo=None)

    raw_df = raw_df.sort_values(["city", "timestamp"]).reset_index(drop=True)

    logger.info("Raw historical shape: %s", raw_df.shape)

    return raw_df


# ─────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["city", "timestamp"]).reset_index(drop=True)

    numeric_columns = RAW_FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = [col for col in numeric_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill short API gaps in historical data.
    df[numeric_columns] = df.groupby("city", group_keys=False)[numeric_columns].apply(
        lambda group: group.ffill().bfill()
    )

    for col, (lower, upper) in CLIP_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lower, upper=upper)

    return df.drop_duplicates(["city", "timestamp"], keep="last").reset_index(drop=True)


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive final base model features from raw Open-Meteo pollutant/weather data.
    """

    df = df.copy().sort_values(["city", "timestamp"]).reset_index(drop=True)

    logger.info("Deriving final 12 base model features...")

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


def add_time_features(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Add forecast-safe time features.

    No month/month_sin/month_cos are included because the current dataset is not
    large enough for stable yearly seasonality learning.
    """
    df = df.copy()

    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col]).copy()

    df["hour"] = df[timestamp_col].dt.hour.astype("int64")
    df["day_of_week"] = df[timestamp_col].dt.dayofweek.astype("int64")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int64")

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float64")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float64")

    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7).astype("float64")
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7).astype("float64")

    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_columns = RAW_FEATURE_COLUMNS + DERIVED_DIAGNOSTIC_COLUMNS + MODEL_FEATURE_COLUMNS + [TARGET_COLUMN]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing after feature engineering: {missing}")

    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().sum() > 0:
            median_value = df[col].median()
            if pd.isna(median_value):
                median_value = 0.0
            df[col] = df[col].fillna(median_value)

    for col, (lower, upper) in CLIP_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lower, upper=upper)

    df = df.dropna(subset=required_columns).reset_index(drop=True)

    return df


def build_training_dataframe(cfg: dict[str, Any]) -> pd.DataFrame:
    raw_df = fetch_historical_raw_dataframe(cfg)
    clean_df = clean_raw_data(raw_df)
    feature_df = derive_features(clean_df)
    feature_df = add_time_features(feature_df, timestamp_col="timestamp")
    feature_df = preprocess_features(feature_df)

    final_df = feature_df[TRAINING_OUTPUT_COLUMNS].copy()

    missing_feature_values = int(final_df[MODEL_FEATURE_COLUMNS].isna().sum().sum())
    if missing_feature_values > 0:
        raise ValueError(f"Model feature columns contain {missing_feature_values} missing values.")

    if final_df.empty:
        raise ValueError("Final training dataframe is empty.")

    logger.info("Final training shape: %s", final_df.shape)
    logger.info("Training range: %s → %s", final_df["timestamp"].min(), final_df["timestamp"].max())
    logger.info("Columns: %s", len(final_df.columns))

    return final_df


def build_historical_features(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Backward-compatible helper used by training_pipeline.py fallbacks.
    Builds historical training features locally without uploading to Hopsworks.
    """
    cfg = load_config()
    if config:
        cfg.update(config)

    validate_config(cfg, upload=False)
    return build_training_dataframe(cfg)


# ─────────────────────────────────────────────────────────────
# Hopsworks / output
# ─────────────────────────────────────────────────────────────

def save_local_copy(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    cache_path = Path(cfg["training_cache_path"])
    csv_path = Path(cfg["training_csv_path"])

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_pickle(cache_path)
    df.to_csv(csv_path, index=False)

    logger.info("Saved training cache: %s", cache_path)
    logger.info("Saved training CSV: %s", csv_path)


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

    # Strict dtypes help avoid Hopsworks schema mismatch errors.
    integer_columns = ["hour", "day_of_week", "is_weekend"]
    float_columns = [
        col for col in RAW_FEATURE_COLUMNS
        + DERIVED_DIAGNOSTIC_COLUMNS
        + BASE_MODEL_FEATURE_COLUMNS
        + ["hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", TARGET_COLUMN]
        if col in df.columns
    ]

    for col in float_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    for col in integer_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    return df.dropna(subset=["timestamp", "ingestion_timestamp"])


def write_to_hopsworks(df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    fs = connect_to_hopsworks(cfg)

    fg = fs.get_or_create_feature_group(
        name=cfg["feature_group_name"],
        version=cfg["feature_group_version"],
        description=(
            "Historical training feature group for 19-feature AQI model. Includes raw Open-Meteo "
            "air-quality/weather features, 12 derived model features, 7 time features, and us_aqi target."
        ),
        primary_key=["city", "timestamp"],
        event_time="timestamp",
        online_enabled=cfg["online_enabled"],
    )

    df_to_insert = prepare_for_hopsworks(df)

    logger.info("Rows to insert: %s", len(df_to_insert))
    logger.info("Columns to insert: %s", len(df_to_insert.columns))
    logger.info("Insert dataframe dtypes:\n%s", df_to_insert.dtypes)

    # wait_for_job=False avoids local/GitHub Actions connection drops during materialization wait.
    fg.insert(df_to_insert, write_options={"wait_for_job": False})

    logger.info("Training features submitted to Hopsworks successfully.")
    logger.info("Check Hopsworks UI for materialization job status.")


def run_backfill(upload: bool = True) -> pd.DataFrame:
    cfg = load_config()
    validate_config(cfg, upload=upload)

    df = build_training_dataframe(cfg)
    save_local_copy(df, cfg)

    if upload:
        write_to_hopsworks(df, cfg)
    else:
        logger.info("Upload skipped because --no-upload was used.")

    logger.info("Backfill pipeline completed successfully.")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    run_backfill(upload=not args.no_upload)


if __name__ == "__main__":
    main()
