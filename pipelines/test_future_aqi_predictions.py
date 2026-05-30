

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import hopsworks

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Final feature set
# ─────────────────────────────────────────────────────────────

REFERENCE_COLUMN = "openmeteo_us_aqi_reference"

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
]


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

def load_config() -> dict[str, Any]:
    load_dotenv(ENV_PATH)

    return {
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

        "model_output_path": PROJECT_ROOT / os.getenv("MODEL_OUTPUT_PATH", "models/best_model.pkl"),

        "local_forecast_output_path": PROJECT_ROOT / os.getenv(
            "FORECAST_FEATURE_OUTPUT_PATH",
            "reports/latest_72h_forecast_features.csv",
        ),

        "next_72h_hourly_output_path": PROJECT_ROOT / os.getenv(
            "NEXT_72H_HOURLY_OUTPUT_PATH",
            "reports/next_72h_openmeteo_target_predictions.csv",
        ),
        "next_72h_daily_output_path": PROJECT_ROOT / os.getenv(
            "NEXT_72H_DAILY_OUTPUT_PATH",
            "reports/next_72h_openmeteo_target_daily_comparison.csv",
        ),

        "prediction_hours": int(os.getenv("PREDICTION_HOURS", "72")),
    }


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def aqi_category(aqi: float) -> str:
    if pd.isna(aqi):
        return "Unknown"
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


def calculate_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true, y_pred) -> float:
    try:
        return float(r2_score(y_true, y_pred))
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────

def load_best_model(cfg: dict[str, Any]):
    model_path = Path(cfg["model_output_path"])

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run training first: python pipelines/training_pipeline.py --no-register"
        )

    bundle = joblib.load(model_path)

    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        feature_columns = bundle.get("feature_columns") or bundle.get("selected_features") or DEFAULT_FEATURE_COLUMNS
        best_model_name = bundle.get("best_model_name", "Unknown")
    else:
        model = bundle
        feature_columns = DEFAULT_FEATURE_COLUMNS
        best_model_name = "Unknown"

    logger.info("Loaded model: %s", model_path)
    logger.info("Best model name: %s", best_model_name)
    logger.info("Feature count: %s", len(feature_columns))

    return model, feature_columns, best_model_name


# ─────────────────────────────────────────────────────────────
# Hopsworks read
# ─────────────────────────────────────────────────────────────

def connect_to_hopsworks(cfg: dict[str, Any]):
    project = hopsworks.login(
        host=cfg["hopsworks_host"],
        project=cfg["hopsworks_project"],
        api_key_value=cfg["hopsworks_api_key"],
        engine="python",
    )
    return project


def read_forecast_features_from_hopsworks(cfg: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    project = connect_to_hopsworks(cfg)
    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name=cfg["forecast_feature_group_name"],
        version=cfg["forecast_feature_group_version"],
    )

    selected_columns = list(dict.fromkeys([
        "city",
        "timestamp",
        "forecast_run_timestamp",
        "is_future",
        *feature_columns,
        REFERENCE_COLUMN,
    ]))

    logger.info(
        "Reading forecast features from Hopsworks FG: %s v%s",
        cfg["forecast_feature_group_name"],
        cfg["forecast_feature_group_version"],
    )

    query = fg.select(selected_columns)

    try:
        df = query.read(dataframe_type="pandas", read_options={"use_hive": True})
    except Exception as error:
        logger.warning("Hive read failed, trying default read. Reason: %s", error)
        df = query.read(dataframe_type="pandas")

    if df.empty:
        raise ValueError("Forecast dataframe read from Hopsworks is empty.")

    logger.info("Loaded forecast dataframe from Hopsworks: %s", df.shape)
    return df


def read_forecast_features_local(cfg: dict[str, Any]) -> pd.DataFrame:
    path = Path(cfg["local_forecast_output_path"])

    if not path.exists():
        raise FileNotFoundError(
            f"Local forecast feature file not found: {path}. "
            "Run feature pipeline first: python pipelines/feature_pipeline.py --no-upload"
        )

    df = pd.read_csv(path)
    logger.info("Loaded local forecast dataframe: %s | shape=%s", path, df.shape)
    return df


def load_forecast_features(cfg: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    try:
        return read_forecast_features_from_hopsworks(cfg, feature_columns)
    except Exception as error:
        logger.warning("Could not read forecast features from Hopsworks: %s", error)
        logger.warning("Falling back to local forecast CSV.")
        return read_forecast_features_local(cfg)


# ─────────────────────────────────────────────────────────────
# Prediction preparation
# ─────────────────────────────────────────────────────────────

def prepare_forecast_dataframe(
    df: pd.DataFrame,
    feature_columns: list[str],
    prediction_hours: int,
) -> pd.DataFrame:
    df = df.copy()

    required_columns = ["timestamp", *feature_columns, REFERENCE_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Forecast dataframe missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    if "is_future" in df.columns:
        df["is_future"] = pd.to_numeric(df["is_future"], errors="coerce").fillna(1).astype(int)
        df = df[df["is_future"] == 1].copy()

    if "forecast_run_timestamp" in df.columns:
        df["forecast_run_timestamp"] = pd.to_datetime(df["forecast_run_timestamp"], errors="coerce")

        if df["forecast_run_timestamp"].notna().any():
            latest_run = df["forecast_run_timestamp"].max()
            df = df[df["forecast_run_timestamp"] == latest_run].copy()
            logger.info("Using latest forecast run: %s", latest_run)

    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    for col in feature_columns + [REFERENCE_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in feature_columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=feature_columns).reset_index(drop=True)

    df = df.head(prediction_hours).copy()

    if len(df) < prediction_hours:
        raise ValueError(
            f"Only {len(df)} forecast rows available. Need {prediction_hours}. "
            "Run feature_pipeline.py again with FORECAST_DAYS=5."
        )

    logger.info("Prepared forecast rows: %s", len(df))
    logger.info("Prediction range: %s → %s", df["timestamp"].min(), df["timestamp"].max())

    return df


# ─────────────────────────────────────────────────────────────
# Predict + evaluate
# ─────────────────────────────────────────────────────────────

def predict_next_72_hours(model, forecast_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    X = forecast_df[feature_columns]

    pred = model.predict(X)

    result_df = forecast_df.copy()
    result_df["predicted_aqi"] = np.clip(pred, 0, 500)
    result_df["predicted_category"] = result_df["predicted_aqi"].apply(aqi_category)

    result_df["openmeteo_reference_aqi"] = result_df[REFERENCE_COLUMN]
    result_df["openmeteo_category"] = result_df["openmeteo_reference_aqi"].apply(aqi_category)

    result_df["error"] = result_df["predicted_aqi"] - result_df["openmeteo_reference_aqi"]
    result_df["absolute_error"] = result_df["error"].abs()
    result_df["squared_error"] = result_df["error"] ** 2

    result_df["forecast_hour"] = np.arange(1, len(result_df) + 1)
    result_df["forecast_day"] = ((result_df["forecast_hour"] - 1) // 24) + 1

    return result_df


def create_daily_summary(hourly_df: pd.DataFrame) -> pd.DataFrame:
    daily_df = (
        hourly_df.groupby("forecast_day")
        .agg(
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            avg_predicted_aqi=("predicted_aqi", "mean"),
            peak_predicted_aqi=("predicted_aqi", "max"),
            avg_openmeteo_reference=("openmeteo_reference_aqi", "mean"),
            peak_openmeteo_reference=("openmeteo_reference_aqi", "max"),
            mae=("absolute_error", "mean"),
            rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x)))),
            avg_gap=("error", "mean"),
            hourly_rows=("predicted_aqi", "count"),
        )
        .reset_index()
    )

    daily_df["predicted_category"] = daily_df["avg_predicted_aqi"].apply(aqi_category)
    daily_df["openmeteo_category"] = daily_df["avg_openmeteo_reference"].apply(aqi_category)

    return daily_df


def print_results(hourly_df: pd.DataFrame, daily_df: pd.DataFrame, best_model_name: str) -> None:
    valid_reference = hourly_df["openmeteo_reference_aqi"].notna()

    if valid_reference.sum() > 1:
        y_true = hourly_df.loc[valid_reference, "openmeteo_reference_aqi"]
        y_pred = hourly_df.loc[valid_reference, "predicted_aqi"]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = calculate_rmse(y_true, y_pred)
        r2 = safe_r2(y_true, y_pred)
        corr = y_true.corr(y_pred)
        bias = (y_pred - y_true).mean()

        print("\n" + "=" * 80)
        print("Next 3 Days AQI Prediction — Model vs Open-Meteo Forecast AQI")
        print("=" * 80)
        print(f"Model used             : {best_model_name}")
        print(f"Hourly rows compared   : {len(y_true)}")
        print(f"MAE                    : {mae:.3f}")
        print(f"RMSE                   : {rmse:.3f}")
        print(f"R²                     : {r2:.4f}")
        print(f"Pearson correlation    : {corr:.4f}")
        print(f"Mean bias              : {bias:+.3f}")
        print("=" * 80)
    else:
        print("\nOpen-Meteo reference AQI is missing, so metrics could not be calculated.")

    print("\nClean daily summary:")
    for _, row in daily_df.iterrows():
        print(
            f"Day {int(row['forecast_day'])}: "
            f"Pred Avg AQI={row['avg_predicted_aqi']:.2f} ({row['predicted_category']}), "
            f"Pred Peak={row['peak_predicted_aqi']:.2f}, "
            f"Open-Meteo Avg={row['avg_openmeteo_reference']:.2f} ({row['openmeteo_category']}), "
            f"Gap={row['avg_gap']:+.2f}, "
            f"MAE={row['mae']:.2f}, "
            f"RMSE={row['rmse']:.2f}"
        )


def save_outputs(hourly_df: pd.DataFrame, daily_df: pd.DataFrame, cfg: dict[str, Any]) -> None:
    hourly_path = Path(cfg["next_72h_hourly_output_path"])
    daily_path = Path(cfg["next_72h_daily_output_path"])

    hourly_path.parent.mkdir(parents=True, exist_ok=True)
    daily_path.parent.mkdir(parents=True, exist_ok=True)

    hourly_df.to_csv(hourly_path, index=False)
    daily_df.to_csv(daily_path, index=False)

    logger.info("Saved hourly predictions: %s", hourly_path)
    logger.info("Saved daily comparison: %s", daily_path)


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def run_prediction_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()

    model, feature_columns, best_model_name = load_best_model(cfg)

    raw_forecast_df = load_forecast_features(cfg, feature_columns)

    forecast_df = prepare_forecast_dataframe(
        raw_forecast_df,
        feature_columns=feature_columns,
        prediction_hours=cfg["prediction_hours"],
    )

    hourly_predictions = predict_next_72_hours(
        model=model,
        forecast_df=forecast_df,
        feature_columns=feature_columns,
    )

    daily_summary = create_daily_summary(hourly_predictions)

    print_results(hourly_predictions, daily_summary, best_model_name)
    save_outputs(hourly_predictions, daily_summary, cfg)

    logger.info("Prediction pipeline completed successfully.")

    return hourly_predictions, daily_summary


def main() -> None:
    run_prediction_pipeline()


if __name__ == "__main__":
    main()
