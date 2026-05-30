from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import hopsworks
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Environment
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# Constants
# ============================================================

BASE_MODEL_FEATURES = [
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

# Keep this locked. No month, month_sin, or month_cos.
TIME_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
]

DEFAULT_FEATURE_COLUMNS = BASE_MODEL_FEATURES + TIME_FEATURES
FORBIDDEN_TIME_FEATURES = {"month", "month_sin", "month_cos"}

DEFAULT_MODEL_NAME = "aqi_openmeteo_19f_best_model"
DEFAULT_FORECAST_FG_NAME = "aqi_openmeteo_19f_forecast_fg"
DEFAULT_FORECAST_FG_VERSION = 1
DEFAULT_REFERENCE_COLUMN = "openmeteo_us_aqi_reference"
DEFAULT_TIMESTAMP_COLUMN = "timestamp"


# ============================================================
# Utility helpers
# ============================================================

def get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value in [None, ""]:
        return default
    return value


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


def parse_timestamp(series: pd.Series, timezone_name: str) -> pd.Series:
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


def add_time_features(df: pd.DataFrame, timestamp_col: str, timezone_name: str) -> pd.DataFrame:
    """
    Adds only 7 short-term time features.
    No month-based features are created here.
    """
    df = df.copy()

    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column not found: {timestamp_col}")

    df[timestamp_col] = parse_timestamp(df[timestamp_col], timezone_name)
    df = df.dropna(subset=[timestamp_col]).copy()

    df["hour"] = df[timestamp_col].dt.hour.astype("int64")
    df["day_of_week"] = df[timestamp_col].dt.dayofweek.astype("int64")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int64")

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float64")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float64")
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7).astype("float64")
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7).astype("float64")

    return df


def find_file(root: Path, preferred_names: list[str], fallback_patterns: list[str]) -> Path:
    for filename in preferred_names:
        matches = list(root.rglob(filename))
        if matches:
            return matches[0]

    for pattern in fallback_patterns:
        matches = list(root.rglob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(f"Could not find model file in downloaded model directory: {root}")


def find_optional_file(root: Path, filename: str) -> Path | None:
    matches = list(root.rglob(filename))
    return matches[0] if matches else None


def validate_feature_columns(feature_columns: list[str]) -> list[str]:
    feature_columns = [str(col).strip().lower() for col in feature_columns]

    forbidden_found = sorted(FORBIDDEN_TIME_FEATURES.intersection(feature_columns))
    if forbidden_found:
        raise ValueError(
            "Forbidden month-based features found in model feature columns: "
            + ", ".join(forbidden_found)
            + "\nRemove month/month_sin/month_cos from training and re-register the model."
        )

    missing_time = [col for col in TIME_FEATURES if col not in feature_columns]
    if missing_time:
        print("WARNING: Model feature columns do not include all 7 time features:")
        for col in missing_time:
            print(f" - {col}")

    return feature_columns


# ============================================================
# Hopsworks/model loading
# ============================================================

def connect_to_hopsworks():
    host = get_env("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
    project_name = get_env("HOPSWORKS_PROJECT")
    api_key = get_env("HOPSWORKS_API_KEY")

    missing = []
    if not host:
        missing.append("HOPSWORKS_HOST")
    if not project_name:
        missing.append("HOPSWORKS_PROJECT")
    if not api_key:
        missing.append("HOPSWORKS_API_KEY")

    if missing:
        raise ValueError("Missing Hopsworks environment values: " + ", ".join(missing))

    print("=" * 100)
    print("Connecting to Hopsworks")
    print("=" * 100)
    print(f"Host   : {host}")
    print(f"Project: {project_name}")

    return hopsworks.login(
        host=host,
        project=project_name,
        api_key_value=api_key,
    )


def get_latest_model_metadata(model_registry, model_name: str, model_version: str | None):
    if model_version not in [None, "", "None", "none"]:
        return model_registry.get_model(model_name, version=int(model_version))

    try:
        return model_registry.get_model(model_name)
    except Exception:
        models = model_registry.get_models(model_name)
        if not models:
            raise ValueError(f"No model versions found in registry for: {model_name}")

        return sorted(
            models,
            key=lambda model_obj: int(getattr(model_obj, "version", 0)),
            reverse=True,
        )[0]


def load_model_bundle(project, model_name: str, model_version: str | None) -> dict:
    print("\n" + "=" * 100)
    print("Loading model from Hopsworks Model Registry")
    print("=" * 100)
    print(f"Model name   : {model_name}")
    print(f"Model version: {model_version or 'latest'}")

    model_registry = project.get_model_registry()
    model_meta = get_latest_model_metadata(model_registry, model_name, model_version)
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
        best_model_name = loaded.get("best_model_name", "best_model")
    else:
        model = loaded
        feature_columns = getattr(model, "feature_names_in_", DEFAULT_FEATURE_COLUMNS)
        best_model_name = "best_model"

    feature_json_path = find_optional_file(model_dir, "feature_columns.json")
    if feature_json_path is not None:
        with open(feature_json_path, "r", encoding="utf-8") as file:
            feature_json = json.load(file)

        if isinstance(feature_json, list):
            feature_columns = feature_json
        elif isinstance(feature_json, dict):
            feature_columns = feature_json.get("feature_columns", feature_columns)

    metadata = {}
    for filename in ["model_metadata.json", "metadata.json", "metrics.json"]:
        metadata_path = find_optional_file(model_dir, filename)
        if metadata_path is not None:
            with open(metadata_path, "r", encoding="utf-8") as file:
                loaded_metadata = json.load(file)
            if isinstance(loaded_metadata, dict):
                metadata.update(loaded_metadata)

    best_model_name = metadata.get("best_model_name", metadata.get("model_name", best_model_name))
    feature_columns = validate_feature_columns(list(feature_columns))

    print(f"Downloaded dir : {model_dir}")
    print(f"Model file     : {model_path}")
    print(f"Registry ver   : {getattr(model_meta, 'version', None)}")
    print(f"Best model     : {best_model_name}")
    print(f"Feature count  : {len(feature_columns)}")
    print("Feature columns:")
    for col in feature_columns:
        print(f" - {col}")

    return {
        "model": model,
        "feature_columns": feature_columns,
        "metadata": metadata,
        "model_version": getattr(model_meta, "version", None),
        "best_model_name": best_model_name,
        "model_path": str(model_path),
    }


def load_forecast_features(project, fg_name: str, fg_version: int) -> pd.DataFrame:
    print("\n" + "=" * 100)
    print("Loading forecast features from Hopsworks")
    print("=" * 100)
    print(f"Forecast FG: {fg_name} v{fg_version}")

    feature_store = project.get_feature_store()
    feature_group = feature_store.get_feature_group(name=fg_name, version=fg_version)

    try:
        df = feature_group.read(dataframe_type="pandas", read_options={"use_hive": True})
    except Exception:
        try:
            df = feature_group.read(dataframe_type="pandas")
        except Exception:
            df = feature_group.read()

    if df.empty:
        raise ValueError(f"Forecast Feature Group is empty: {fg_name} v{fg_version}")

    print(f"Rows loaded   : {len(df)}")
    print(f"Columns loaded: {len(df.columns)}")

    return df


# ============================================================
# Forecast preparation/evaluation
# ============================================================

def prepare_forecast_dataframe(
    raw_df: pd.DataFrame,
    feature_columns: list[str],
    timestamp_col: str,
    timezone_name: str,
    prediction_hours: int,
) -> pd.DataFrame:
    df = normalize_columns(raw_df)

    timestamp_col = timestamp_col.strip().lower()
    if timestamp_col not in df.columns:
        raise ValueError(f"Timestamp column not found in forecast Feature Group: {timestamp_col}")

    df = add_time_features(df, timestamp_col=timestamp_col, timezone_name=timezone_name)

    for run_col in ["forecast_run_timestamp", "ingestion_timestamp"]:
        if run_col in df.columns:
            df[run_col] = parse_timestamp(df[run_col], timezone_name)

    latest_run_col = None
    if "forecast_run_timestamp" in df.columns and df["forecast_run_timestamp"].notna().any():
        latest_run_col = "forecast_run_timestamp"
    elif "ingestion_timestamp" in df.columns and df["ingestion_timestamp"].notna().any():
        latest_run_col = "ingestion_timestamp"

    if latest_run_col is not None:
        latest_run = df[latest_run_col].max()
        df = df[df[latest_run_col] == latest_run].copy()
        print(f"Using latest forecast run from {latest_run_col}: {latest_run}")

    if "is_future" in df.columns:
        is_future = pd.to_numeric(df["is_future"], errors="coerce").fillna(0).astype(int)
        future_df = df[is_future == 1].copy()
    else:
        now_local = pd.Timestamp.now(tz=ZoneInfo(timezone_name)).tz_localize(None).floor("h")
        future_df = df[df[timestamp_col] > now_local].copy()

    if future_df.empty:
        raise ValueError("No future forecast rows found. Run feature_pipeline.py first.")

    subset_cols = [timestamp_col]
    if "city" in future_df.columns:
        subset_cols = ["city", timestamp_col]

    future_df = (
        future_df.sort_values(timestamp_col)
        .drop_duplicates(subset=subset_cols, keep="last")
        .sort_values(timestamp_col)
        .head(prediction_hours)
        .reset_index(drop=True)
    )

    missing_features = [col for col in feature_columns if col not in future_df.columns]
    if missing_features:
        raise ValueError(
            "Forecast Feature Group is missing model features:\n"
            + "\n".join([f" - {col}" for col in missing_features])
            + "\n\nMake sure feature_pipeline.py is writing the 19-feature schema."
        )

    for col in feature_columns:
        future_df[col] = pd.to_numeric(future_df[col], errors="coerce")
        median_value = future_df[col].median()
        if pd.isna(median_value):
            median_value = 0.0
        future_df[col] = future_df[col].fillna(median_value)

    print("\nForecast dataframe prepared")
    print("=" * 100)
    print(f"Rows used       : {len(future_df)}")
    print(f"Prediction hours: {prediction_hours}")
    print(f"Future range    : {future_df[timestamp_col].min()} -> {future_df[timestamp_col].max()}")

    return future_df


def create_predictions(
    df: pd.DataFrame,
    model,
    feature_columns: list[str],
    reference_column: str,
    timestamp_col: str,
) -> pd.DataFrame:
    output_df = df.copy()

    predictions = model.predict(output_df[feature_columns])
    predictions = np.clip(predictions, 0, 500)

    output_df["predicted_aqi"] = predictions.round(2)

    if reference_column in output_df.columns:
        output_df[reference_column] = pd.to_numeric(output_df[reference_column], errors="coerce")
        output_df["absolute_error"] = (output_df[reference_column] - output_df["predicted_aqi"]).abs().round(2)
        output_df["squared_error"] = ((output_df[reference_column] - output_df["predicted_aqi"]) ** 2).round(2)
    else:
        output_df["absolute_error"] = np.nan
        output_df["squared_error"] = np.nan

    output_df = output_df.sort_values(timestamp_col).reset_index(drop=True)
    output_df["forecast_hour"] = np.arange(1, len(output_df) + 1)
    output_df["forecast_day"] = ((output_df["forecast_hour"] - 1) // 24) + 1

    return output_df


def evaluate_hourly_predictions(df: pd.DataFrame, reference_column: str) -> dict:
    if reference_column not in df.columns:
        return {}

    eval_df = df[[reference_column, "predicted_aqi"]].dropna().copy()
    if eval_df.empty:
        return {}

    y_true = eval_df[reference_column]
    y_pred = eval_df["predicted_aqi"]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred) if len(eval_df) >= 2 else np.nan

    return {
        "level": "hourly",
        "rows": int(len(eval_df)),
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4) if not pd.isna(r2) else np.nan,
    }


def build_daily_summary(df: pd.DataFrame, reference_column: str, timestamp_col: str) -> pd.DataFrame:
    agg_dict = {
        "predicted_aqi": "mean",
        timestamp_col: ["min", "max"],
    }

    if reference_column in df.columns:
        agg_dict[reference_column] = "mean"

    pollutant_cols = [
        "pm25_24h",
        "pm10_24h",
        "o3_8h_ppb",
        "co_8h_ppm",
        "no2_1h_ppb",
    ]

    for col in pollutant_cols:
        if col in df.columns:
            agg_dict[col] = "mean"

    daily_df = df.groupby("forecast_day").agg(agg_dict).reset_index()
    daily_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in daily_df.columns
    ]

    daily_df = daily_df.rename(
        columns={
            "predicted_aqi_mean": "predicted_daily_avg_aqi",
            f"{timestamp_col}_min": "start_time",
            f"{timestamp_col}_max": "end_time",
            f"{reference_column}_mean": "openmeteo_daily_avg_aqi",
            "pm25_24h_mean": "pm25_24h_avg",
            "pm10_24h_mean": "pm10_24h_avg",
            "o3_8h_ppb_mean": "o3_8h_ppb_avg",
            "co_8h_ppm_mean": "co_8h_ppm_avg",
            "no2_1h_ppb_mean": "no2_1h_ppb_avg",
        }
    )

    daily_df["predicted_daily_avg_aqi"] = daily_df["predicted_daily_avg_aqi"].round(2)

    if "openmeteo_daily_avg_aqi" in daily_df.columns:
        daily_df["openmeteo_daily_avg_aqi"] = daily_df["openmeteo_daily_avg_aqi"].round(2)
        daily_df["absolute_error"] = (
            daily_df["openmeteo_daily_avg_aqi"] - daily_df["predicted_daily_avg_aqi"]
        ).abs().round(2)
        daily_df["squared_error"] = (
            (daily_df["openmeteo_daily_avg_aqi"] - daily_df["predicted_daily_avg_aqi"]) ** 2
        ).round(2)

    for col in daily_df.columns:
        if col.endswith("_avg"):
            daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce").round(2)

    return daily_df


def evaluate_daily_predictions(daily_df: pd.DataFrame) -> dict:
    required = {"openmeteo_daily_avg_aqi", "predicted_daily_avg_aqi"}
    if not required.issubset(daily_df.columns):
        return {}

    eval_df = daily_df[["openmeteo_daily_avg_aqi", "predicted_daily_avg_aqi"]].dropna().copy()
    if eval_df.empty:
        return {}

    y_true = eval_df["openmeteo_daily_avg_aqi"]
    y_pred = eval_df["predicted_daily_avg_aqi"]

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred) if len(eval_df) >= 2 else np.nan

    return {
        "level": "daily",
        "rows": int(len(eval_df)),
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4) if not pd.isna(r2) else np.nan,
    }


def save_outputs(
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    metrics: list[dict],
    output_dir: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    hourly_file = output_path / "latest_future_aqi_predictions_hourly.csv"
    daily_file = output_path / "latest_future_aqi_predictions_daily.csv"
    metrics_file = output_path / "latest_future_aqi_prediction_metrics.csv"

    hourly_df.to_csv(hourly_file, index=False)
    daily_df.to_csv(daily_file, index=False)
    pd.DataFrame(metrics).to_csv(metrics_file, index=False)

    print("\nSaved outputs")
    print("=" * 100)
    print(hourly_file)
    print(daily_file)
    print(metrics_file)


# ============================================================
# Main runner
# ============================================================

def run_future_prediction_test(
    model_name: str,
    model_version: str | None,
    forecast_fg_name: str,
    forecast_fg_version: int,
    timestamp_col: str,
    timezone_name: str,
    prediction_hours: int,
    reference_column: str,
    output_dir: str,
) -> None:
    if "12f" in forecast_fg_name.lower():
        raise ValueError(
            f"Refusing to use old 12-feature forecast Feature Group: {forecast_fg_name}\n"
            "Use FORECAST_FEATURE_GROUP_NAME=aqi_openmeteo_19f_forecast_fg"
        )

    if "12f" in model_name.lower():
        raise ValueError(
            f"Refusing to use old 12-feature model: {model_name}\n"
            "Use MODEL_NAME=aqi_openmeteo_19f_best_model"
        )

    project = connect_to_hopsworks()
    model_bundle = load_model_bundle(project, model_name=model_name, model_version=model_version)

    raw_forecast_df = load_forecast_features(
        project=project,
        fg_name=forecast_fg_name,
        fg_version=forecast_fg_version,
    )

    feature_columns = model_bundle["feature_columns"]

    prepared_df = prepare_forecast_dataframe(
        raw_df=raw_forecast_df,
        feature_columns=feature_columns,
        timestamp_col=timestamp_col,
        timezone_name=timezone_name,
        prediction_hours=prediction_hours,
    )

    hourly_predictions = create_predictions(
        df=prepared_df,
        model=model_bundle["model"],
        feature_columns=feature_columns,
        reference_column=reference_column,
        timestamp_col=timestamp_col,
    )

    daily_predictions = build_daily_summary(
        df=hourly_predictions,
        reference_column=reference_column,
        timestamp_col=timestamp_col,
    )

    hourly_metrics = evaluate_hourly_predictions(hourly_predictions, reference_column)
    daily_metrics = evaluate_daily_predictions(daily_predictions)

    metrics = []
    if hourly_metrics:
        metrics.append(hourly_metrics)
    if daily_metrics:
        metrics.append(daily_metrics)

    print("\nHourly prediction sample")
    print("=" * 100)
    display_cols = [timestamp_col, "forecast_hour", "forecast_day", "predicted_aqi"]
    if reference_column in hourly_predictions.columns:
        display_cols.extend([reference_column, "absolute_error"])
    print(hourly_predictions[display_cols].head(12).to_string(index=False))

    print("\nDaily summary")
    print("=" * 100)
    print(daily_predictions.to_string(index=False))

    if metrics:
        print("\nMetrics")
        print("=" * 100)
        print(pd.DataFrame(metrics).to_string(index=False))
    else:
        print("\nNo reference AQI column found, so metrics were skipped.")
        print(f"Expected reference column: {reference_column}")

    save_outputs(
        hourly_df=hourly_predictions,
        daily_df=daily_predictions,
        metrics=metrics,
        output_dir=output_dir,
    )


# ============================================================
# CLI
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test next-72-hour AQI predictions using the 19-feature model and forecast Feature Group."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=get_env("MODEL_NAME", DEFAULT_MODEL_NAME),
        help="Hopsworks model registry model name.",
    )

    parser.add_argument(
        "--model-version",
        type=str,
        default=get_env("MODEL_VERSION", None),
        help="Optional model version. Defaults to latest.",
    )

    parser.add_argument(
        "--forecast-fg-name",
        type=str,
        default=get_env("FORECAST_FEATURE_GROUP_NAME", DEFAULT_FORECAST_FG_NAME),
        help="Forecast Feature Group name.",
    )

    parser.add_argument(
        "--forecast-fg-version",
        type=int,
        default=int(get_env("FORECAST_FEATURE_GROUP_VERSION", str(DEFAULT_FORECAST_FG_VERSION))),
        help="Forecast Feature Group version.",
    )

    parser.add_argument(
        "--timestamp-col",
        type=str,
        default=get_env("TIMESTAMP_COLUMN", DEFAULT_TIMESTAMP_COLUMN),
        help="Timestamp column in forecast Feature Group.",
    )

    parser.add_argument(
        "--timezone",
        type=str,
        default=get_env("TIMEZONE", "Asia/Karachi"),
        help="Local timezone for time feature creation.",
    )

    parser.add_argument(
        "--prediction-hours",
        type=int,
        default=int(get_env("PREDICTION_HOURS", "72")),
        help="Number of future hours to predict.",
    )

    parser.add_argument(
        "--reference-column",
        type=str,
        default=get_env("REFERENCE_COLUMN", DEFAULT_REFERENCE_COLUMN),
        help="Reference AQI column in forecast Feature Group for evaluation.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/future_predictions",
        help="Folder to save prediction outputs.",
    )

    args = parser.parse_args()

    run_future_prediction_test(
        model_name=args.model_name,
        model_version=args.model_version,
        forecast_fg_name=args.forecast_fg_name,
        forecast_fg_version=args.forecast_fg_version,
        timestamp_col=args.timestamp_col,
        timezone_name=args.timezone,
        prediction_hours=args.prediction_hours,
        reference_column=args.reference_column,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
