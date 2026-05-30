from __future__ import annotations

import json
import os
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import hopsworks
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="Hyderabad AQI",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


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

POLLUTANT_COLUMNS = [
    ("PM2.5", "pm25_24h", "µg/m³", "Fine particles"),
    ("PM10", "pm10_24h", "µg/m³", "Coarse particles"),
    ("Ozone O₃", "o3_8h_ppb", "ppb", "Ground-level ozone"),
    ("CO", "co_8h_ppm", "ppm", "Carbon monoxide"),
    ("NO₂", "no2_1h_ppb", "ppb", "Nitrogen dioxide"),
]

AQI_COLORS = {
    "Good": "#22c55e",
    "Moderate": "#facc15",
    "Unhealthy for Sensitive Groups": "#fb923c",
    "Unhealthy": "#ef4444",
    "Very Unhealthy": "#a855f7",
    "Hazardous": "#7f1d1d",
    "Unknown": "#64748b",
}

AQI_TEXT_COLORS = {
    "Good": "#052e16",
    "Moderate": "#422006",
    "Unhealthy for Sensitive Groups": "#431407",
    "Unhealthy": "#450a0a",
    "Very Unhealthy": "#2e1065",
    "Hazardous": "#ffffff",
    "Unknown": "#ffffff",
}


def render_html(parts: list[str] | str) -> None:
    if isinstance(parts, list):
        html = "".join(str(part) for part in parts)
    else:
        html = str(parts).strip()

    st.markdown(html, unsafe_allow_html=True)


def inject_css() -> None:
    css = """
<style>
    html, body, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top left, rgba(34, 197, 94, 0.10), transparent 34rem),
            radial-gradient(circle at top right, rgba(59, 130, 246, 0.12), transparent 36rem),
            linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%) !important;
    }

    [data-testid="stHeader"] {
        background: rgba(248, 250, 252, 0.72) !important;
        backdrop-filter: blur(16px);
    }

    .block-container {
        padding-top: 4.7rem !important;
        padding-bottom: 3rem !important;
        max-width: 1540px !important;
    }

    .hero-card {
        width: 100%;
        min-height: 330px;
        padding: 42px;
        border-radius: 38px;
        color: white;
        background:
            radial-gradient(circle at 12% 18%, rgba(45, 212, 191, 0.30), transparent 28%),
            radial-gradient(circle at 82% 18%, rgba(96, 165, 250, 0.28), transparent 30%),
            radial-gradient(circle at 72% 92%, rgba(15, 23, 42, 0.68), transparent 45%),
            linear-gradient(135deg, #0f2f2e 0%, #122033 46%, #31476d 100%);
        box-shadow: 0 26px 75px rgba(15, 23, 42, 0.24);
        border: 1px solid rgba(255, 255, 255, 0.17);
        overflow: hidden;
        margin-top: 0.9rem;
        margin-bottom: 1.25rem;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.65fr);
        gap: 34px;
        align-items: center;
        height: 100%;
    }

    .hero-title {
        font-size: clamp(56px, 7vw, 96px);
        font-weight: 1000;
        line-height: 0.92;
        letter-spacing: -3.2px;
        margin-bottom: 34px;
    }

    .hero-meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 14px;
        align-items: center;
    }

    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 13px 18px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.13);
        color: #f8fafc;
        font-size: 15px;
        font-weight: 850;
        border: 1px solid rgba(255, 255, 255, 0.16);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }

    .hero-aqi-box {
        position: relative;
        padding: 30px;
        border-radius: 32px;
        background: rgba(255, 255, 255, 0.14);
        border: 1px solid rgba(255, 255, 255, 0.20);
        box-shadow: 0 18px 50px rgba(2, 6, 23, 0.22);
        backdrop-filter: blur(18px);
        text-align: center;
        overflow: hidden;
    }

    .hero-aqi-box::before {
        content: "";
        position: absolute;
        width: 190px;
        height: 190px;
        border-radius: 999px;
        top: -82px;
        right: -70px;
        background: rgba(255, 255, 255, 0.15);
    }

    .hero-aqi-label {
        position: relative;
        z-index: 2;
        color: #cbd5e1;
        font-size: 14px;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        margin-bottom: 12px;
    }

    .hero-aqi-value {
        position: relative;
        z-index: 2;
        font-size: clamp(74px, 8vw, 112px);
        font-weight: 1000;
        line-height: 0.90;
        letter-spacing: -4px;
        margin-bottom: 18px;
    }

    .hero-category {
        position: relative;
        z-index: 2;
        display: inline-flex;
        padding: 10px 18px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 950;
        margin-bottom: 14px;
    }

    .hero-advice {
        position: relative;
        z-index: 2;
        color: #e2e8f0;
        font-size: 14px;
        line-height: 1.55;
        font-weight: 650;
        max-width: 340px;
        margin: 0 auto;
    }

    .section-title {
        color: #0f172a;
        font-size: 28px;
        font-weight: 950;
        letter-spacing: -0.8px;
        margin: 26px 0 14px 0;
    }

    .section-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-top: -8px;
        margin-bottom: 16px;
        font-weight: 650;
    }

    .day-card {
        padding: 24px;
        border-radius: 30px;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(226, 232, 240, 0.95);
        box-shadow: 0 16px 38px rgba(15, 23, 42, 0.10);
        min-height: 260px;
        position: relative;
        overflow: hidden;
    }

    .day-card::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, rgba(255,255,255,0.38), transparent);
        pointer-events: none;
    }

    .day-top {
        position: relative;
        z-index: 2;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 14px;
    }

    .day-label {
        color: #0f172a;
        font-size: 18px;
        font-weight: 950;
    }

    .day-badge {
        padding: 7px 11px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 950;
    }

    .day-time {
        position: relative;
        z-index: 2;
        color: #64748b;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 18px;
    }

    .day-value {
        position: relative;
        z-index: 2;
        color: #0f172a;
        font-size: 62px;
        font-weight: 1000;
        line-height: 0.95;
        letter-spacing: -2.5px;
        margin-bottom: 12px;
    }

    .day-advice {
        position: relative;
        z-index: 2;
        color: #64748b;
        font-size: 13px;
        font-weight: 650;
        line-height: 1.50;
    }

    .pollutant-card {
        padding: 20px;
        border-radius: 26px;
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid rgba(226, 232, 240, 0.95);
        box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
        min-height: 170px;
    }

    .pollutant-name {
        color: #64748b;
        font-size: 13px;
        font-weight: 850;
        margin-bottom: 6px;
    }

    .pollutant-desc {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 650;
        margin-bottom: 12px;
    }

    .pollutant-value {
        color: #0f172a;
        font-size: 31px;
        font-weight: 1000;
        letter-spacing: -1px;
        line-height: 1;
        margin-bottom: 8px;
    }

    .pollutant-level {
        display: inline-flex;
        padding: 7px 11px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 950;
        margin-top: 8px;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(226, 232, 240, 0.95);
        padding: 16px 18px;
        border-radius: 22px;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.07);
    }

    div[data-testid="stMetricValue"] {
        color: #0f172a;
        font-size: 32px;
        font-weight: 950;
    }

    div[data-testid="stMetricLabel"] {
        color: #475569;
        font-weight: 850;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 13px;
        font-weight: 850;
    }

    @media (max-width: 980px) {
        .block-container {
            padding-top: 3.7rem !important;
        }

        .hero-card {
            padding: 28px;
            border-radius: 30px;
            min-height: auto;
        }

        .hero-grid {
            grid-template-columns: 1fr;
        }

        .hero-title {
            font-size: clamp(44px, 12vw, 68px);
            margin-bottom: 26px;
        }

        .hero-pill {
            white-space: normal;
        }
    }
</style>
"""
    st.markdown(css.strip(), unsafe_allow_html=True)


def get_setting(name: str, default: str | None = None) -> str | None:
    try:
        value = st.secrets.get(name)
        if value not in [None, ""]:
            return str(value)
    except Exception:
        pass

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
        "Good": "Air quality is clean. Great time for normal outdoor activity.",
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


@st.cache_resource(show_spinner=False)
def connect_to_hopsworks(host: str, project_name: str, api_key: str):
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


@st.cache_resource(show_spinner=False)
def load_model_from_registry(_project, model_name: str, model_version: str | None):
    mr = _project.get_model_registry()
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
        "model_version": getattr(model_meta, "version", None),
        "best_model_name": best_model_name,
        "model_path": str(model_path),
    }


@st.cache_data(ttl=900, show_spinner=False)
def load_forecast_features_from_hopsworks(_project, fg_name: str, fg_version: int) -> pd.DataFrame:
    fs = _project.get_feature_store()
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
            "No future forecast rows found. Run the hourly feature pipeline first, then refresh this app."
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
        raise ValueError("Forecast Feature Group is missing model features: " + ", ".join(missing_features))

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
    df["aqi_color"] = df["aqi_category"].map(AQI_COLORS).fillna(AQI_COLORS["Unknown"])

    return df


def make_three_day_summary(hourly_df: pd.DataFrame) -> pd.DataFrame:
    df = hourly_df.copy().sort_values("timestamp").reset_index(drop=True)
    df["forecast_day"] = (np.arange(len(df)) // 24) + 1

    agg_dict = {
        "predicted_aqi": "mean",
        "timestamp": ["min", "max"],
    }

    if "openmeteo_us_aqi_reference" in df.columns:
        df["openmeteo_us_aqi_reference"] = pd.to_numeric(df["openmeteo_us_aqi_reference"], errors="coerce")
        agg_dict["openmeteo_us_aqi_reference"] = "mean"

    for _, col, _, _ in POLLUTANT_COLUMNS:
        if col in df.columns:
            agg_dict[col] = "mean"

    summary = df.groupby("forecast_day").agg(agg_dict).head(3)

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    summary = summary.reset_index()

    summary = summary.rename(
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

    summary["predicted_aqi"] = summary["predicted_aqi"].round(1)
    summary["aqi_category"] = summary["predicted_aqi"].apply(get_aqi_category)

    return summary


def render_hero(city_name: str, country_name: str, hourly_predictions: pd.DataFrame) -> None:
    current_aqi = float(hourly_predictions.iloc[0]["predicted_aqi"])
    avg_aqi = float(hourly_predictions["predicted_aqi"].mean())
    max_aqi = float(hourly_predictions["predicted_aqi"].max())

    category = get_aqi_category(current_aqi)
    color = AQI_COLORS.get(category, AQI_COLORS["Unknown"])
    text_color = AQI_TEXT_COLORS.get(category, "#0f172a")
    advice = get_aqi_advice(category)

    start_time = hourly_predictions["timestamp"].min().strftime("%d %b, %I:%M %p")
    end_time = hourly_predictions["timestamp"].max().strftime("%d %b, %I:%M %p")

    city_safe = escape(city_name)
    country_safe = escape(country_name)
    category_safe = escape(category)
    advice_safe = escape(advice)

    render_html(
        [
            '<div class="hero-card">',
            '<div class="hero-grid">',
            '<div>',
            f'<div class="hero-title">{city_safe} AQI</div>',
            '<div class="hero-meta-row">',
            f'<div class="hero-pill">📍 {city_safe}, {country_safe}</div>',
            f'<div class="hero-pill">🕒 {start_time} → {end_time}</div>',
            f'<div class="hero-pill">📊 3-Day Avg: {avg_aqi:.0f}</div>',
            f'<div class="hero-pill">⚠️ Peak AQI: {max_aqi:.0f}</div>',
            '</div>',
            '</div>',
            '<div class="hero-aqi-box">',
            '<div class="hero-aqi-label">Current AQI</div>',
            f'<div class="hero-aqi-value">{current_aqi:.0f}</div>',
            f'<div class="hero-category" style="background:{color}; color:{text_color};">{category_safe}</div>',
            f'<div class="hero-advice">{advice_safe}</div>',
            '</div>',
            '</div>',
            '</div>',
        ]
    )


def render_section(title: str, subtitle: str | None = None) -> None:
    render_html([f'<div class="section-title">{escape(title)}</div>'])
    if subtitle:
        render_html([f'<div class="section-subtitle">{escape(subtitle)}</div>'])


def render_day_card(day_label: str, start_time, end_time, aqi: float, category: str) -> None:
    color = AQI_COLORS.get(category, AQI_COLORS["Unknown"])
    text_color = AQI_TEXT_COLORS.get(category, "#0f172a")
    advice = get_aqi_advice(category)

    start_label = start_time.strftime("%d %b, %I:%M %p")
    end_label = end_time.strftime("%d %b, %I:%M %p")

    render_html(
        [
            f'<div class="day-card" style="border-top: 8px solid {color};">',
            '<div class="day-top">',
            f'<div class="day-label">{escape(day_label)}</div>',
            f'<div class="day-badge" style="background:{color}; color:{text_color};">{escape(category)}</div>',
            '</div>',
            f'<div class="day-time">{start_label} → {end_label}</div>',
            f'<div class="day-value">{aqi:.0f}</div>',
            f'<div class="day-advice">{escape(advice)}</div>',
            '</div>',
        ]
    )


def render_pollutant_card(name: str, col: str, unit: str, desc: str, value: float) -> None:
    level = get_pollutant_level(col, value)
    color = AQI_COLORS.get(level, AQI_COLORS["Unknown"])
    text_color = AQI_TEXT_COLORS.get(level, "#0f172a")

    render_html(
        [
            '<div class="pollutant-card">',
            f'<div class="pollutant-name">{escape(name)}</div>',
            f'<div class="pollutant-desc">{escape(desc)}</div>',
            f'<div class="pollutant-value">{value:.2f}</div>',
            f'<div class="section-subtitle" style="margin:0;">{escape(unit)}</div>',
            f'<div class="pollutant-level" style="background:{color}; color:{text_color};">{escape(level)}</div>',
            '</div>',
        ]
    )


def get_chart_y_range(hourly_df: pd.DataFrame) -> tuple[float, float]:
    y_values = pd.to_numeric(hourly_df["predicted_aqi"], errors="coerce").dropna()

    if "openmeteo_us_aqi_reference" in hourly_df.columns:
        ref_values = pd.to_numeric(hourly_df["openmeteo_us_aqi_reference"], errors="coerce").dropna()
        y_values = pd.concat([y_values, ref_values], ignore_index=True)

    if y_values.empty:
        return 0.0, 100.0

    actual_min = float(y_values.min())
    actual_max = float(y_values.max())

    padding = max(8.0, (actual_max - actual_min) * 0.40)

    y_min = max(0.0, actual_min - padding)
    y_max = min(500.0, actual_max + padding)

    if y_max - y_min < 30:
        center = (y_min + y_max) / 2
        y_min = max(0.0, center - 15)
        y_max = min(500.0, center + 15)

    return y_min, y_max


def render_hourly_chart(hourly_df: pd.DataFrame) -> None:
    y_min, y_max = get_chart_y_range(hourly_df)

    fig = go.Figure()

    fig.add_hrect(y0=0, y1=50, fillcolor="#22c55e", opacity=0.10, line_width=0)
    fig.add_hrect(y0=50, y1=100, fillcolor="#facc15", opacity=0.13, line_width=0)
    fig.add_hrect(y0=100, y1=150, fillcolor="#fb923c", opacity=0.10, line_width=0)
    fig.add_hrect(y0=150, y1=200, fillcolor="#ef4444", opacity=0.08, line_width=0)
    fig.add_hrect(y0=200, y1=300, fillcolor="#a855f7", opacity=0.07, line_width=0)
    fig.add_hrect(y0=300, y1=500, fillcolor="#7f1d1d", opacity=0.06, line_width=0)

    fig.add_trace(
        go.Scatter(
            x=hourly_df["timestamp"],
            y=hourly_df["predicted_aqi"],
            mode="lines+markers",
            name="Predicted AQI",
            line=dict(width=4, color="#2563eb", shape="spline", smoothing=0.8),
            marker=dict(size=7, color="#2563eb"),
        )
    )

    if "openmeteo_us_aqi_reference" in hourly_df.columns:
        fig.add_trace(
            go.Scatter(
                x=hourly_df["timestamp"],
                y=hourly_df["openmeteo_us_aqi_reference"],
                mode="lines",
                name="Open-Meteo AQI Reference",
                line=dict(width=2.5, dash="dash", color="#64748b"),
            )
        )

    fig.update_layout(
        height=450,
        template="plotly_white",
        margin=dict(l=20, r=20, t=35, b=20),
        xaxis_title="Forecast Time",
        yaxis_title="AQI",
        yaxis=dict(range=[y_min, y_max], fixedrange=False),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.65)",
        ),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.82)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


def render_pollutant_bar_chart(hourly_df: pd.DataFrame) -> None:
    rows = []

    for name, col, unit, _ in POLLUTANT_COLUMNS:
        if col in hourly_df.columns:
            rows.append(
                {
                    "Pollutant": name,
                    "Value": float(pd.to_numeric(hourly_df[col], errors="coerce").mean()),
                    "Unit": unit,
                }
            )

    if not rows:
        return

    chart_df = pd.DataFrame(rows)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=chart_df["Pollutant"],
            y=chart_df["Value"],
            text=[f"{value:.2f} {unit}" for value, unit in zip(chart_df["Value"], chart_df["Unit"])],
            textposition="outside",
            marker_color="#2563eb",
            name="72h Average",
        )
    )

    fig.update_layout(
        height=390,
        template="plotly_white",
        margin=dict(l=20, r=20, t=28, b=20),
        xaxis_title="Pollutant",
        yaxis_title="Average level",
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.75)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


def render_sidebar() -> dict:
    with st.sidebar:
        st.header("⚙️ Dashboard Settings")

        city_name = get_setting("CITY_NAME", "Hyderabad")
        country_name = get_setting("COUNTRY_NAME", "Pakistan")
        timezone_name = get_setting("TIMEZONE", "Asia/Karachi")

        hopsworks_host = get_setting("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai")
        hopsworks_project = get_setting("HOPSWORKS_PROJECT")
        hopsworks_api_key = get_setting("HOPSWORKS_API_KEY")

        forecast_fg_name = get_setting("FORECAST_FEATURE_GROUP_NAME", "aqi_openmeteo_12f_forecast_fg")
        forecast_fg_version = int(get_setting("FORECAST_FEATURE_GROUP_VERSION", "1"))

        model_name = get_setting("MODEL_NAME", "aqi_openmeteo_12f_best_model")
        model_version = get_setting("MODEL_VERSION", None)

        prediction_hours = int(get_setting("PREDICTION_HOURS", "72"))

        st.caption("Connected project")
        st.write(f"**City:** {city_name}")
        st.write(f"**Timezone:** {timezone_name}")
        st.write(f"**Forecast FG:** `{forecast_fg_name}` v{forecast_fg_version}")
        st.write(f"**Model:** `{model_name}`")

        if st.button("🔄 Refresh from Hopsworks", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    return {
        "city_name": city_name,
        "country_name": country_name,
        "timezone_name": timezone_name,
        "hopsworks_host": hopsworks_host,
        "hopsworks_project": hopsworks_project,
        "hopsworks_api_key": hopsworks_api_key,
        "forecast_fg_name": forecast_fg_name,
        "forecast_fg_version": forecast_fg_version,
        "model_name": model_name,
        "model_version": model_version,
        "prediction_hours": prediction_hours,
    }


def main() -> None:
    inject_css()

    settings = render_sidebar()

    missing = []
    if not settings["hopsworks_project"]:
        missing.append("HOPSWORKS_PROJECT")
    if not settings["hopsworks_api_key"]:
        missing.append("HOPSWORKS_API_KEY")
    if not settings["hopsworks_host"]:
        missing.append("HOPSWORKS_HOST")

    if missing:
        st.error(f"Missing required secrets/env values: {', '.join(missing)}")
        st.stop()

    try:
        with st.spinner("Connecting to Hopsworks..."):
            project = connect_to_hopsworks(
                host=settings["hopsworks_host"],
                project_name=settings["hopsworks_project"],
                api_key=settings["hopsworks_api_key"],
            )

        with st.spinner("Loading latest model from Hopsworks Model Registry..."):
            model_bundle = load_model_from_registry(
                project,
                model_name=settings["model_name"],
                model_version=settings["model_version"],
            )

        with st.spinner("Loading latest forecast features from Hopsworks..."):
            raw_forecast_df = load_forecast_features_from_hopsworks(
                project,
                fg_name=settings["forecast_fg_name"],
                fg_version=settings["forecast_fg_version"],
            )

        forecast_df = prepare_prediction_dataframe(
            raw_df=raw_forecast_df,
            feature_columns=model_bundle["feature_columns"],
            prediction_hours=settings["prediction_hours"],
            timezone_name=settings["timezone_name"],
        )

        hourly_predictions = predict_aqi(
            forecast_df=forecast_df,
            model=model_bundle["model"],
            feature_columns=model_bundle["feature_columns"],
        )

        three_day_summary = make_three_day_summary(hourly_predictions)

    except Exception as error:
        st.error("App failed while loading data/model or generating predictions.")
        st.exception(error)
        st.stop()

    render_hero(settings["city_name"], settings["country_name"], hourly_predictions)

    first_aqi = float(hourly_predictions.iloc[0]["predicted_aqi"])
    max_aqi = float(hourly_predictions["predicted_aqi"].max())
    avg_aqi = float(hourly_predictions["predicted_aqi"].mean())

    first_category = get_aqi_category(first_aqi)
    max_category = get_aqi_category(max_aqi)
    avg_category = get_aqi_category(avg_aqi)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Current AQI", f"{first_aqi:.0f}", first_category)
    metric_col2.metric("3-Day Average", f"{avg_aqi:.0f}", avg_category)
    metric_col3.metric("Peak AQI", f"{max_aqi:.0f}", max_category)
    metric_col4.metric("Forecast Window", f"{len(hourly_predictions)}h", "Next 3 days")

    if max_aqi > 150:
        st.error(f"⚠️ AQI may become **{max_category}** in the next 72 hours. Limit outdoor exposure.")
    elif max_aqi > 100:
        st.warning(f"⚠️ AQI may reach **{max_category}**. Sensitive groups should be careful.")
    else:
        st.success("✅ AQI forecast looks mostly safe for the next 3 days.")

    render_section("📅 Next 3 Days AQI", "Daily cards are calculated from 24-hour forecast blocks.")

    day_cols = st.columns(3)
    for idx, (_, row) in enumerate(three_day_summary.head(3).iterrows()):
        with day_cols[idx]:
            render_day_card(
                day_label=f"Day {int(row['forecast_day'])}",
                start_time=row["start_time"],
                end_time=row["end_time"],
                aqi=float(row["predicted_aqi"]),
                category=str(row["aqi_category"]),
            )

    render_section(
        "📈 Hourly AQI Trend",
        "The chart is zoomed around your actual predicted AQI range, so small changes are visible.",
    )
    render_hourly_chart(hourly_predictions)

    render_section("🧪 Pollutant Levels", "72-hour average pollutant values with AQI-style level badges.")

    pollutant_values = {}
    for name, col, unit, desc in POLLUTANT_COLUMNS:
        if col in hourly_predictions.columns:
            pollutant_values[col] = {
                "name": name,
                "unit": unit,
                "desc": desc,
                "value": float(pd.to_numeric(hourly_predictions[col], errors="coerce").mean()),
            }

    if not pollutant_values:
        st.info("No pollutant columns were found in the forecast Feature Group.")
    else:
        pollutant_cols = st.columns(len(pollutant_values))
        for i, (col, item) in enumerate(pollutant_values.items()):
            with pollutant_cols[i]:
                render_pollutant_card(
                    name=item["name"],
                    col=col,
                    unit=item["unit"],
                    desc=item["desc"],
                    value=item["value"],
                )

        render_pollutant_bar_chart(hourly_predictions)

    render_section("📋 3-Day Forecast Table")

    daily_display = three_day_summary.copy()

    if "start_time" in daily_display.columns:
        daily_display["start_time"] = daily_display["start_time"].dt.strftime("%Y-%m-%d %H:%M")

    if "end_time" in daily_display.columns:
        daily_display["end_time"] = daily_display["end_time"].dt.strftime("%Y-%m-%d %H:%M")

    daily_display = daily_display.rename(
        columns={
            "forecast_day": "Forecast Day",
            "start_time": "Start Time",
            "end_time": "End Time",
            "predicted_aqi": "Predicted AQI",
            "aqi_category": "AQI Category",
            "openmeteo_us_aqi_reference": "Open-Meteo AQI Reference",
            "pm25_24h": "PM2.5",
            "pm10_24h": "PM10",
            "o3_8h_ppb": "O₃",
            "co_8h_ppm": "CO",
            "no2_1h_ppb": "NO₂",
        }
    )

    st.dataframe(daily_display, use_container_width=True, hide_index=True)

    with st.expander("Show hourly prediction details"):
        hourly_display_cols = [
            "timestamp",
            "predicted_aqi",
            "aqi_category",
            "pm25_24h",
            "pm10_24h",
            "o3_8h_ppb",
            "co_8h_ppm",
            "no2_1h_ppb",
        ]

        if "openmeteo_us_aqi_reference" in hourly_predictions.columns:
            hourly_display_cols.insert(2, "openmeteo_us_aqi_reference")

        hourly_display_cols = [col for col in hourly_display_cols if col in hourly_predictions.columns]
        hourly_display = hourly_predictions[hourly_display_cols].copy()

        if "timestamp" in hourly_display.columns:
            hourly_display["timestamp"] = hourly_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")

        hourly_display = hourly_display.rename(
            columns={
                "timestamp": "Time",
                "predicted_aqi": "Predicted AQI",
                "openmeteo_us_aqi_reference": "Open-Meteo AQI Reference",
                "aqi_category": "AQI Category",
                "pm25_24h": "PM2.5",
                "pm10_24h": "PM10",
                "o3_8h_ppb": "O₃",
                "co_8h_ppm": "CO",
                "no2_1h_ppb": "NO₂",
            }
        )

        st.dataframe(hourly_display, use_container_width=True, hide_index=True)

    with st.expander("Model and data source info"):
        st.write("**Best model:**", model_bundle.get("best_model_name"))
        st.write("**Model registry name:**", settings["model_name"])
        st.write("**Model version:**", model_bundle.get("model_version"))
        st.write("**Model path:**", model_bundle.get("model_path"))
        st.write("**Feature Group:**", f"{settings['forecast_fg_name']} v{settings['forecast_fg_version']}")
        st.write("**Feature count:**", len(model_bundle["feature_columns"]))
        st.write("**Feature columns:**", model_bundle["feature_columns"])

        if model_bundle["metadata"]:
            st.json(model_bundle["metadata"])


if __name__ == "__main__":
    main()