from __future__ import annotations

import os
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
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


@st.cache_data(ttl=900, show_spinner=False)
def fetch_predictions(api_url: str, hours: int) -> dict:
    api_url = api_url.rstrip("/")
    endpoint = f"{api_url}/predictions"

    response = requests.get(
        endpoint,
        params={"hours": hours},
        timeout=120,
    )

    response.raise_for_status()
    payload = response.json()

    if payload.get("status") != "success":
        raise ValueError(f"API returned unsuccessful status: {payload}")

    return payload


def payload_to_dataframes(payload: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hourly_df = pd.DataFrame(payload.get("hourly", []))
    daily_df = pd.DataFrame(payload.get("daily", []))
    pollutant_df = pd.DataFrame(payload.get("pollutants", []))

    if hourly_df.empty:
        raise ValueError("API returned no hourly prediction rows.")

    if "timestamp" in hourly_df.columns:
        hourly_df["timestamp"] = pd.to_datetime(hourly_df["timestamp"], errors="coerce")

    for col in ["start_time", "end_time"]:
        if col in daily_df.columns:
            daily_df[col] = pd.to_datetime(daily_df[col], errors="coerce")

    numeric_cols = [
        "predicted_aqi",
        "openmeteo_us_aqi_reference",
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

    for col in numeric_cols:
        if col in hourly_df.columns:
            hourly_df[col] = pd.to_numeric(hourly_df[col], errors="coerce")

        if col in daily_df.columns:
            daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce")

    if "value" in pollutant_df.columns:
        pollutant_df["value"] = pd.to_numeric(pollutant_df["value"], errors="coerce")

    return hourly_df, daily_df, pollutant_df


def render_hero(
    city_name: str,
    country_name: str,
    hourly_predictions: pd.DataFrame,
    summary: dict,
) -> None:
    current_aqi = float(summary.get("current_aqi", hourly_predictions.iloc[0]["predicted_aqi"]))
    avg_aqi = float(summary.get("average_aqi_72h", hourly_predictions["predicted_aqi"].mean()))
    max_aqi = float(summary.get("peak_aqi_72h", hourly_predictions["predicted_aqi"].max()))

    category = str(summary.get("current_category", get_aqi_category(current_aqi)))
    color = AQI_COLORS.get(category, AQI_COLORS["Unknown"])
    text_color = AQI_TEXT_COLORS.get(category, "#0f172a")
    advice = str(summary.get("current_advice", get_aqi_advice(category)))

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

    start_label = pd.to_datetime(start_time).strftime("%d %b, %I:%M %p")
    end_label = pd.to_datetime(end_time).strftime("%d %b, %I:%M %p")

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


def render_pollutant_card(
    name: str,
    col: str,
    unit: str,
    desc: str,
    value: float,
    level: str | None = None,
) -> None:
    if level is None:
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


def render_pollutant_bar_chart(pollutant_df: pd.DataFrame) -> None:
    if pollutant_df.empty:
        return

    chart_df = pollutant_df.copy()

    if not {"name", "value", "unit"}.issubset(chart_df.columns):
        return

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=chart_df["name"],
            y=chart_df["value"],
            text=[f"{value:.2f} {unit}" for value, unit in zip(chart_df["value"], chart_df["unit"])],
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

        api_url = get_setting("FASTAPI_URL", "http://127.0.0.1:8000")
        prediction_hours = int(get_setting("PREDICTION_HOURS", "72"))

        st.caption("Backend API")
        st.write(f"**FastAPI URL:** `{api_url}`")
        st.write(f"**Forecast window:** `{prediction_hours}` hours")

        if st.button("🔄 Refresh from FastAPI", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    return {
        "api_url": api_url,
        "prediction_hours": prediction_hours,
    }


def main() -> None:
    inject_css()

    settings = render_sidebar()

    try:
        with st.spinner("Loading AQI predictions from FastAPI..."):
            payload = fetch_predictions(
                api_url=settings["api_url"],
                hours=settings["prediction_hours"],
            )

        hourly_predictions, three_day_summary, pollutant_df = payload_to_dataframes(payload)

        location = payload.get("location", {})
        summary = payload.get("summary", {})
        model_info = payload.get("model", {})

        city_name = location.get("city", "Hyderabad")
        country_name = location.get("country", "Pakistan")

    except requests.exceptions.ConnectionError:
        st.error(
            "FastAPI backend is not running. Start it first with:\n\n"
            "`uvicorn api.main:app --reload --port 8000`"
        )
        st.stop()

    except requests.exceptions.HTTPError as error:
        st.error("FastAPI returned an error.")
        st.exception(error)
        st.stop()

    except Exception as error:
        st.error("App failed while loading predictions from FastAPI.")
        st.exception(error)
        st.stop()

    render_hero(city_name, country_name, hourly_predictions, summary)

    first_aqi = float(summary.get("current_aqi", hourly_predictions.iloc[0]["predicted_aqi"]))
    max_aqi = float(summary.get("peak_aqi_72h", hourly_predictions["predicted_aqi"].max()))
    avg_aqi = float(summary.get("average_aqi_72h", hourly_predictions["predicted_aqi"].mean()))

    first_category = str(summary.get("current_category", get_aqi_category(first_aqi)))
    max_category = str(summary.get("peak_category", get_aqi_category(max_aqi)))
    avg_category = str(summary.get("average_category", get_aqi_category(avg_aqi)))

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
        "Predictions are served by FastAPI and displayed in the Streamlit dashboard.",
    )
    render_hourly_chart(hourly_predictions)

    render_section("🧪 Pollutant Levels", "72-hour average pollutant values with AQI-style level badges.")

    if pollutant_df.empty:
        st.info("No pollutant values were returned by the FastAPI backend.")
    else:
        pollutant_cols = st.columns(len(pollutant_df))

        for i, (_, row) in enumerate(pollutant_df.iterrows()):
            with pollutant_cols[i]:
                render_pollutant_card(
                    name=str(row.get("name", "Pollutant")),
                    col=str(row.get("column", "")),
                    unit=str(row.get("unit", "")),
                    desc=str(row.get("description", "")),
                    value=float(row.get("value", 0.0)),
                    level=str(row.get("level", "Unknown")),
                )

        render_pollutant_bar_chart(pollutant_df)

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

    with st.expander("FastAPI model and data source info"):
        st.write("**Backend API:**", settings["api_url"])
        st.write("**Model registry name:**", model_info.get("model_name"))
        st.write("**Model version:**", model_info.get("model_version"))
        st.write("**Best model:**", model_info.get("best_model_name"))
        st.write("**Feature count:**", model_info.get("feature_count"))
        st.write("**Feature columns:**", model_info.get("feature_columns"))

        st.json(
            {
                "location": location,
                "summary": summary,
                "model": model_info,
            }
        )


if __name__ == "__main__":
    main()