from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import hopsworks
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "us_aqi"

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


def load_config() -> dict[str, Any]:
    load_dotenv(ENV_PATH)

    return {
        "hopsworks_host": os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
        "hopsworks_project": os.getenv("HOPSWORKS_PROJECT", os.getenv("HOPSWORKS_PROJECT_NAME")),
        "hopsworks_api_key": os.getenv("HOPSWORKS_API_KEY"),
        "feature_group_name": os.getenv(
            "FEATURE_GROUP_NAME",
            os.getenv("HOPSWORKS_FEATURE_GROUP", "aqi_openmeteo_12f_training_fg"),
        ),
        "feature_group_version": int(
            os.getenv("FEATURE_GROUP_VERSION", os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "1"))
        ),
        "model_name": os.getenv("MODEL_NAME", "aqi_openmeteo_12f_best_model"),
        "model_output_path": PROJECT_ROOT / os.getenv("MODEL_OUTPUT_PATH", "models/best_model.pkl"),
        "metrics_output_path": PROJECT_ROOT / os.getenv("METRICS_OUTPUT_PATH", "reports/model_metrics.csv"),
        "metadata_output_path": PROJECT_ROOT / os.getenv("MODEL_METADATA_OUTPUT_PATH", "reports/model_metadata.json"),
        "selected_features_output_path": PROJECT_ROOT / os.getenv(
            "SELECTED_FEATURES_OUTPUT_PATH", "reports/selected_features.csv"
        ),
        "feature_importance_output_path": PROJECT_ROOT / os.getenv(
            "FEATURE_IMPORTANCE_OUTPUT_PATH", "reports/feature_importance.csv"
        ),
    }


def validate_config(cfg: dict[str, Any], register_model: bool) -> None:
    required = [
        "hopsworks_host",
        "hopsworks_project",
        "hopsworks_api_key",
        "feature_group_name",
        "feature_group_version",
    ]
    if register_model:
        required.append("model_name")

    missing = [key for key in required if not cfg.get(key)]
    if missing:
        raise ValueError(f"Missing .env values: {missing}")

    logger.info("Hopsworks project: %s", cfg["hopsworks_project"])
    logger.info("Training FG: %s v%s", cfg["feature_group_name"], cfg["feature_group_version"])
    logger.info("Model registry name: %s", cfg["model_name"])
    logger.info("Feature count: %s", len(MODEL_FEATURE_COLUMNS))


def connect_to_hopsworks(cfg: dict[str, Any]):
    return hopsworks.login(
        host=cfg["hopsworks_host"],
        project=cfg["hopsworks_project"],
        api_key_value=cfg["hopsworks_api_key"],
        engine="python",
    )


def read_training_data_from_hopsworks(cfg: dict[str, Any]) -> pd.DataFrame:
    project = connect_to_hopsworks(cfg)
    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name=cfg["feature_group_name"],
        version=cfg["feature_group_version"],
    )

    selected_columns = ["city", "timestamp", *MODEL_FEATURE_COLUMNS, TARGET_COLUMN]

    logger.info("Reading training features from Hopsworks...")
    query = fg.select(selected_columns)

    try:
        df = query.read(dataframe_type="pandas", read_options={"use_hive": True})
    except Exception as error:
        logger.warning("Hive read failed, trying default read. Reason: %s", error)
        df = query.read(dataframe_type="pandas")

    if df.empty:
        raise ValueError("Training dataframe read from Hopsworks is empty.")

    logger.info("Loaded training dataframe shape: %s", df.shape)
    return df


def prepare_training_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required_columns = ["city", "timestamp", *MODEL_FEATURE_COLUMNS, TARGET_COLUMN]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns from Hopsworks data: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    for col in MODEL_FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before_rows = len(df)
    df = (
        df.sort_values(["city", "timestamp"])
        .drop_duplicates(subset=["city", "timestamp"], keep="last")
        .reset_index(drop=True)
    )
    df = df.dropna(subset=MODEL_FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No rows left after cleaning training dataframe.")

    logger.info("Rows before cleaning: %s", before_rows)
    logger.info("Rows after cleaning : %s", len(df))
    logger.info("Training date range : %s to %s", df["timestamp"].min(), df["timestamp"].max())
    return df


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_index = int(len(df) * 0.80)
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    logger.info("Train rows: %s | %s to %s", len(train_df), train_df["timestamp"].min(), train_df["timestamp"].max())
    logger.info("Test rows : %s | %s to %s", len(test_df), test_df["timestamp"].min(), test_df["timestamp"].max())
    return train_df, test_df


def make_preprocessor(scale: bool) -> ColumnTransformer:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[("num", Pipeline(steps), MODEL_FEATURE_COLUMNS)],
        remainder="drop",
    )


def make_pipeline(model, scale: bool) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(scale=scale)),
            ("model", model),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    models = {
        "Ridge Regression": make_pipeline(Ridge(alpha=1.0), scale=True),
        "Random Forest": make_pipeline(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=14,
                min_samples_split=8,
                min_samples_leaf=3,
                max_features=0.85,
                random_state=42,
                n_jobs=-1,
            ),
            scale=False,
        ),
    }

    if XGBRegressor is not None:
        models["XGBoost"] = make_pipeline(
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=2.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            ),
            scale=False,
        )
    else:
        logger.warning("XGBoost is not installed. Install with: pip install xgboost")

    return models


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate_models(train_df: pd.DataFrame, test_df: pd.DataFrame):
    models = build_models()

    X_train = train_df[MODEL_FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df[MODEL_FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    rows = []
    trained_models = {}

    print("\n" + "=" * 80)
    print("Training and Evaluating Models")
    print("=" * 80)

    for model_name, model in models.items():
        logger.info("Training %s...", model_name)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_metrics = calculate_metrics(y_train, train_pred)
        test_metrics = calculate_metrics(y_test, test_pred)

        rows.append(
            {
                "model_name": model_name,
                "train_MAE": train_metrics["MAE"],
                "train_RMSE": train_metrics["RMSE"],
                "train_R2": train_metrics["R2"],
                "test_MAE": test_metrics["MAE"],
                "test_RMSE": test_metrics["RMSE"],
                "test_R2": test_metrics["R2"],
            }
        )
        trained_models[model_name] = model

        print(
            f"{model_name:<18} | "
            f"Test MAE: {test_metrics['MAE']:.3f} | "
            f"Test RMSE: {test_metrics['RMSE']:.3f} | "
            f"Test R²: {test_metrics['R2']:.4f}"
        )

    metrics_df = pd.DataFrame(rows).sort_values(["test_RMSE", "test_MAE"], ascending=[True, True])
    best_model_name = str(metrics_df.iloc[0]["model_name"])
    best_model = trained_models[best_model_name]

    print("\n" + "=" * 80)
    print("Model Comparison")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print("\n" + "=" * 80)
    print(f"Best model selected: {best_model_name}")
    print("=" * 80)

    return best_model_name, best_model, metrics_df


def retrain_best_model_on_full_data(best_model_name: str, full_df: pd.DataFrame) -> Pipeline:
    logger.info("Retraining best model on full dataset: %s", best_model_name)
    model = build_models()[best_model_name]
    model.fit(full_df[MODEL_FEATURE_COLUMNS], full_df[TARGET_COLUMN])
    return model


def get_feature_importance(best_model: Pipeline, best_model_name: str) -> pd.DataFrame:
    fitted_model = best_model.named_steps["model"]

    if hasattr(fitted_model, "feature_importances_"):
        importance = fitted_model.feature_importances_
    elif hasattr(fitted_model, "coef_"):
        importance = np.abs(fitted_model.coef_)
    else:
        importance = np.zeros(len(MODEL_FEATURE_COLUMNS))

    return (
        pd.DataFrame(
            {
                "feature": MODEL_FEATURE_COLUMNS,
                "importance": importance,
                "model_name": best_model_name,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_local_artifacts(
    best_model: Pipeline,
    best_model_name: str,
    metrics_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    training_df: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    for key in [
        "model_output_path",
        "metrics_output_path",
        "metadata_output_path",
        "selected_features_output_path",
        "feature_importance_output_path",
    ]:
        cfg[key].parent.mkdir(parents=True, exist_ok=True)

    model_bundle = {
        "model": best_model,
        "best_model_name": best_model_name,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(model_bundle, cfg["model_output_path"])

    metrics_df.to_csv(cfg["metrics_output_path"], index=False)
    feature_importance_df.to_csv(cfg["feature_importance_output_path"], index=False)

    selected_features_df = pd.DataFrame(
        {
            "feature": MODEL_FEATURE_COLUMNS,
            "selected": True,
            "rank": range(1, len(MODEL_FEATURE_COLUMNS) + 1),
        }
    )
    selected_features_df.to_csv(cfg["selected_features_output_path"], index=False)

    best_row = metrics_df.iloc[0].to_dict()
    metadata = {
        "model_registry_name": cfg["model_name"],
        "best_model_name": best_model_name,
        "target_column": TARGET_COLUMN,
        "feature_columns": MODEL_FEATURE_COLUMNS,
        "feature_count": len(MODEL_FEATURE_COLUMNS),
        "training_rows": int(len(training_df)),
        "training_start": str(training_df["timestamp"].min()),
        "training_end": str(training_df["timestamp"].max()),
        "metrics": {
            "MAE": float(best_row["test_MAE"]),
            "RMSE": float(best_row["test_RMSE"]),
            "R2": float(best_row["test_R2"]),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Best AQI model selected from Ridge Regression, Random Forest, and XGBoost. "
            "Model uses 12 derived Open-Meteo pollutant/weather features and predicts Open-Meteo us_aqi."
        ),
    }

    with open(cfg["metadata_output_path"], "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    logger.info("Saved best model: %s", cfg["model_output_path"])
    logger.info("Saved metrics: %s", cfg["metrics_output_path"])
    logger.info("Saved metadata: %s", cfg["metadata_output_path"])
    logger.info("Saved selected features: %s", cfg["selected_features_output_path"])
    logger.info("Saved feature importance: %s", cfg["feature_importance_output_path"])
    return metadata


def register_model_to_hopsworks(
    best_model: Pipeline,
    best_model_name: str,
    metrics_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    metadata: dict[str, Any],
    cfg: dict[str, Any],
) -> None:
    project = connect_to_hopsworks(cfg)
    mr = project.get_model_registry()

    best_row = metrics_df.iloc[0]
    registry_metrics = {
        "mae": float(best_row["test_MAE"]),
        "rmse": float(best_row["test_RMSE"]),
        "r2": float(best_row["test_R2"]),
        "feature_count": float(len(MODEL_FEATURE_COLUMNS)),
    }

    logger.info("Registering best model to Hopsworks Model Registry...")
    logger.info("Model registry name: %s", cfg["model_name"])
    logger.info("Winning algorithm: %s", best_model_name)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        joblib.dump(best_model, temp_dir / "model.pkl")

        with open(temp_dir / "feature_columns.json", "w", encoding="utf-8") as file:
            json.dump(MODEL_FEATURE_COLUMNS, file, indent=4)

        with open(temp_dir / "model_metadata.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        metrics_df.to_csv(temp_dir / "model_metrics.csv", index=False)
        feature_importance_df.to_csv(temp_dir / "feature_importance.csv", index=False)

        model_obj = mr.python.create_model(
            name=cfg["model_name"],
            metrics=registry_metrics,
            description=(
                "Best AQI model selected from Ridge Regression, Random Forest, and XGBoost. "
                "Uses 12 derived Open-Meteo features to predict us_aqi."
            ),
        )
        saved_model = model_obj.save(str(temp_dir))

    logger.info("Model registered successfully: %s", saved_model)


def run_training_pipeline(register_model: bool = True) -> pd.DataFrame:
    cfg = load_config()
    validate_config(cfg, register_model=register_model)

    print("\n" + "=" * 80)
    print("AQI Training Pipeline — 12 Derived Open-Meteo Features")
    print("=" * 80)

    raw_df = read_training_data_from_hopsworks(cfg)
    training_df = prepare_training_dataframe(raw_df)
    train_df, test_df = time_based_split(training_df)

    best_model_name, _, metrics_df = train_and_evaluate_models(train_df, test_df)
    best_model = retrain_best_model_on_full_data(best_model_name, training_df)
    feature_importance_df = get_feature_importance(best_model, best_model_name)

    metadata = save_local_artifacts(
        best_model=best_model,
        best_model_name=best_model_name,
        metrics_df=metrics_df,
        feature_importance_df=feature_importance_df,
        training_df=training_df,
        cfg=cfg,
    )

    if register_model:
        register_model_to_hopsworks(
            best_model=best_model,
            best_model_name=best_model_name,
            metrics_df=metrics_df,
            feature_importance_df=feature_importance_df,
            metadata=metadata,
            cfg=cfg,
        )
    else:
        logger.info("Model registry upload skipped because --no-register was used.")

    print("\n" + "=" * 80)
    print("Training completed successfully.")
    print(f"Best model: {best_model_name}")
    print("=" * 80)
    return metrics_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    run_training_pipeline(register_model=not args.no_register)


if __name__ == "__main__":
    main()
