from datetime import timedelta

import mlflow
import mlflow.lightgbm
import pandas as pd
from mlflow import MlflowClient

from mlops_air_quality_forecast.config import settings
from mlops_air_quality_forecast.features import HISTORY_HOURS, build_inference_features
from mlops_air_quality_forecast.openmeteo import fetch_weather_forecast_live
from mlops_air_quality_forecast.store import read_no2, upsert_predictions

MODEL_NAME = "no2-forecaster"
ALIAS = "champion"


def load_champion():
    """The model version currently holding the champion alias."""
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
    model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    return model, version.version


def score(issue_time: pd.Timestamp) -> pd.DataFrame:
    """Predict the next 24 hours from `issue_time` and store the predictions."""
    model, version = load_champion()

    no2 = read_no2(settings.no2_sensor_id, issue_time - timedelta(hours=HISTORY_HOURS))
    weather = pd.DataFrame(
        fetch_weather_forecast_live(
            settings.station_lat, settings.station_lon, settings.location_key
        )
    )
    weather["observed_at"] = pd.to_datetime(weather["observed_at"], utc=True)

    features = build_inference_features(no2, weather, issue_time)
    # The model knows its own input columns, so scoring can never select the wrong ones
    predicted = model.predict(features[model.feature_name()])

    result = pd.DataFrame(
        {
            "location_key": settings.location_key,
            "issue_time": issue_time,
            "horizon": features["horizon"].astype(int),
            "target_time": features["target_time"],
            "predicted_ugm3": predicted.clip(min=0),  # concentrations can't be negative
            "model_version": str(version),
        }
    )
    upsert_predictions(result.to_dict("records"))
    return result
