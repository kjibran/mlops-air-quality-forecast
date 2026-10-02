from dataclasses import dataclass

from mlflow import MlflowClient

from mlops_air_quality_forecast.config import settings
from mlops_air_quality_forecast.db import connect
from mlops_air_quality_forecast.scoring import ALIAS, MODEL_NAME
from mlops_air_quality_forecast.store import read_verification

MAX_NO2_AGE_HOURS = 12
MAX_FORECAST_AGE_HOURS = 3
MAE_ALERT_FACTOR = 1.5
MIN_VERIFIED = 500
WINDOW_DAYS = 7


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _hours_since_latest(sql: str, params: tuple) -> float | None:
    with connect() as conn:
        value = conn.execute(
            f"select extract(epoch from now() - ({sql})) / 3600", params
        ).fetchone()[0]
    return float(value) if value is not None else None


def champion_test_mae() -> tuple[str, float]:
    """The champion's version and its realistic test MAE, from its MLflow run."""
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    run = client.get_run(version.run_id)
    return version.version, run.data.metrics["mae_lightgbm"]


def run_checks() -> tuple[list[Check], dict]:
    metrics = {}

    no2_age = _hours_since_latest(
        "select max(observed_at) from no2_hourly where sensor_id = %s and value_ugm3 is not null",
        (settings.no2_sensor_id,),
    )
    forecast_age = _hours_since_latest(
        "select max(issue_time) from predictions where location_key = %s",
        (settings.location_key,),
    )
    metrics["no2_age_hours"] = no2_age
    metrics["forecast_age_hours"] = forecast_age

    rows = read_verification(settings.no2_sensor_id, settings.location_key, WINDOW_DAYS)
    verified = sum(r[1] for r in rows)
    live_mae = sum(r[1] * r[2] for r in rows) / verified if verified else None
    version, test_mae = champion_test_mae()
    metrics["verified_forecasts"] = verified
    metrics["live_mae_7d"] = live_mae
    metrics["champion_test_mae"] = test_mae

    checks = [
        Check(
            "NO2 data is fresh",
            no2_age is not None and no2_age <= MAX_NO2_AGE_HOURS,
            f"latest value {no2_age:.1f} h old (limit {MAX_NO2_AGE_HOURS} h)"
            if no2_age is not None
            else "no NO2 data",
        ),
        Check(
            "Forecasts are being made",
            forecast_age is not None and forecast_age <= MAX_FORECAST_AGE_HOURS,
            f"latest forecast {forecast_age:.1f} h old (limit {MAX_FORECAST_AGE_HOURS} h)"
            if forecast_age is not None
            else "no forecasts",
        ),
    ]

    if verified < MIN_VERIFIED:
        checks.append(
            Check(
                "Live accuracy",
                True,
                f"skipped: only {verified} verified forecasts (need {MIN_VERIFIED})",
            )
        )
    else:
        limit = MAE_ALERT_FACTOR * test_mae
        checks.append(
            Check(
                "Live accuracy",
                live_mae <= limit,
                f"7-day MAE {live_mae:.2f} vs limit {limit:.2f} (model v{version}, test MAE {test_mae:.2f})",
            )
        )
    return checks, metrics
