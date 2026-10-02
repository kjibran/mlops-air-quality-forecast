import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient

from mlops_air_quality_forecast.config import settings
from mlops_air_quality_forecast.data import (
    latest_snapshot_tag,
    load_background,
    load_snapshot,
)
from mlops_air_quality_forecast.evaluation import split_by_time
from mlops_air_quality_forecast.features import build_features
from mlops_air_quality_forecast.model import PRODUCTION_FEATURES, train_model
from mlops_air_quality_forecast.scoring import ALIAS, MODEL_NAME, load_champion

EVAL_DAYS = 60
MIN_IMPROVEMENT = 0.02  # challenger must be at least 2% better to be promoted


def eval_mae(booster, rows) -> float:
    """MAE of a model on given rows, using the model's own stored feature names."""
    predicted = booster.predict(rows[booster.feature_name()]).clip(min=0)
    return float(abs(predicted - rows["target"]).mean())


def retrain() -> dict:
    tag = latest_snapshot_tag()
    no2, weather = load_snapshot(tag)
    background = load_background(tag)
    archive = build_features(
        no2, weather, settings.training_start, "archive", background
    )
    forecast = build_features(
        no2, weather, settings.training_start, "forecast_day1", background
    )

    # Train on everything before the evaluation window, evaluate both models inside it
    train, _, eval_start = split_by_time(archive, test_days=EVAL_DAYS)
    fit_part, valid, _ = split_by_time(train, test_days=60)
    eval_rows = forecast[forecast["issue_time"] >= eval_start].reset_index(drop=True)

    challenger = train_model(fit_part, valid, PRODUCTION_FEATURES).booster_
    champion, champion_version = load_champion()

    challenger_mae = eval_mae(challenger, eval_rows)
    champion_mae = eval_mae(champion, eval_rows)
    promote = challenger_mae < champion_mae * (1 - MIN_IMPROVEMENT)

    client = MlflowClient()
    mlflow.set_experiment("no2-forecast")
    with mlflow.start_run(run_name="weekly-retrain") as run:
        mlflow.set_tags(
            {
                "data_snapshot": tag,
                "train_weather": "archive",
                "eval_weather": "forecast_day1",
                "decision": "promoted" if promote else "kept champion",
                "compared_with_version": champion_version,
            }
        )
        mlflow.log_params(
            {
                "features": ",".join(PRODUCTION_FEATURES),
                "n_features": len(PRODUCTION_FEATURES),
                "eval_start": f"{eval_start:%Y-%m-%d}",
                "eval_days": EVAL_DAYS,
                "n_train": len(fit_part),
                "n_eval": len(eval_rows),
                "min_improvement": MIN_IMPROVEMENT,
            }
        )
        mlflow.log_metrics(
            {
                "mae_lightgbm": challenger_mae,  # read by the monitoring check if promoted
                "champion_mae_same_window": champion_mae,
            }
        )
        info = mlflow.lightgbm.log_model(challenger, name="model")

    new_version = mlflow.register_model(info.model_uri, MODEL_NAME).version
    client.set_registered_model_alias(MODEL_NAME, "challenger", new_version)
    if promote:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, new_version)

    return {
        "snapshot": tag,
        "eval_start": f"{eval_start:%Y-%m-%d}",
        "champion_version": champion_version,
        "champion_mae": round(champion_mae, 3),
        "challenger_version": new_version,
        "challenger_features": len(PRODUCTION_FEATURES),
        "challenger_mae": round(challenger_mae, 3),
        "promoted": promote,
        "run_id": run.info.run_id,
    }
