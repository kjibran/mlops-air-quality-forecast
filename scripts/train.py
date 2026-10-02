import subprocess
import tempfile
from pathlib import Path

import mlflow
import mlflow.lightgbm
import pandas as pd

from mlops_air_quality_forecast.config import settings
from mlops_air_quality_forecast.data import latest_snapshot_tag, load_snapshot
from mlops_air_quality_forecast.evaluation import (
    fit_climatology,
    mae_by_horizon,
    predict_baselines,
    split_by_time,
)
from mlops_air_quality_forecast.features import HORIZONS, LATENCY_HOURS, build_features
from mlops_air_quality_forecast.model import FEATURES, PARAMS, train_model

EXPERIMENT = "no2-forecast"
RUN_NAME = "lgbm-archive-weather"


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


# Data and features
tag = latest_snapshot_tag()
no2, weather = load_snapshot(tag)
features = build_features(no2, weather, settings.training_start)
train, test, test_start = split_by_time(features)
fit_part, valid, _ = split_by_time(train, test_days=60)

# Model and evaluation
model = train_model(fit_part, valid)
predictions = predict_baselines(test, fit_climatology(train))
predictions["lightgbm"] = model.predict(test[FEATURES])
table = mae_by_horizon(test, predictions)
table["gain_vs_persistence_%"] = 100 * (1 - table["lightgbm"] / table["persistence"])

importance = pd.Series(
    model.booster_.feature_importance(importance_type="gain"), index=FEATURES
).sort_values(ascending=False)
importance = 100 * importance / importance.sum()

print(f"Snapshot {tag}, best iteration: {model.best_iteration_}")
print(table.loc[[1, 3, 6, 12, 24, "all"]].round(2).to_string())

# Log everything to MLflow
mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run(run_name=RUN_NAME):
    mlflow.set_tags(
        {
            "data_snapshot": tag,
            "git_commit": git_commit(),
            "weather_input": "archive (perfect weather, upper bound)",
        }
    )
    mlflow.log_params(
        {
            **PARAMS,
            "training_start": settings.training_start,
            "test_start": f"{test_start:%Y-%m-%d}",
            "latency_hours": LATENCY_HOURS,
            "n_features": len(FEATURES),
            "n_train": len(fit_part),
            "n_valid": len(valid),
            "n_test": len(test),
            "best_iteration": model.best_iteration_,
        }
    )
    for method in ["climatology", "persistence", "last_week", "lightgbm"]:
        mlflow.log_metric(f"mae_{method}", table.loc["all", method])
    mlflow.log_metric(
        "gain_vs_persistence_pct", table.loc["all", "gain_vs_persistence_%"]
    )
    for h in HORIZONS:
        mlflow.log_metric("mae_lightgbm_by_horizon", table.loc[h, "lightgbm"], step=h)
        mlflow.log_metric(
            "mae_persistence_by_horizon", table.loc[h, "persistence"], step=h
        )

    with tempfile.TemporaryDirectory() as tmp:
        table.to_csv(Path(tmp) / "mae_by_horizon.csv")
        importance.to_csv(Path(tmp) / "feature_importance.csv", header=["gain_pct"])
        mlflow.log_artifacts(tmp, artifact_path="evaluation")

    mlflow.lightgbm.log_model(model.booster_, name="model")

print("Logged to MLflow.")
