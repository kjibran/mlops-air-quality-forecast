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
from mlops_air_quality_forecast.model import (
    FEATURES,
    FEATURES_NO_BLH,
    PARAMS,
    train_model,
)

EXPERIMENT = "no2-forecast"

# (run name, feature set, weather used for evaluation, description)
EXPERIMENTS = [
    (
        "lgbm-archive-weather",
        FEATURES,
        "archive",
        "Upper bound: perfect weather incl. BLH",
    ),
    (
        "lgbm-archive-weather-no-blh",
        FEATURES_NO_BLH,
        "archive",
        "Perfect weather, no BLH",
    ),
    (
        "lgbm-forecast-weather-no-blh",
        FEATURES_NO_BLH,
        "forecast_day1",
        "Realistic: Day 1 forecast weather",
    ),
]


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


tag = latest_snapshot_tag()
no2, weather = load_snapshot(tag)
print(f"Snapshot {tag}")

# Training always uses archive weather; evaluation uses the source named per experiment
archive_features = build_features(no2, weather, settings.training_start, "archive")
train, archive_test, test_start = split_by_time(archive_features)
fit_part, valid, _ = split_by_time(train, test_days=60)
climatology = fit_climatology(train)

forecast_features = build_features(
    no2, weather, settings.training_start, "forecast_day1"
)
test_sets = {
    "archive": archive_test,
    "forecast_day1": forecast_features[
        forecast_features["issue_time"] >= test_start
    ].reset_index(drop=True),
}

models = {}  # train each feature set only once
mlflow.set_experiment(EXPERIMENT)

for run_name, features, eval_source, description in EXPERIMENTS:
    key = tuple(features)
    if key not in models:
        models[key] = train_model(fit_part, valid, features)
    model = models[key]
    test = test_sets[eval_source]

    predictions = predict_baselines(test, climatology)
    predictions["lightgbm"] = model.predict(test[features])
    table = mae_by_horizon(test, predictions)
    table["gain_vs_persistence_%"] = 100 * (
        1 - table["lightgbm"] / table["persistence"]
    )

    importance = pd.Series(
        model.booster_.feature_importance(importance_type="gain"), index=features
    ).sort_values(ascending=False)
    importance = 100 * importance / importance.sum()

    print(f"\n{run_name}: {description}")
    print(
        table.loc[
            [1, 6, 12, 24, "all"], ["persistence", "lightgbm", "gain_vs_persistence_%"]
        ]
        .round(2)
        .to_string()
    )

    with mlflow.start_run(run_name=run_name, description=description):
        mlflow.set_tags(
            {
                "data_snapshot": tag,
                "git_commit": git_commit(),
                "train_weather": "archive",
                "eval_weather": eval_source,
            }
        )
        mlflow.log_params(
            {
                **PARAMS,
                "features": ",".join(features),
                "training_start": settings.training_start,
                "test_start": f"{test_start:%Y-%m-%d}",
                "latency_hours": LATENCY_HOURS,
                "n_features": len(features),
                "n_train": len(fit_part),
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
            mlflow.log_metric(
                "mae_lightgbm_by_horizon", table.loc[h, "lightgbm"], step=h
            )
            mlflow.log_metric(
                "mae_persistence_by_horizon", table.loc[h, "persistence"], step=h
            )

        with tempfile.TemporaryDirectory() as tmp:
            table.to_csv(Path(tmp) / "mae_by_horizon.csv")
            importance.to_csv(Path(tmp) / "feature_importance.csv", header=["gain_pct"])
            mlflow.log_artifacts(tmp, artifact_path="evaluation")

        mlflow.lightgbm.log_model(model.booster_, name="model")

print("\nAll runs logged to MLflow.")
