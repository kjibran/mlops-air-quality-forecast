import pandas as pd

from mlops_air_quality_forecast.config import settings
from mlops_air_quality_forecast.data import latest_snapshot_tag, load_snapshot
from mlops_air_quality_forecast.evaluation import (
    fit_climatology,
    mae_by_horizon,
    predict_baselines,
    split_by_time,
)
from mlops_air_quality_forecast.features import build_features
from mlops_air_quality_forecast.model import FEATURES, train_model

tag = latest_snapshot_tag()
no2, weather = load_snapshot(tag)
features = build_features(no2, weather, settings.training_start)

train, test, test_start = split_by_time(features)
fit_part, valid, _ = split_by_time(train, test_days=60)

model = train_model(fit_part, valid)
print(f"Snapshot {tag}, best iteration: {model.best_iteration_}")

predictions = predict_baselines(test, fit_climatology(train))
predictions["lightgbm"] = model.predict(test[FEATURES])
table = mae_by_horizon(test, predictions)
table["gain_vs_persistence_%"] = 100 * (1 - table["lightgbm"] / table["persistence"])

print("MAE (µg/m³) on the test year:")
print(table.loc[[1, 3, 6, 12, 24, "all"]].round(2).to_string())

importance = pd.Series(
    model.booster_.feature_importance(importance_type="gain"), index=FEATURES
).sort_values(ascending=False)
print("\nFeature importance (share of total gain, %):")
print((100 * importance / importance.sum()).round(1).head(10).to_string())
