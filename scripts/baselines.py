from mlops_air_quality_forecast.config import settings
from mlops_air_quality_forecast.data import latest_snapshot_tag, load_snapshot
from mlops_air_quality_forecast.evaluation import (
    fit_climatology,
    mae_by_horizon,
    predict_baselines,
    split_by_time,
)
from mlops_air_quality_forecast.features import build_features

tag = latest_snapshot_tag()
no2, weather = load_snapshot(tag)
features = build_features(no2, weather, settings.training_start)
train, test, test_start = split_by_time(features)

climatology = fit_climatology(train)
predictions = predict_baselines(test, climatology)
table = mae_by_horizon(test, predictions)

print(f"Snapshot {tag}, test period from {test_start:%Y-%m-%d}")
print(f"Train rows: {len(train)}, test rows: {len(test)}")
print("MAE (µg/m³) by horizon:")
print(table.loc[[1, 3, 6, 12, 24, "all"]].round(2).to_string())
