import pandas as pd

from mlops_air_quality_forecast.scoring import score

issue_time = pd.Timestamp.now(tz="UTC").floor("h")
result = score(issue_time)
print(
    f"Issue time {issue_time:%Y-%m-%d %H:%M} UTC, model version {result['model_version'].iloc[0]}"
)
print(
    result[["horizon", "target_time", "predicted_ugm3"]]
    .round({"predicted_ugm3": 1})
    .to_string(index=False)
)
