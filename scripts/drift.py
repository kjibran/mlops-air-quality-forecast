import tempfile
from pathlib import Path

import mlflow

from mlops_air_quality_forecast.config import settings
from mlops_air_quality_forecast.data import latest_snapshot_tag, load_snapshot
from mlops_air_quality_forecast.drift import drift_report
from mlops_air_quality_forecast.features import build_features

tag = latest_snapshot_tag()
no2, weather = load_snapshot(tag)
features = build_features(no2, weather, settings.training_start, "forecast_day1")
report = drift_report(features)

print(f"Snapshot {tag}: current week vs. variability of past weeks")
print(report.round(3).to_string(index=False))

unusual = report.loc[report["status"] == "unusual", "feature"]

mlflow.set_experiment("no2-monitoring")
with mlflow.start_run(run_name="weekly-drift"):
    mlflow.set_tag("data_snapshot", tag)
    mlflow.set_tag("unusual_features", ",".join(unusual) or "none")
    mlflow.log_metrics({f"psi_{r.feature}": r.psi for r in report.itertuples()})
    mlflow.log_metrics(
        {f"psi_95pct_{r.feature}": r.typical_95pct for r in report.itertuples()}
    )
    with tempfile.TemporaryDirectory() as tmp:
        report.to_csv(Path(tmp) / "drift_report.csv", index=False)
        mlflow.log_artifacts(tmp, artifact_path="drift")
