import sys

import mlflow

from mlops_air_quality_forecast.monitoring import run_checks

checks, metrics = run_checks()

for check in checks:
    print(f"[{'OK  ' if check.ok else 'FAIL'}] {check.name}: {check.detail}")

mlflow.set_experiment("no2-monitoring")
with mlflow.start_run(run_name="daily-check"):
    mlflow.log_metrics({k: v for k, v in metrics.items() if v is not None})
    mlflow.set_tag("all_checks_passed", all(c.ok for c in checks))

if not all(c.ok for c in checks):
    sys.exit(1)  # a failing check turns the workflow red and GitHub sends an email
