import numpy as np
import pandas as pd

from mlops_air_quality_forecast.evaluation import split_by_time


def test_train_targets_never_reach_into_test_period():
    issue = pd.date_range("2024-01-01", periods=24 * 400, freq="h", tz="UTC")
    rows = []
    for h in (1, 12, 24):
        rows.append(
            pd.DataFrame(
                {
                    "issue_time": issue,
                    "horizon": h,
                    "target_time": issue + pd.Timedelta(hours=h),
                }
            )
        )
    features = pd.concat(rows, ignore_index=True)
    features["target"] = np.zeros(len(features))

    train, test, test_start = split_by_time(features, test_days=30)

    assert (train["target_time"] < test_start).all()
    assert (test["issue_time"] >= test_start).all()
    assert len(train) > 0 and len(test) > 0
