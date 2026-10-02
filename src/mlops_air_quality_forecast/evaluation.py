import pandas as pd

TEST_DAYS = 365


def split_by_time(features: pd.DataFrame, test_days: int = TEST_DAYS):
    """Test = last `test_days` of issue times. Train = rows whose target lies before the test period."""
    test_start = features["issue_time"].max() - pd.Timedelta(days=test_days)
    train = features[features["target_time"] < test_start]
    test = features[features["issue_time"] >= test_start]
    return train.reset_index(drop=True), test.reset_index(drop=True), test_start


def fit_climatology(train: pd.DataFrame) -> pd.Series:
    """Average NO2 per (local hour, weekday), learned from training data only."""
    return train.groupby(["hour", "weekday"])["target"].mean()


def predict_baselines(df: pd.DataFrame, climatology: pd.Series) -> pd.DataFrame:
    """Baseline predictions for every row, falling back to climatology when inputs are missing."""
    keys = pd.MultiIndex.from_arrays([df["hour"], df["weekday"]])
    clim = pd.Series(climatology.reindex(keys).to_numpy(), index=df.index)
    return pd.DataFrame(
        {
            "climatology": clim,
            "persistence": df["no2_last"].fillna(clim),
            "last_week": df["no2_same_hour_last_week"].fillna(clim),
        }
    )


def mae_by_horizon(df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """MAE per horizon (rows) for each method (columns), plus an 'all' row."""
    errors = predictions.sub(df["target"], axis=0).abs()
    errors["horizon"] = df["horizon"].to_numpy()
    table = errors.groupby("horizon").mean()
    table.loc["all"] = errors.drop(columns="horizon").mean()
    return table
