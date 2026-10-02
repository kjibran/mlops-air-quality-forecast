import lightgbm as lgb
import pandas as pd

FEATURES = [
    "horizon",
    "no2_last",
    "no2_mean_6h",
    "no2_mean_24h",
    "no2_same_hour_yesterday",
    "no2_same_hour_last_week",
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    "temperature",
    "humidity",
    "boundary_layer_height",
    "hour",
    "weekday",
    "month",
    "is_holiday",
]

FEATURES_NO_BLH = [f for f in FEATURES if f != "boundary_layer_height"]

PARAMS = {
    "objective": "l1",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_estimators": 3000,
    "verbose": -1,
}


def train_model(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    features: list[str] = FEATURES,
    params: dict = PARAMS,
) -> lgb.LGBMRegressor:
    """Fit LightGBM with early stopping on the validation period."""
    model = lgb.LGBMRegressor(**params)
    model.fit(
        train[features],
        train["target"],
        eval_X=valid[features],
        eval_y=valid["target"],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model
