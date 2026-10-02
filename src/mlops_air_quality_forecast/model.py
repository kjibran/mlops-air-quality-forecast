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

# Boundary layer height is not available as a historical forecast, so production models skip it
FEATURES_NO_BLH = [f for f in FEATURES if f != "boundary_layer_height"]

# Urban background station (OpenAQ location 5170): regional NO2 level and ozone chemistry
BACKGROUND_FEATURES = ["bg_no2_last", "bg_no2_mean_24h", "bg_o3_last", "bg_o3_mean_24h"]
FEATURES_V2 = FEATURES_NO_BLH + BACKGROUND_FEATURES

# The feature set used for production models. Change this to introduce a new model version.
PRODUCTION_FEATURES = FEATURES_V2

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
