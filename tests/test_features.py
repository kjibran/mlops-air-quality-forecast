import numpy as np
import pandas as pd

from mlops_air_quality_forecast.features import (
    LATENCY_HOURS,
    build_features,
    build_inference_features,
)

START = "2024-01-01"
N_HOURS = 24 * 14  # two weeks
NO2_FEATURES = [
    "no2_last",
    "no2_mean_6h",
    "no2_mean_24h",
    "no2_same_hour_yesterday",
    "no2_same_hour_last_week",
]
BACKGROUND_FEATURES = ["bg_no2_last", "bg_no2_mean_24h", "bg_o3_last", "bg_o3_mean_24h"]


def make_data():
    """Fake data where every pollutant value equals its own hour position (0, 1, 2, ...)."""
    times = pd.date_range(START, periods=N_HOURS, freq="h", tz="UTC")
    positions = np.arange(N_HOURS, dtype=float)
    no2 = pd.DataFrame({"observed_at": times, "value_ugm3": positions})
    weather = pd.DataFrame(
        {
            "observed_at": times,
            "source": "archive",
            "wind_speed_10m": 3.0,
            "wind_direction_10m": 90.0,
            "temperature_2m": 10.0,
            "relative_humidity_2m": 80.0,
            "boundary_layer_height": 500.0,
        }
    )
    background = pd.concat(
        [
            pd.DataFrame(
                {"observed_at": times, "parameter": p, "value_ugm3": positions}
            )
            for p in ("no2", "o3")
        ],
        ignore_index=True,
    )
    return no2, weather, background


def training_features():
    no2, weather, background = make_data()
    return build_features(no2, weather, START, background=background)


def position(times):
    """Hours since START, matching the fake pollutant values."""
    return ((times - pd.Timestamp(START, tz="UTC")) / pd.Timedelta(hours=1)).astype(int)


def test_target_is_exactly_horizon_hours_ahead():
    f = training_features()
    assert (f["target"] == position(f["issue_time"]) + f["horizon"]).all()


def test_no_pollutant_feature_uses_data_after_latency():
    f = training_features()
    latest_allowed = position(f["issue_time"]) - LATENCY_HOURS
    for col in NO2_FEATURES + BACKGROUND_FEATURES:
        known = f[col].notna()
        assert (f.loc[known, col] <= latest_allowed[known]).all(), col


def test_background_features_are_filled():
    f = training_features()
    for col in BACKGROUND_FEATURES:
        assert f[col].notna().any(), col


def test_yesterday_is_empty_for_long_horizons():
    f = training_features()
    long = f["horizon"] > 24 - LATENCY_HOURS
    assert f.loc[long, "no2_same_hour_yesterday"].isna().all()
    assert f.loc[~long, "no2_same_hour_yesterday"].notna().any()


def test_calendar_uses_copenhagen_local_time():
    f = training_features()
    # In January, Copenhagen is UTC+1, so 06:00 UTC is 07:00 local
    row = f[f["target_time"] == pd.Timestamp("2024-01-05 06:00", tz="UTC")].iloc[0]
    assert row["hour"] == 7


def test_new_year_is_a_holiday():
    f = training_features()
    new_year = f[
        f["target_time"].dt.tz_convert("Europe/Copenhagen").dt.date
        == pd.Timestamp("2024-01-01").date()
    ]
    assert (new_year["is_holiday"] == 1).all()


def test_inference_features_match_training_features():
    no2, weather, background = make_data()
    training = build_features(no2, weather, START, background=background)
    issue = pd.Timestamp(START, tz="UTC") + pd.Timedelta(days=10)

    expected = training[training["issue_time"] == issue].drop(columns="target")
    actual = build_inference_features(no2, weather, issue, background).drop(
        columns="target"
    )

    assert len(actual) == 24
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )
