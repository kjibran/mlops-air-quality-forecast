import numpy as np
import pandas as pd

from mlops_air_quality_forecast.features import LATENCY_HOURS, build_features

START = "2024-01-01"
N_HOURS = 24 * 14  # two weeks


def make_data():
    times = pd.date_range(START, periods=N_HOURS, freq="h", tz="UTC")
    no2 = pd.DataFrame(
        {"observed_at": times, "value_ugm3": np.arange(N_HOURS, dtype=float)}
    )
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
    return no2, weather


def position(times):
    """Hours since START, matching the fake NO2 values."""
    return ((times - pd.Timestamp(START, tz="UTC")) / pd.Timedelta(hours=1)).astype(int)


def test_target_is_exactly_horizon_hours_ahead():
    f = build_features(*make_data(), START)
    assert (f["target"] == position(f["issue_time"]) + f["horizon"]).all()


def test_no_no2_feature_uses_data_after_latency():
    f = build_features(*make_data(), START)
    latest_allowed = position(f["issue_time"]) - LATENCY_HOURS
    for col in [
        "no2_last",
        "no2_mean_6h",
        "no2_mean_24h",
        "no2_same_hour_yesterday",
        "no2_same_hour_last_week",
    ]:
        known = f[col].notna()
        assert (f.loc[known, col] <= latest_allowed[known]).all(), col


def test_yesterday_is_empty_for_long_horizons():
    f = build_features(*make_data(), START)
    long = f["horizon"] > 24 - LATENCY_HOURS
    assert f.loc[long, "no2_same_hour_yesterday"].isna().all()
    assert f.loc[~long, "no2_same_hour_yesterday"].notna().any()


def test_calendar_uses_copenhagen_local_time():
    f = build_features(*make_data(), START)
    # In January, Copenhagen is UTC+1, so 06:00 UTC is 07:00 local
    row = f[f["target_time"] == pd.Timestamp("2024-01-05 06:00", tz="UTC")].iloc[0]
    assert row["hour"] == 7


def test_new_year_is_a_holiday():
    f = build_features(*make_data(), START)
    new_year = f[
        f["target_time"].dt.tz_convert("Europe/Copenhagen").dt.date
        == pd.Timestamp("2024-01-01").date()
    ]
    assert (new_year["is_holiday"] == 1).all()
