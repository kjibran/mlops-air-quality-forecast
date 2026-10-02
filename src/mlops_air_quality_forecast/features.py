import holidays
import numpy as np
import pandas as pd

LATENCY_HOURS = 3  # NO2 arrives about 3 hours late
HORIZONS = range(1, 25)
LOCAL_TZ = "Europe/Copenhagen"


def hourly_no2(no2: pd.DataFrame, start: str) -> pd.Series:
    """NO2 as a complete hourly series (missing hours become NaN)."""
    s = no2.set_index("observed_at")["value_ugm3"].sort_index()
    s = s[s.index >= pd.Timestamp(start, tz="UTC")]
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="h")
    return s.reindex(full_index)


def build_features(
    no2: pd.DataFrame, weather: pd.DataFrame, start: str
) -> pd.DataFrame:
    """One row per (issue time, horizon), using only information available at issue time."""
    s = hourly_no2(no2, start)
    w = weather[weather["source"] == "archive"].set_index("observed_at").sort_index()
    w = w.reindex(s.index)

    # What we actually know at issue time t: values up to t - LATENCY_HOURS
    available = s.shift(LATENCY_HOURS)
    issue_features = pd.DataFrame(
        {
            "no2_last": available,
            "no2_mean_6h": available.rolling(6, min_periods=3).mean(),
            "no2_mean_24h": available.rolling(24, min_periods=12).mean(),
        },
        index=s.index,
    )

    years = range(s.index.min().year, s.index.max().year + 2)
    dk_holidays = set(holidays.Denmark(years=years))

    frames = []
    for h in HORIZONS:
        f = issue_features.copy()
        f["horizon"] = h
        f["target_time"] = s.index + pd.Timedelta(hours=h)
        f["target"] = s.shift(-h).to_numpy()

        # Same hour last week is always known in time (168 > 24 + latency)
        f["no2_same_hour_last_week"] = s.shift(168 - h).to_numpy()
        # Same hour yesterday is only known for short horizons
        if h <= 24 - LATENCY_HOURS:
            f["no2_same_hour_yesterday"] = s.shift(24 - h).to_numpy()
        else:
            f["no2_same_hour_yesterday"] = np.nan

        # Weather at the target hour
        wt = w.shift(-h)
        f["wind_speed"] = wt["wind_speed_10m"].to_numpy()
        direction = np.deg2rad(wt["wind_direction_10m"].to_numpy())
        f["wind_dir_sin"] = np.sin(direction)
        f["wind_dir_cos"] = np.cos(direction)
        f["temperature"] = wt["temperature_2m"].to_numpy()
        f["humidity"] = wt["relative_humidity_2m"].to_numpy()
        f["boundary_layer_height"] = wt["boundary_layer_height"].to_numpy()

        # Calendar at the target hour, in local time (traffic follows local clocks)
        local = f["target_time"].dt.tz_convert(LOCAL_TZ)
        f["hour"] = local.dt.hour
        f["weekday"] = local.dt.dayofweek
        f["month"] = local.dt.month
        f["is_holiday"] = local.dt.date.isin(dk_holidays).astype(int)

        frames.append(f)

    features = pd.concat(frames).rename_axis("issue_time").reset_index()
    return features.dropna(subset=["target"]).reset_index(drop=True)
