import holidays
import numpy as np
import pandas as pd

LATENCY_HOURS = 3  # pollutant data arrives about 3 hours late
HORIZONS = range(1, 25)
LOCAL_TZ = "Europe/Copenhagen"
HISTORY_HOURS = 24 * 8  # history needed at inference for the last-week feature
BACKGROUND_PARAMETERS = ["no2", "o3"]


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def hourly_no2(no2: pd.DataFrame, start) -> pd.Series:
    """NO2 as a complete hourly series (missing hours become NaN)."""
    s = no2.set_index("observed_at")["value_ugm3"].sort_index()
    s = s[s.index >= _utc(start)]
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="h")
    return s.reindex(full_index)


def _background_frame(
    background: pd.DataFrame | None, index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Background pollutants as one column per parameter on the given hourly index."""
    if background is None or background.empty:
        return pd.DataFrame(np.nan, index=index, columns=BACKGROUND_PARAMETERS)
    wide = background.pivot_table(
        index="observed_at", columns="parameter", values="value_ugm3", aggfunc="mean"
    )
    return wide.reindex(index=index, columns=BACKGROUND_PARAMETERS)


def _features_for(s: pd.Series, w: pd.DataFrame, bg: pd.DataFrame) -> pd.DataFrame:
    """Core feature logic shared by training and inference. s, w and bg share one hourly index."""
    # What we actually know at issue time t: values up to t - LATENCY_HOURS
    available = s.shift(LATENCY_HOURS)
    bg_available = bg.shift(LATENCY_HOURS)
    issue_features = pd.DataFrame(
        {
            "no2_last": available,
            "no2_mean_6h": available.rolling(6, min_periods=3).mean(),
            "no2_mean_24h": available.rolling(24, min_periods=12).mean(),
            "bg_no2_last": bg_available["no2"],
            "bg_no2_mean_24h": bg_available["no2"].rolling(24, min_periods=12).mean(),
            "bg_o3_last": bg_available["o3"],
            "bg_o3_mean_24h": bg_available["o3"].rolling(24, min_periods=12).mean(),
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

    return pd.concat(frames).rename_axis("issue_time").reset_index()


def build_features(
    no2: pd.DataFrame,
    weather: pd.DataFrame,
    start,
    weather_source: str = "archive",
    background: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Training table: one row per (issue time, horizon) with a known target."""
    s = hourly_no2(no2, start)
    w = (
        weather[weather["source"] == weather_source]
        .set_index("observed_at")
        .sort_index()
    )
    bg = _background_frame(background, s.index)
    f = _features_for(s, w.reindex(s.index), bg)
    return f.dropna(subset=["target"]).reset_index(drop=True)


def build_inference_features(
    no2: pd.DataFrame,
    weather: pd.DataFrame,
    issue_time,
    background: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Rows for one issue time, horizons 1 to 24. `weather` must hold a single source."""
    issue_time = _utc(issue_time)
    index = pd.date_range(
        issue_time - pd.Timedelta(hours=HISTORY_HOURS),
        issue_time + pd.Timedelta(hours=max(HORIZONS)),
        freq="h",
    )
    s = no2.set_index("observed_at")["value_ugm3"].sort_index().reindex(index)
    w = weather.set_index("observed_at").sort_index().reindex(index)
    bg = _background_frame(background, index)
    f = _features_for(s, w, bg)
    return f[f["issue_time"] == issue_time].reset_index(drop=True)
