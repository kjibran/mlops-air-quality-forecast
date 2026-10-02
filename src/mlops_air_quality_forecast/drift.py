import numpy as np
import pandas as pd

DRIFT_FEATURES = [
    "no2_last",
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    "temperature",
    "humidity",
]
CURRENT_DAYS = 7
SEASON_WINDOW_DAYS = 14


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two samples, using reference quantile bins."""
    reference, current = reference.dropna(), current.dropna()
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    ref_share = np.clip(ref_share, 1e-4, None)  # avoid log(0) for empty bins
    cur_share = np.clip(cur_share, 1e-4, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def _day_of_year_distance(times: pd.Series, center: int) -> pd.Series:
    """Days between each timestamp's day of year and `center`, wrapping around New Year."""
    d = (times.dt.dayofyear - center).abs()
    return np.minimum(d, 365 - d)


def _window_psi(rows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """PSI per feature for one window vs. the same season in other years."""
    in_window = (rows["target_time"] > start) & (rows["target_time"] <= end)
    center = (start + (end - start) / 2).dayofyear
    in_season = _day_of_year_distance(rows["target_time"], center) <= SEASON_WINDOW_DAYS
    far_away = (rows["target_time"] - end).abs() > pd.Timedelta(
        days=60
    )  # other years only
    current, reference = rows[in_window], rows[in_season & far_away]

    result = {}
    for f in DRIFT_FEATURES:
        cur, ref = current[f].dropna(), reference[f].dropna()
        result[f] = psi(ref, cur) if len(cur) >= 100 and len(ref) >= 200 else np.nan
    return result


def drift_report(features: pd.DataFrame) -> pd.DataFrame:
    """Current week's PSI per feature, judged against the PSI of past weeks.

    Weather comes in multi-day regimes, so a single week rarely matches a whole
    season. Instead of fixed textbook thresholds, a week is flagged as unusual only
    if its PSI exceeds the 95th percentile of PSI values seen in past weeks.
    """
    rows = features[
        features["horizon"] == 1
    ]  # one row per hour, no duplicates across horizons
    end = rows["target_time"].max()
    week = pd.Timedelta(days=CURRENT_DAYS)

    current = _window_psi(rows, end - week, end)

    history = []
    window_end = end - week
    while window_end - week > rows["target_time"].min():
        history.append(_window_psi(rows, window_end - week, window_end))
        window_end -= week
    history = pd.DataFrame(history)

    report = pd.DataFrame(
        {
            "feature": DRIFT_FEATURES,
            "psi": [current[f] for f in DRIFT_FEATURES],
            "typical_95pct": [np.nanpercentile(history[f], 95) for f in DRIFT_FEATURES],
            "past_weeks": [int(history[f].notna().sum()) for f in DRIFT_FEATURES],
        }
    )
    report["status"] = np.where(
        report["psi"] > report["typical_95pct"], "unusual", "normal"
    )
    return report
