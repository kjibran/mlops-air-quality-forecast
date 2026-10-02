from datetime import date

import httpx

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "relative_humidity_2m",
    "boundary_layer_height",
]


def fetch_weather_archive(
    lat: float, lon: float, start: date, end: date, location_key: str
) -> list[dict]:
    """Fetch hourly historical weather for one location. Both dates are inclusive."""
    response = httpx.get(
        ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(HOURLY_VARS),
            "wind_speed_unit": "ms",
            "timezone": "GMT",
        },
        timeout=120,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]

    rows = []
    for i, timestamp in enumerate(hourly["time"]):
        row = {
            "location_key": location_key,
            "observed_at": f"{timestamp}:00+00:00",  # e.g. 2025-01-15T00:00 -> UTC timestamp
            "source": "archive",
        }
        for var in HOURLY_VARS:
            row[var] = hourly[var][i]
        rows.append(row)
    return rows


PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_VARS = [v for v in HOURLY_VARS if v != "boundary_layer_height"]


def fetch_weather_forecast_day1(
    lat: float, lon: float, start: date, end: date, location_key: str
) -> list[dict]:
    """Hourly weather as it was forecast about one day earlier. Both dates inclusive."""
    response = httpx.get(
        PREVIOUS_RUNS_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(f"{v}_previous_day1" for v in FORECAST_VARS),
            "wind_speed_unit": "ms",
            "timezone": "GMT",
        },
        timeout=120,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]

    rows = []
    for i, timestamp in enumerate(hourly["time"]):
        row = {
            "location_key": location_key,
            "observed_at": f"{timestamp}:00+00:00",
            "source": "forecast_day1",
            "boundary_layer_height": None,  # not available from previous runs
        }
        for var in FORECAST_VARS:
            row[var] = hourly[f"{var}_previous_day1"][i]
        rows.append(row)
    return rows


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_weather_forecast_live(
    lat: float, lon: float, location_key: str
) -> list[dict]:
    """The latest hourly weather forecast, from yesterday to two days ahead."""
    response = httpx.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": ",".join(FORECAST_VARS),
            "wind_speed_unit": "ms",
            "timezone": "GMT",
            "past_days": 1,
            "forecast_days": 3,
        },
        timeout=60,
    )
    response.raise_for_status()
    hourly = response.json()["hourly"]

    rows = []
    for i, timestamp in enumerate(hourly["time"]):
        row = {
            "location_key": location_key,
            "observed_at": f"{timestamp}:00+00:00",
            "source": "forecast_live",
            "boundary_layer_height": None,
        }
        for var in FORECAST_VARS:
            row[var] = hourly[var][i]
        rows.append(row)
    return rows
