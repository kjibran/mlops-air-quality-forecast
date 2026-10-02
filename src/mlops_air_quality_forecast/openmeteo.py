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
