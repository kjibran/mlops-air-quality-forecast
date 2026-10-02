from datetime import UTC, datetime, timedelta

from mlops_air_quality_forecast.config import settings
from mlops_air_quality_forecast.openaq import fetch_sensor_hours
from mlops_air_quality_forecast.openmeteo import (
    FORECAST_VARS,
    HOURLY_VARS,
    fetch_weather_archive,
    fetch_weather_forecast_day1,
)
from mlops_air_quality_forecast.store import (
    upsert_no2,
    upsert_pollutant,
    upsert_weather,
)

POLLUTANT_WINDOW = timedelta(days=3)
WEATHER_WINDOW_DAYS = 7

now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
pollutant_start = now - POLLUTANT_WINDOW

# NO2 at the target station: re-fetch the last 3 days to catch late or corrected values
no2_rows = fetch_sensor_hours(settings.no2_sensor_id, pollutant_start, now)
print(
    f"NO2: {upsert_no2(no2_rows)} hours upserted "
    f"({pollutant_start:%Y-%m-%d %H:%M} to {now:%Y-%m-%d %H:%M} UTC)"
)

# Background station pollutants, same window
for parameter, sensor_id in settings.background_sensors.items():
    rows = fetch_sensor_hours(sensor_id, pollutant_start, now)
    for r in rows:
        r["location_id"] = settings.background_location_id
        r["parameter"] = parameter
    print(f"Background {parameter}: {upsert_pollutant(rows)} hours upserted")

# Weather: re-fetch the last 7 days, never storing hours after now
today = now.date()
cutoff = f"{now:%Y-%m-%dT%H:%M}:00+00:00"  # same text format as the stored timestamps

weather_rows = fetch_weather_archive(
    settings.station_lat,
    settings.station_lon,
    today - timedelta(days=WEATHER_WINDOW_DAYS),
    today,
    settings.location_key,
)
weather_rows = [
    r
    for r in weather_rows
    if r["observed_at"] <= cutoff and any(r[v] is not None for v in HOURLY_VARS)
]
print(
    f"Weather: {upsert_weather(weather_rows)} hours upserted (last {WEATHER_WINDOW_DAYS} days)"
)

# Day 1 forecast weather: needed to evaluate models the way they run in production
forecast_rows = fetch_weather_forecast_day1(
    settings.station_lat,
    settings.station_lon,
    today - timedelta(days=WEATHER_WINDOW_DAYS),
    today,
    settings.location_key,
)
forecast_rows = [
    r
    for r in forecast_rows
    if r["observed_at"] <= cutoff and any(r[v] is not None for v in FORECAST_VARS)
]
print(f"Forecast weather (day 1): {upsert_weather(forecast_rows)} hours upserted")
