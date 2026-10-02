import argparse
import time
from datetime import date

from mlops_air_quality_forecast.config import settings
from mlops_air_quality_forecast.openmeteo import (
    FORECAST_VARS,
    fetch_weather_forecast_day1,
)
from mlops_air_quality_forecast.store import upsert_weather

parser = argparse.ArgumentParser(
    description="Fetch Day 1 forecast weather into the database."
)
parser.add_argument("--start", required=True, help="Date, e.g. 2024-01-01")
parser.add_argument("--end", required=True, help="Date, e.g. 2026-10-01 (inclusive)")
args = parser.parse_args()

start = date.fromisoformat(args.start)
end = date.fromisoformat(args.end)

total = 0
for year in range(start.year, end.year + 1):
    chunk_start = max(start, date(year, 1, 1))
    chunk_end = min(end, date(year, 12, 31))
    rows = fetch_weather_forecast_day1(
        settings.station_lat,
        settings.station_lon,
        chunk_start,
        chunk_end,
        settings.location_key,
    )
    rows = [r for r in rows if any(r[v] is not None for v in FORECAST_VARS)]
    total += upsert_weather(rows)
    print(f"{year}: {len(rows)} hours")
    time.sleep(1)

print(f"Done. Upserted {total} rows in total.")
