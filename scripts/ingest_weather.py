import argparse
import time
from datetime import date

from mlops_air_quality_forecast.config import settings
from mlops_air_quality_forecast.openmeteo import fetch_weather_archive
from mlops_air_quality_forecast.store import upsert_weather

parser = argparse.ArgumentParser(
    description="Fetch hourly weather from Open-Meteo into the database."
)
parser.add_argument("--start", required=True, help="Date, e.g. 2019-11-01")
parser.add_argument("--end", required=True, help="Date, e.g. 2026-10-02 (inclusive)")
args = parser.parse_args()

start = date.fromisoformat(args.start)
end = date.fromisoformat(args.end)

total = 0
for year in range(start.year, end.year + 1):
    chunk_start = max(start, date(year, 1, 1))
    chunk_end = min(end, date(year, 12, 31))
    rows = fetch_weather_archive(
        settings.station_lat,
        settings.station_lon,
        chunk_start,
        chunk_end,
        settings.location_key,
    )
    total += upsert_weather(rows)
    print(f"{year}: {len(rows)} hours")
    time.sleep(1)

print(f"Done. Upserted {total} rows in total.")
