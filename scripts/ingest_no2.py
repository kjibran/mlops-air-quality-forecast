import argparse
import time
from datetime import UTC, datetime

from mlops_air_quality_forecast.config import settings
from mlops_air_quality_forecast.openaq import fetch_no2_hours
from mlops_air_quality_forecast.store import upsert_no2
from mlops_air_quality_forecast.timewindows import month_windows

parser = argparse.ArgumentParser(
    description="Fetch hourly NO2 from OpenAQ into the database."
)
parser.add_argument("--start", required=True, help="UTC date, e.g. 2019-01-01")
parser.add_argument("--end", required=True, help="UTC date, e.g. 2026-10-03")
args = parser.parse_args()

start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)

total = 0
for window_start, window_end in month_windows(start, end):
    rows = fetch_no2_hours(settings.no2_sensor_id, window_start, window_end)
    total += upsert_no2(rows)
    print(f"{window_start:%Y-%m}: {len(rows)} hours")
    time.sleep(1)

print(f"Done. Upserted {total} rows in total.")
