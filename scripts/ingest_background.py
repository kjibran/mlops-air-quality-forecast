import argparse
import time
from datetime import UTC, datetime

from mlops_air_quality_forecast.config import settings
from mlops_air_quality_forecast.openaq import fetch_sensor_hours
from mlops_air_quality_forecast.store import upsert_pollutant
from mlops_air_quality_forecast.timewindows import month_windows

parser = argparse.ArgumentParser(
    description="Fetch background station pollutants into the database."
)
parser.add_argument("--start", required=True, help="UTC date, e.g. 2019-11-01")
parser.add_argument("--end", required=True, help="UTC date, e.g. 2026-10-03")
args = parser.parse_args()

start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)

for parameter, sensor_id in settings.background_sensors.items():
    total = 0
    for window_start, window_end in month_windows(start, end):
        rows = fetch_sensor_hours(sensor_id, window_start, window_end)
        for r in rows:
            r["location_id"] = settings.background_location_id
            r["parameter"] = parameter
        total += upsert_pollutant(rows)
        time.sleep(1)
    print(f"{parameter} (sensor {sensor_id}): {total} hours")
