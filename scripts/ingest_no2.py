import argparse
from datetime import UTC, datetime

from mlops_air_quality_forecast.config import settings
from mlops_air_quality_forecast.openaq import fetch_no2_hours
from mlops_air_quality_forecast.store import upsert_no2

parser = argparse.ArgumentParser(
    description="Fetch hourly NO2 from OpenAQ into the database."
)
parser.add_argument("--start", required=True, help="UTC date, e.g. 2026-09-25")
parser.add_argument("--end", required=True, help="UTC date, e.g. 2026-10-02")
args = parser.parse_args()

start = datetime.fromisoformat(args.start).replace(tzinfo=UTC)
end = datetime.fromisoformat(args.end).replace(tzinfo=UTC)

rows = fetch_no2_hours(settings.no2_sensor_id, start, end)
print(f"Fetched {len(rows)} hours from OpenAQ")
print(f"Upserted {upsert_no2(rows)} rows into no2_hourly")
