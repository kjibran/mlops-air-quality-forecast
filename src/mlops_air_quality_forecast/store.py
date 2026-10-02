from mlops_air_quality_forecast.db import connect

UPSERT_NO2 = """
insert into no2_hourly (sensor_id, observed_at, value_ugm3, coverage_pct)
values (%(sensor_id)s, %(observed_at)s, %(value_ugm3)s, %(coverage_pct)s)
on conflict (sensor_id, observed_at) do update set
    value_ugm3 = excluded.value_ugm3,
    coverage_pct = excluded.coverage_pct,
    ingested_at = now()
"""

UPSERT_WEATHER = """
insert into weather_hourly (
    location_key, observed_at, source,
    temperature_2m, wind_speed_10m, wind_direction_10m,
    relative_humidity_2m, boundary_layer_height
)
values (
    %(location_key)s, %(observed_at)s, %(source)s,
    %(temperature_2m)s, %(wind_speed_10m)s, %(wind_direction_10m)s,
    %(relative_humidity_2m)s, %(boundary_layer_height)s
)
on conflict (location_key, observed_at, source) do update set
    temperature_2m = excluded.temperature_2m,
    wind_speed_10m = excluded.wind_speed_10m,
    wind_direction_10m = excluded.wind_direction_10m,
    relative_humidity_2m = excluded.relative_humidity_2m,
    boundary_layer_height = excluded.boundary_layer_height,
    ingested_at = now()
"""


def upsert_weather(rows: list[dict]) -> int:
    """Insert new weather hours and update existing ones. Safe to run repeatedly."""
    if not rows:
        return 0
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(UPSERT_WEATHER, rows)
    return len(rows)


def upsert_no2(rows: list[dict]) -> int:
    """Insert new hours and update existing ones. Safe to run repeatedly."""
    if not rows:
        return 0
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(UPSERT_NO2, rows)
    return len(rows)
