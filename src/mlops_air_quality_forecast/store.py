from mlops_air_quality_forecast.db import connect

UPSERT_NO2 = """
insert into no2_hourly (sensor_id, observed_at, value_ugm3, coverage_pct)
values (%(sensor_id)s, %(observed_at)s, %(value_ugm3)s, %(coverage_pct)s)
on conflict (sensor_id, observed_at) do update set
    value_ugm3 = excluded.value_ugm3,
    coverage_pct = excluded.coverage_pct,
    ingested_at = now()
"""


def upsert_no2(rows: list[dict]) -> int:
    """Insert new hours and update existing ones. Safe to run repeatedly."""
    if not rows:
        return 0
    with connect() as conn, conn.cursor() as cur:
        cur.executemany(UPSERT_NO2, rows)
    return len(rows)
