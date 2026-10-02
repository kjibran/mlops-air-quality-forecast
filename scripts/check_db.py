from mlops_air_quality_forecast.db import connect

with connect() as conn:
    print(conn.execute("select version()").fetchone()[0])
    for table in ("no2_hourly", "weather_hourly"):
        count = conn.execute(f"select count(*) from {table}").fetchone()[0]
        print(f"{table}: {count} rows")
