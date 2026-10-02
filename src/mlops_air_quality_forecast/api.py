from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse

from mlops_air_quality_forecast.config import settings
from mlops_air_quality_forecast.db import connect

app = FastAPI(
    title="Copenhagen NO2 Forecast API",
    description="Hourly NO2 forecasts for the next 24 hours, with live verification against measurements.",
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecast/latest")
def latest_forecast():
    """The most recent 24-hour forecast."""
    key = settings.location_key
    with connect() as conn:
        rows = conn.execute(
            """
            select issue_time, horizon, target_time, predicted_ugm3, model_version
            from predictions
            where location_key = %s
              and issue_time = (select max(issue_time) from predictions where location_key = %s)
            order by horizon
            """,
            (key, key),
        ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="No forecast available yet")
    return {
        "location": key,
        "issue_time": rows[0][0],
        "model_version": rows[0][4],
        "unit": "µg/m³",
        "forecast": [
            {"horizon": r[1], "target_time": r[2], "no2": round(r[3], 1)} for r in rows
        ],
    }


@app.get("/forecast/verification")
def verification(days: int = Query(7, ge=1, le=90)):
    """How forecasts from the last `days` days compared with measured NO2, per horizon."""
    with connect() as conn:
        rows = conn.execute(
            """
            select p.horizon,
                   count(*) as n,
                   avg(abs(p.predicted_ugm3 - m.value_ugm3)) as mae
            from predictions p
            join no2_hourly m
              on m.observed_at = p.target_time and m.sensor_id = %s
            where p.location_key = %s
              and p.issue_time >= now() - make_interval(days => %s)
              and m.value_ugm3 is not null
            group by p.horizon
            order by p.horizon
            """,
            (settings.no2_sensor_id, settings.location_key, days),
        ).fetchall()
    total = sum(r[1] for r in rows)
    overall = sum(r[1] * r[2] for r in rows) / total if total else None
    return {
        "days": days,
        "unit": "µg/m³",
        "forecasts_verified": total,
        "overall_mae": round(overall, 2) if overall is not None else None,
        "by_horizon": [
            {"horizon": r[0], "n": r[1], "mae": round(r[2], 2)} for r in rows
        ],
    }
