import json
from datetime import UTC
from pathlib import Path

from mlops_air_quality_forecast.config import settings
from mlops_air_quality_forecast.db import connect
from mlops_air_quality_forecast.monitoring import champion_test_mae

SITE_DIR = Path("site")
EARLIER_HORIZON = 6  # show what was predicted this many hours in advance
ACCURACY_DAYS = 7

ACCURACY_SQL = """
select p.horizon,
       count(*) as n,
       avg(abs(p.predicted_ugm3 - m.value_ugm3)) as model_mae,
       avg(abs(b.value_ugm3 - m.value_ugm3)) as persistence_mae
from predictions p
join no2_hourly m
  on m.observed_at = p.target_time and m.sensor_id = %(sensor)s
join no2_hourly b  -- the latest value known at issue time, for the persistence comparison
  on b.observed_at = p.issue_time - interval '3 hours' and b.sensor_id = %(sensor)s
where p.location_key = %(key)s
  and p.issue_time >= now() - make_interval(days => %(days)s)
  and m.value_ugm3 is not null
  and b.value_ugm3 is not null
group by p.horizon
order by p.horizon
"""


def iso(ts) -> str:
    return ts.astimezone(UTC).isoformat()


def points(rows) -> list[dict]:
    return [{"t": iso(t), "v": round(float(v), 1)} for t, v in rows]


key, sensor = settings.location_key, settings.no2_sensor_id

with connect() as conn:
    latest = conn.execute(
        "select issue_time, model_version from predictions "
        "where location_key = %s order by issue_time desc limit 1",
        (key,),
    ).fetchone()
    if latest is None:
        raise RuntimeError("No predictions yet, nothing to publish.")
    issue_time, model_version = latest

    forecast = conn.execute(
        "select target_time, predicted_ugm3 from predictions "
        "where location_key = %s and issue_time = %s order by horizon",
        (key, issue_time),
    ).fetchall()
    measured = conn.execute(
        "select observed_at, value_ugm3 from no2_hourly "
        "where sensor_id = %s and observed_at >= %s - interval '48 hours' "
        "and value_ugm3 is not null order by observed_at",
        (sensor, issue_time),
    ).fetchall()
    earlier = conn.execute(
        "select target_time, predicted_ugm3 from predictions "
        "where location_key = %s and horizon = %s "
        "and target_time >= %s - interval '48 hours' and target_time <= %s "
        "order by target_time",
        (key, EARLIER_HORIZON, issue_time, issue_time),
    ).fetchall()
    accuracy_rows = conn.execute(
        ACCURACY_SQL, {"sensor": sensor, "key": key, "days": ACCURACY_DAYS}
    ).fetchall()

verified = sum(r[1] for r in accuracy_rows)
accuracy = {
    "days": ACCURACY_DAYS,
    "verified": verified,
    "model_mae": round(sum(r[1] * r[2] for r in accuracy_rows) / verified, 2)
    if verified
    else None,
    "persistence_mae": round(sum(r[1] * r[3] for r in accuracy_rows) / verified, 2)
    if verified
    else None,
    "by_horizon": [
        {"h": h, "n": n, "model": round(float(m), 2), "persistence": round(float(p), 2)}
        for h, n, m, p in accuracy_rows
    ],
}

_, test_mae = champion_test_mae()

data = {
    "generated_at": iso(__import__("datetime").datetime.now(UTC)),
    "issue_time": iso(issue_time),
    "model_version": str(model_version),
    "champion_test_mae": round(float(test_mae), 2),
    "earlier_horizon": EARLIER_HORIZON,
    "forecast": points(forecast),
    "measured": points(measured),
    "earlier": points(earlier),
    "accuracy": accuracy,
}

SITE_DIR.mkdir(exist_ok=True)
(SITE_DIR / "data.json").write_text(json.dumps(data, indent=1))
print(
    f"Wrote {SITE_DIR / 'data.json'}: forecast from {issue_time:%Y-%m-%d %H:%M} UTC, "
    f"{len(measured)} measured hours, {verified} verified forecasts"
)
