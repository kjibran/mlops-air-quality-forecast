import time
from datetime import datetime

import httpx

from mlops_air_quality_forecast.config import settings

BASE_URL = "https://api.openaq.org/v3"
PAGE_SIZE = 1000


def _ts(dt: datetime) -> str:
    """OpenAQ's hourly endpoint needs full UTC timestamps, not bare dates."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


RETRY_STATUSES = {408, 429, 500, 502, 503, 504}


def _get_with_retry(client: httpx.Client, url: str, params: dict, attempts: int = 4):
    """GET with retries for timeouts, rate limits, and server errors."""
    for attempt in range(attempts):
        try:
            response = client.get(url, params=params)
            if response.status_code not in RETRY_STATUSES:
                response.raise_for_status()
                return response
            reason = f"HTTP {response.status_code}"
        except httpx.TimeoutException:
            reason = "client timeout"
        wait = 5 * 2**attempt  # 5, 10, 20, 40 seconds
        print(f"  {reason}, retrying in {wait}s")
        time.sleep(wait)
    raise RuntimeError(f"OpenAQ request failed after {attempts} attempts: {url}")


def fetch_no2_hours(sensor_id: int, start: datetime, end: datetime) -> list[dict]:
    """Fetch hourly NO2 values for one sensor between start and end (UTC)."""
    rows = []
    page = 1
    headers = {"X-API-Key": settings.openaq_api_key}
    with httpx.Client(timeout=60, headers=headers) as client:
        while True:
            response = _get_with_retry(
                client,
                f"{BASE_URL}/sensors/{sensor_id}/hours",
                params={
                    "datetime_from": _ts(start),
                    "datetime_to": _ts(end),
                    "limit": PAGE_SIZE,
                    "page": page,
                },
            )
            results = response.json().get("results", [])
            for r in results:
                rows.append(
                    {
                        "sensor_id": sensor_id,
                        "observed_at": r["period"]["datetimeFrom"]["utc"],
                        "value_ugm3": r["value"],  # may be None, stored as NULL
                        "coverage_pct": (r.get("coverage") or {}).get(
                            "percentComplete"
                        ),
                    }
                )
            if len(results) < PAGE_SIZE:
                break
            page += 1
            time.sleep(1)  # stay well within the API rate limit
    return rows
