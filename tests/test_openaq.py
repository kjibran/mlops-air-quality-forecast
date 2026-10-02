from datetime import UTC, datetime

import httpx

from mlops_air_quality_forecast import openaq


def fake_result(hour, value):
    return {
        "value": value,
        "period": {"datetimeFrom": {"utc": f"2026-01-01T{hour:02d}:00:00Z"}},
        "coverage": {"percentComplete": 100.0},
    }


def test_fetches_all_pages_and_keeps_missing_values(monkeypatch):
    # Page 1 is full (forces a second request), page 2 is short (ends the loop)
    pages = [
        [fake_result(0, 12.5)] * openaq.PAGE_SIZE,
        [fake_result(1, None), fake_result(2, 8.0)],
    ]
    calls = []

    def fake_get(client, url, params, attempts=4):
        calls.append(params["page"])
        return httpx.Response(200, json={"results": pages[params["page"] - 1]})

    monkeypatch.setattr(openaq, "_get_with_retry", fake_get)
    monkeypatch.setattr(openaq.time, "sleep", lambda seconds: None)

    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)
    rows = openaq.fetch_sensor_hours(13371, start, end)

    assert calls == [1, 2]
    assert len(rows) == openaq.PAGE_SIZE + 2
    assert rows[-2]["value_ugm3"] is None
    assert rows[-1]["value_ugm3"] == 8.0
