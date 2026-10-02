from datetime import UTC, datetime

from mlops_air_quality_forecast.timewindows import month_windows


def utc(*args):
    return datetime(*args, tzinfo=UTC)


def test_splits_at_month_boundaries():
    windows = list(month_windows(utc(2024, 1, 15), utc(2024, 3, 10)))
    assert windows == [
        (utc(2024, 1, 15), utc(2024, 2, 1)),
        (utc(2024, 2, 1), utc(2024, 3, 1)),
        (utc(2024, 3, 1), utc(2024, 3, 10)),
    ]


def test_handles_year_change():
    windows = list(month_windows(utc(2019, 12, 1), utc(2020, 1, 15)))
    assert windows[-1] == (utc(2020, 1, 1), utc(2020, 1, 15))


def test_empty_when_start_equals_end():
    assert list(month_windows(utc(2024, 5, 1), utc(2024, 5, 1))) == []
