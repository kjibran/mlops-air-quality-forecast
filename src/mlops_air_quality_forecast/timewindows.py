from datetime import datetime, timedelta


def month_windows(start: datetime, end: datetime):
    """Split a date range into calendar-month chunks."""
    current = start
    while current < end:
        next_month = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
        yield current, min(next_month, end)
        current = next_month
