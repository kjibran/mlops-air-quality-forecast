import psycopg

from mlops_air_quality_forecast.config import settings


def connect() -> psycopg.Connection:
    """Open a connection to the Supabase Postgres database."""
    return psycopg.connect(settings.database_url)
