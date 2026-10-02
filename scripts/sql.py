import sys

from mlops_air_quality_forecast.db import connect

query = sys.argv[1]
with connect() as conn:
    cur = conn.execute(query)
    print(" | ".join(col.name for col in cur.description))
    for row in cur.fetchall():
        print(" | ".join(str(value) for value in row))
