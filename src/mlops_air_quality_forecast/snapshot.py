import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from mlops_air_quality_forecast.config import settings
from mlops_air_quality_forecast.db import connect

TABLES = {
    "no2_hourly": "select * from no2_hourly order by sensor_id, observed_at",
    "weather_hourly": "select * from weather_hourly order by location_key, source, observed_at",
    "pollutant_hourly": "select * from pollutant_hourly order by sensor_id, observed_at",
}


def export_tables(out_dir: Path) -> dict[str, int]:
    """Write each table to a Parquet file. Returns row counts per table."""
    counts = {}
    with connect() as conn:
        for name, query in TABLES.items():
            cur = conn.execute(query)
            columns = [col.name for col in cur.description]
            df = pd.DataFrame(cur.fetchall(), columns=columns)
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
            counts[name] = len(df)
    return counts


def publish_snapshot() -> str:
    """Export the database to Parquet, upload to Hugging Face, and tag the commit."""
    if not settings.hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add it to .env locally or as a GitHub secret."
        )

    tag = f"snapshot-{datetime.now(UTC):%Y%m%d-%H%M}"
    api = HfApi(token=settings.hf_token)

    with tempfile.TemporaryDirectory() as tmp:
        counts = export_tables(Path(tmp))
        summary = ", ".join(f"{name}: {n} rows" for name, n in counts.items())
        api.upload_folder(
            repo_id=settings.hf_dataset_repo,
            repo_type="dataset",
            folder_path=tmp,
            path_in_repo="data",
            commit_message=f"{tag} ({summary})",
        )

    api.create_tag(settings.hf_dataset_repo, tag=tag, repo_type="dataset")
    print(f"Published {tag}: {summary}")
    return tag
