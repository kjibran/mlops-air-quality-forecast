import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from mlops_air_quality_forecast.config import settings


def latest_snapshot_tag() -> str:
    """Return the most recent snapshot tag on the Hugging Face dataset."""
    api = HfApi(token=settings.hf_token or None)
    refs = api.list_repo_refs(settings.hf_dataset_repo, repo_type="dataset")
    tags = sorted(t.name for t in refs.tags if t.name.startswith("snapshot-"))
    if not tags:
        raise RuntimeError("No snapshot tags found on the dataset.")
    return tags[-1]


def _load_table(name: str, tag: str) -> pd.DataFrame:
    """One table exactly as it was at a given snapshot tag."""
    path = hf_hub_download(
        repo_id=settings.hf_dataset_repo,
        repo_type="dataset",
        filename=f"data/{name}.parquet",
        revision=tag,
        token=settings.hf_token or None,
    )
    return pd.read_parquet(path)


def load_snapshot(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the NO2 and weather tables exactly as they were at one snapshot."""
    return _load_table("no2_hourly", tag), _load_table("weather_hourly", tag)


def load_background(tag: str) -> pd.DataFrame:
    """Load the background station pollutants. Only in snapshots from October 2026 onwards."""
    return _load_table("pollutant_hourly", tag)
