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


def load_snapshot(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the NO2 and weather tables exactly as they were at one snapshot."""
    frames = []
    for name in ("no2_hourly", "weather_hourly"):
        path = hf_hub_download(
            repo_id=settings.hf_dataset_repo,
            repo_type="dataset",
            filename=f"data/{name}.parquet",
            revision=tag,
            token=settings.hf_token or None,
        )
        frames.append(pd.read_parquet(path))
    return frames[0], frames[1]
