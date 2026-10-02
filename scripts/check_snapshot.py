from mlops_air_quality_forecast.data import latest_snapshot_tag, load_snapshot

tag = latest_snapshot_tag()
no2, weather = load_snapshot(tag)
print("Snapshot:", tag)
print("NO2:", no2.shape, no2["observed_at"].min(), "to", no2["observed_at"].max())
print(
    "Weather:",
    weather.shape,
    weather["observed_at"].min(),
    "to",
    weather["observed_at"].max(),
)
print(no2.dtypes)
