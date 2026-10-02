from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = ""
    openaq_api_key: str = ""
    no2_sensor_id: int = 13371  # NO2 at Copenhagen/1257 (OpenAQ location 5177)
    training_start: str = "2019-11-01"  # data before this is unreliable, see README
    location_key: str = "station_5177"
    station_lat: float = 55.6983
    station_lon: float = 12.5533
    hf_token: str = ""
    hf_dataset_repo: str = "khajlk/copenhagen-air-quality-hourly"
    background_location_id: int = 5170
    background_sensors: dict[str, int] = {"no2": 13375, "o3": 13360}


settings = Settings()
