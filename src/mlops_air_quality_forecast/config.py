from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = ""
    openaq_api_key: str = ""
    no2_sensor_id: int = 13371  # NO2 at Copenhagen/1257 (OpenAQ location 5177)


settings = Settings()
