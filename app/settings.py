from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings_(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        secrets_dir="secrets",
        extra="ignore",
    )

    MODEL_TRACKING_URI: str = Field(default="")
    RENDER_API_TOKEN: str = Field(default="")


Settings = Settings_()
