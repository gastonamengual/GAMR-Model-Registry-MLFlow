from pydantic import BaseModel


class ModelRegistryConfig(BaseModel):
    model_name: str
    model_version: str | None = None
