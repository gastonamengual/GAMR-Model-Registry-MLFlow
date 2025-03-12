from typing import Any, Protocol

from mlflow.models.model import ModelInfo

from app.model import MLModel
from app.model_pipeline import ModelPipeline

from .config import ModelRegistryConfig


class ModelRegistry(Protocol):
    def load(self, config: ModelRegistryConfig) -> ModelPipeline: ...

    def create_model(
        self, model: MLModel, config: ModelRegistryConfig
    ) -> ModelRegistryConfig: ...

    def get_latest_model(self, config: ModelRegistryConfig) -> ModelPipeline: ...

    def log_sklearn_model(
        self, sk_model: MLModel, artifact_path: str, registered_model_name: str
    ) -> ModelInfo: ...

    def load_model(self, model_uri: str) -> Any: ...

    def get_latest_version(self, model_name: str) -> str: ...

    def get_all_models(self) -> list[str]: ...

    def get_all_model_versions(self, model_name: str) -> list[int]: ...

    def get_model_version(self, model_name: str, version: str) -> Any: ...
