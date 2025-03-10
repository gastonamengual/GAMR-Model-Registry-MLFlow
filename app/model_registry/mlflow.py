from dataclasses import dataclass, field

import mlflow
import mlflow.entities
from mlflow.entities import model_registry
from mlflow.exceptions import MlflowException
from mlflow.models.model import ModelInfo
from mlflow.sklearn import load_model as sklearn_load_model
from mlflow.sklearn import log_model as sklearn_log_model

from app.clients import MLFlowClient
from app.experiment_tracking import AbstractExperimentTracker, MLFlowExperimentTracker
from app.model import MLModel
from app.model_pipeline import ModelPipeline

from .config import ModelRegistryConfig


@dataclass
class MLFlowModelRegistry:
    experiment_tracker: AbstractExperimentTracker = field(
        default_factory=MLFlowExperimentTracker
    )

    def model_uri(self, model_name: str, model_version: str) -> str:
        return f"models:/{model_name}/{model_version}"

    @property
    def client(self) -> mlflow.MlflowClient:
        return MLFlowClient().client

    def load(self, config: ModelRegistryConfig) -> ModelPipeline:
        model_version = config.model_version or self.get_latest_version(
            config.model_name
        )
        if not model_version:
            raise ValueError

        model_uri = self.model_uri(
            model_name=config.model_name, model_version=model_version
        )
        model = self.load_model(model_uri)
        return ModelPipeline(model)

    def create_model(
        self, model: MLModel, config: ModelRegistryConfig
    ) -> ModelRegistryConfig:
        model_info = self.log_sklearn_model(
            sk_model=model,
            artifact_path=f"{config.model_name}",
            registered_model_name=config.model_name,
        )
        return config.model_copy(
            update={"model_version": model_info.__dict__["_registered_model_version"]}
        )

    def get_latest_model(self, config: ModelRegistryConfig) -> ModelPipeline:
        try:
            version = self.get_latest_version(model_name=config.model_name)
            config = ModelRegistryConfig(
                model_name=config.model_name, model_version=version
            )
            return self.load(config)
        except MlflowException:
            return ModelPipeline()

    def log_sklearn_model(
        self, sk_model: MLModel, artifact_path: str, registered_model_name: str
    ) -> ModelInfo:
        model_info: ModelInfo = sklearn_log_model(
            sk_model=sk_model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name,
        )
        return model_info

    def load_model(self, model_uri: str) -> MLModel | None:
        return sklearn_load_model(model_uri)  # type: ignore

    def get_latest_version(self, model_name: str) -> str:
        versions = self.client.get_latest_versions(model_name)
        return versions[-1].version  # type: ignore

    def get_all_models(self) -> list[str]:
        models = self.client.search_registered_models()
        return [model.name for model in models]

    def get_all_model_versions(self, model_name: str) -> list[int]:
        model_versions = self.client.search_model_versions(f"name='{model_name}'")
        return [model_version.version for model_version in model_versions]

    def get_model_version(
        self, model_name: str, version: str
    ) -> model_registry.ModelVersion:
        return self.client.get_model_version(model_name, version=version)
