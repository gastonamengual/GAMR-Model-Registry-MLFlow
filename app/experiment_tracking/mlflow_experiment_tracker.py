from dataclasses import dataclass
import mlflow
import mlflow.entities
from mlflow.entities import model_registry
from mlflow.pyfunc import PyFuncModel

from .abstract_experiment_tracker import AbstractExperimentTracker
from mlflow.models.model import ModelInfo


@dataclass
class MLFlowExperimentTracker(AbstractExperimentTracker):
    @property
    def _client(self) -> mlflow.MlflowClient:
        return mlflow.MlflowClient()

    def model_uri(self, model_name: str, model_version: str) -> str:
        return f"models:/{model_name}/{model_version}"
        # return f"runs:/{run_id}/GAMR_Model"

    def __post_init__(self) -> None:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

    def set_experiment(self, experiment_name) -> mlflow.entities.Experiment:
        experiment = self._client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment = self._client.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        return experiment

    def log_params(self, run_id, params) -> None:
        for key, value in params.items():
            self._client.log_param(run_id, key, value)

    def log_metrics(self, run_id, metrics) -> None:
        for key, value in metrics.items():
            self._client.log_metric(run_id, key, value)

    def set_tag(self, run_id, tag_name, tag_value) -> None:
        self._client.set_tag(run_id, tag_name, tag_value)

    def log_model(
        self, sk_model, artifact_path, input_example, registered_model_name
    ) -> ModelInfo:
        model_info: ModelInfo = mlflow.sklearn.log_model(
            sk_model=sk_model,
            artifact_path=artifact_path,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )
        return model_info

    def start_run(self) -> mlflow.ActiveRun:
        return mlflow.start_run()

    def get_run_info(self) -> mlflow.entities.RunInfo:
        active_run = mlflow.active_run()
        if active_run is None:
            return None
        return active_run.info

    def log_artifacts(self, artifact_path, artifact_file) -> None:
        self._client.log_artifact(artifact_file, artifact_path)

    def get_latest_version(self, model_name) -> str:
        versions = self._client.get_latest_versions(model_name)
        latest_version = versions[-1].version
        return latest_version

    def get_model_version(self, model_name, version) -> model_registry.ModelVersion:
        model_version = self._client.get_model_version(model_name, version=int(version))
        return model_version

    def load_model(self, model_uri) -> PyFuncModel:
        return mlflow.pyfunc.load_model(model_uri)
