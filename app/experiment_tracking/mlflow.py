from dataclasses import dataclass
from uuid import uuid4

import mlflow
import mlflow.entities

from app.clients import MLFlowClient

from .abstract import AbstractExperimentTracker


@dataclass
class MLFlowExperimentTracker(AbstractExperimentTracker):
    @property
    def client(self) -> mlflow.MlflowClient:
        return MLFlowClient().client

    def set_experiment(self, experiment_name: str | None = None) -> None:
        experiment_name = experiment_name or f"experiment-{uuid4()}"
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment = self.client.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.client.log_metric(run_id, key, value)

    def start_run(self) -> mlflow.ActiveRun:
        return mlflow.start_run()
