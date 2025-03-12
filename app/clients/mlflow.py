from dataclasses import dataclass

import mlflow
import mlflow.entities

from app.settings import Settings


@dataclass
class MLFlowClient:
    @property
    def client(self) -> mlflow.MlflowClient:
        return mlflow.MlflowClient()

    def __post_init__(self) -> None:
        mlflow.set_tracking_uri(Settings.MODEL_TRACKING_URI)
        mlflow.autolog()
