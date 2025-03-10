from dataclasses import dataclass

import mlflow
import mlflow.entities


@dataclass
class MLFlowClient:
    @property
    def client(self) -> mlflow.MlflowClient:
        return mlflow.MlflowClient()

    def __post_init__(self) -> None:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.autolog()
