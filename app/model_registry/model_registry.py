from dataclasses import dataclass

import mlflow


@dataclass
class ModelRegistry:
    
    @property
    def _client(self):
        return mlflow.tracking.MlflowClient()
    
    def __post_init__(self):
        mlflow.set_tracking_uri("http://127.0.0.1:8080")