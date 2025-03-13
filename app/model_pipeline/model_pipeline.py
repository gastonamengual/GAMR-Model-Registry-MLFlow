from dataclasses import dataclass, field
from typing import Any

from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from app.experiment_tracking import AbstractExperimentTracker, MLFlowExperimentTracker
from app.model import InputData, MLModel, PredictionResult


@dataclass
class ModelPipeline:
    model: MLModel | None = field(default_factory=SGDClassifier)
    experiment_tracker: AbstractExperimentTracker = field(
        default_factory=MLFlowExperimentTracker
    )

    def predict(self, data: InputData) -> PredictionResult:
        if not self.model:
            msg = "Model cannot be None"
            raise ValueError(msg)

        y = self.model.predict(data.X)
        return PredictionResult(prediction=int(y[0]))

    def _split_data(self, data: InputData) -> tuple[Any, Any, Any, Any]:
        X_train, X_test, y_train, y_test = train_test_split(
            data.X, data.y, test_size=0.3, random_state=42
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def train(self, data: InputData, run_id: str) -> MLModel:
        X_train, X_test, y_train, y_test = self._split_data(data)
        model = self.model.fit(X_train, y_train)  # type: ignore
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.experiment_tracker.log_metrics(run_id, {"accuracy": accuracy})
        return model
