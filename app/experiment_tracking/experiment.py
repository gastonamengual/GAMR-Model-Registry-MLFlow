from dataclasses import dataclass
from app.experiment_tracking.mlflow_experiment_tracker import MLFlowExperimentTracker


@dataclass
class Experiment:
    experiment_name: str = "MLflow Afternoon"
    model_name: str = "tracking-quickstart"
    tracker: MLFlowExperimentTracker = field(default_factory=MLFlowExperimentTracker)

    def track(self, model, X_train, y_train, X_test, y_test):
        """
        Log experiment parameters, metrics, and model to MLflow.
        """
        self.tracker.set_experiment(self.experiment_name)

        with self.tracker.start_run():
            run_info = self.tracker.get_run_info()
            
            # Log parameters
            self.tracker.log_params(run_info.run_id, model.get_params())
            
            # Predictions and metrics
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            self.tracker.log_metrics(run_info.run_id, {"accuracy": accuracy})
            
            # Add tag
            self.tracker.set_tag(run_info.run_id, "Training Info", "Logistic regression model for Iris data")
            
            # Infer model signature
            signature = infer_signature(X_train, model.predict(X_train))

            # Log the model
            self.tracker.log_model(
                sk_model=model,
                artifact_path="iris_model",
                signature=signature,
                input_example=X_train,
                registered_model_name=self.model_name,
            )
