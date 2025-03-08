from dataclasses import asdict, dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .pipeline import Pipeline


@dataclass
class TrainingPipeline(Pipeline):
    def read_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.payload.data.X, self.payload.data.y, test_size=0.3, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def execute(self) -> int:
        print("Reading data")
        X_train, X_test, y_train, y_test = self.read_data()

        print("Training model")
        trained_model = self.model_trainer.train(X_train, y_train)

        y_pred = trained_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        run_info = self.experiment_tracker.get_run_info()
        self.experiment_tracker.log_params(run_info.run_id, asdict(self.model_trainer))
        self.experiment_tracker.log_metrics(run_info.run_id, {"accuracy": accuracy})

        print("Logging model")
        model_info = self.experiment_tracker.log_model(
            sk_model=trained_model,
            artifact_path=f"{self.payload.model_name}",
            input_example=X_train,
            registered_model_name=self.payload.model_name,
        )
        new_model_version = model_info.__dict__["_registered_model_version"]
        return new_model_version
