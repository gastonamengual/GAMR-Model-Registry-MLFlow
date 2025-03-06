from dataclasses import dataclass
import mlflow
import mlflow.entities

@dataclass
class ExperimentTracker:

    @property
    def _client(self):
        return mlflow.MlflowClient()
    
    def model_uri(self, run_id: str):
        return f"runs:/{run_id}/GAMR_Model"
    
    def __post_init__(self):
        mlflow.set_tracking_uri("http://127.0.0.1:8080")

    def set_experiment(self, experiment_name):
        self.experiment_name = experiment_name
        experiment = self._client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment = self._client.create_experiment(experiment_name)
        mlflow.set_experiment(experiment_name)
        return experiment

    def log_params(self, run_id, params):
        for key, value in params.items():
            self._client.log_param(run_id, key, value)

    def log_metrics(self, run_id, metrics):
        for key, value in metrics.items():
            self._client.log_metric(run_id, key, value)

    def set_tag(self, run_id, tag_name, tag_value):
        self._client.set_tag(run_id, tag_name, tag_value)

    def log_model(self, sk_model, artifact_path, signature, input_example, registered_model_name):
        model_info = mlflow.sklearn.log_model(
            sk_model=sk_model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )
        return model_info

    def start_run(self):
        return mlflow.start_run()

    def get_run_info(self) -> mlflow.entities.RunInfo:
        active_run = mlflow.active_run()
        if active_run is None:
            return None
        return active_run.info

    def log_artifacts(self, artifact_path, artifact_file):
        self._client.log_artifact(artifact_file, artifact_path)

    def get_latest_version(self, model_name):
        versions = self._client.get_latest_versions(model_name)
        latest_version = versions[-1].version
        return latest_version
    
    def get_model_version(self, model_name, version):
        model_version = self._client.get_model_version(model_name, version=version)
        return model_version

    def load_model(self, model_uri):
        return mlflow.pyfunc.load_model(model_uri)