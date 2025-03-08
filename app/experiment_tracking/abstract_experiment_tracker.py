from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AbstractExperimentTracker(ABC):
    @abstractmethod
    def model_uri(self, run_id: str) -> str: ...

    @abstractmethod
    def set_experiment(self, experiment_name): ...

    @abstractmethod
    def log_params(self, run_id, params): ...

    @abstractmethod
    def log_metrics(self, run_id, metrics): ...

    @abstractmethod
    def set_tag(self, run_id, tag_name, tag_value): ...

    @abstractmethod
    def log_model(
        self, sk_model, artifact_path, input_example, registered_model_name
    ): ...

    @abstractmethod
    def start_run(self): ...

    @abstractmethod
    def get_run_info(self): ...

    @abstractmethod
    def log_artifacts(self, artifact_path, artifact_file): ...

    @abstractmethod
    def get_latest_version(self, model_name): ...

    @abstractmethod
    def get_model_version(self, model_name, version): ...

    @abstractmethod
    def load_model(self, model_uri): ...
