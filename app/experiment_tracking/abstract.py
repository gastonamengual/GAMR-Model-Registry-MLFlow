from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class AbstractExperimentTracker(ABC):
    @abstractmethod
    def set_experiment(self, experiment_name: str | None = None) -> None: ...

    @abstractmethod
    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None: ...

    @abstractmethod
    def start_run(self) -> Any: ...
