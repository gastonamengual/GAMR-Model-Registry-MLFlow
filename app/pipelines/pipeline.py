from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.model import Payload
from app.experiment_tracking import MLFlowExperimentTracker
from app.ai_model import AbstractModel
from app.model import PipelineResponse

@dataclass
class Pipeline(ABC):
    experiment_tracker: MLFlowExperimentTracker
    model_trainer: AbstractModel
    payload: Payload

    @abstractmethod
    def execute(self): pass
    
    def run(self) -> PipelineResponse:
        print("Setting experiment")
        print(self.payload.experiment_name)
        self.experiment_tracker.set_experiment(self.payload.experiment_name)
        with self.experiment_tracker.start_run():
            pipeline_result = self.execute()
            prediction_response = PipelineResponse(
                result=pipeline_result
            )
            return prediction_response
