from dataclasses import dataclass
from app.ai_model import LogisticRegressionModel
from app.experiment_tracking import MLFlowExperimentTracker
from app.model import PipelineResponse, Payload
from app.pipelines import TrainingPipeline, PredictionPipeline


@dataclass
class MLService:
    def train_ml_service(self, payload: Payload):
        pipeline = TrainingPipeline(
            experiment_tracker=MLFlowExperimentTracker(),
            model_trainer=LogisticRegressionModel(),
            payload=payload,
        )

        training_response: PipelineResponse = pipeline.run()
        return training_response

    def predict_ml_service(self, payload: Payload):
        pipeline = PredictionPipeline(
            experiment_tracker=MLFlowExperimentTracker(),
            model_trainer=LogisticRegressionModel(),
            payload=payload,
        )

        prediction_response: PipelineResponse = pipeline.run()
        return prediction_response
