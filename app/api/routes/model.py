from typing import Annotated

from fastapi import APIRouter, Depends

from app.experiment_tracking import MLFlowExperimentTracker
from app.model import InputData, PredictionResult
from app.model_registry import MLFlowModelRegistry, ModelRegistry, ModelRegistryConfig
from app.monitoring import PROMETHEUS_MONITOR

router = APIRouter(prefix="/model")


@router.get("/")
async def models(
    model_registry: Annotated[ModelRegistry, Depends(MLFlowModelRegistry)],
) -> dict[str, list[str]]:
    start_time = PROMETHEUS_MONITOR.start_time()
    PROMETHEUS_MONITOR.increase_current_requests("/model/")

    models = model_registry.get_all_models()
    PROMETHEUS_MONITOR.record_request_duration(start_time, "/model/")
    PROMETHEUS_MONITOR.decrease_current_requests("/model/")

    return {"models": models}


@router.get("/{model_name}/versions")
async def get_versions(
    model_name: str,
    model_registry: Annotated[ModelRegistry, Depends(MLFlowModelRegistry)],
) -> dict[str, list[int]]:
    start_time = PROMETHEUS_MONITOR.start_time()
    PROMETHEUS_MONITOR.increase_current_requests("/model_versions")

    versions = model_registry.get_all_model_versions(model_name=model_name)

    PROMETHEUS_MONITOR.record_request_duration(start_time, "/model_versions")
    PROMETHEUS_MONITOR.decrease_current_requests("/model_versions")

    return {"versions": versions}


@router.post("/{model_name}/train")
async def train(
    model_name: str,
    data: InputData,
) -> ModelRegistryConfig:
    experiment_tracker = MLFlowExperimentTracker()
    model_registry = MLFlowModelRegistry()
    experiment_tracker.set_experiment()
    with experiment_tracker.start_run() as run:
        run_id = run.__dict__["_info"].__dict__["_run_id"]
        config = ModelRegistryConfig(model_name=model_name)

        start_time = PROMETHEUS_MONITOR.start_time()
        PROMETHEUS_MONITOR.increase_current_requests("/train/")

        model_pipeline = model_registry.get_latest_model(config=config)
        model = model_pipeline.train(data, run_id)
        result = model_registry.create_model(model=model, config=config)

        PROMETHEUS_MONITOR.record_request_duration(start_time, "/train/")
        PROMETHEUS_MONITOR.decrease_current_requests("/train/")

        return result


@router.post("/{model_name}/{model_version}/predict")
async def predict(
    model_name: str,
    model_version: str,
    data: InputData,
) -> PredictionResult:
    model_registry = MLFlowModelRegistry()
    model_registry_config = ModelRegistryConfig(
        model_name=model_name,
        model_version=model_version,
    )
    start_time = PROMETHEUS_MONITOR.start_time()
    PROMETHEUS_MONITOR.increase_current_requests("/predict/")

    model_pipeline = model_registry.load(model_registry_config)
    result = model_pipeline.predict(data)

    PROMETHEUS_MONITOR.record_request_duration(start_time, "/predict/")
    PROMETHEUS_MONITOR.decrease_current_requests("/predict/")

    return result
