from typing import Annotated

from fastapi import APIRouter, Depends

from app.experiment_tracking import AbstractExperimentTracker, MLFlowExperimentTracker
from app.model import InputData, PredictionResult
from app.model_registry import MLFlowModelRegistry, ModelRegistry, ModelRegistryConfig

router = APIRouter()


@router.get("/")
async def root() -> dict[str, str]:
    return {"message": "access the /predict or /train endpoint"}


@router.get("/models")
async def models(
    model_registry: Annotated[ModelRegistry, Depends(MLFlowModelRegistry)],
) -> dict[str, list[str]]:
    models = model_registry.get_all_models()
    return {"models": models}


@router.get("/models/{model_name}/versions")
async def get_versions(
    model_name: str,
    model_registry: Annotated[ModelRegistry, Depends(MLFlowModelRegistry)],
) -> dict[str, list[int]]:
    versions = model_registry.get_all_model_versions(model_name=model_name)
    return {"versions": versions}


@router.post("/model/{model_name}/train")
async def train(
    model_name: str,
    data: InputData,
    model_registry: Annotated[ModelRegistry, Depends(MLFlowModelRegistry)],
    experiment_tracker: Annotated[
        AbstractExperimentTracker, Depends(MLFlowExperimentTracker)
    ],
) -> ModelRegistryConfig:
    experiment_tracker.set_experiment()
    with experiment_tracker.start_run() as run:
        run_id = run.__dict__["_info"].__dict__["_run_id"]
        config = ModelRegistryConfig(model_name=model_name)
        model_pipeline = model_registry.get_latest_model(config=config)
        model = model_pipeline.train(data, run_id)
        return model_registry.create_model(model=model, config=config)


@router.post("/model/{model_name}/{model_version}/predict")
async def predict(
    model_name: str,
    model_version: str,
    data: InputData,
    model_registry: Annotated[ModelRegistry, Depends(MLFlowModelRegistry)],
) -> PredictionResult:
    model_registry_config = ModelRegistryConfig(
        model_name=model_name,
        model_version=model_version,
    )
    model_pipeline = model_registry.load(model_registry_config)
    return model_pipeline.predict(data)
