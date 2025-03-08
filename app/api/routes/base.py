from typing import Optional
from fastapi import APIRouter, Query, Response

from app.model import Payload, Dataset, PayloadRawData
from app.ml_service import MLService

router = APIRouter()


@router.get("/")
async def root() -> dict[str, str]:
    return {"message": "access the /predict or /train endpoint"}


@router.post("/model/{model_name}/train")
async def train(
    model_name: str,
    data: PayloadRawData,
    experiment_name: Optional[str] = Query(
        None, description="Optional experiment name"
    ),
):
    print(data)
    X, y = data["X"], data["y"]

    payload = Payload(
        model_name=model_name,
        data=Dataset(X=X, y=y),
        experiment_nam=experiment_name,
    )

    prediction_response = MLService().train_ml_service(payload)
    return {"model_version": prediction_response.result}


@router.post("/model/{model_name}/{model_version}/predict")
async def predict(
    model_name: str,
    model_version: str,
    data: PayloadRawData,
) -> Response:
    X = data["X"]

    payload = Payload(
        model_name=model_name,
        model_version=model_version,
        data=Dataset(X=X),
    )

    prediction_response = MLService().predict_ml_service(payload)
    return {"prediction": prediction_response.result}
