from typing import Optional, TypedDict
from pydantic import BaseModel
from uuid import uuid4


class PayloadRawData(TypedDict):
    X: list[list[float]]
    y: Optional[list[int]] = None


class Dataset(BaseModel):
    X: list[list[float]]
    y: Optional[list[int]] = None


class Payload(BaseModel):
    data: Dataset
    model_name: str
    model_version: Optional[str] = ""
    experiment_name: Optional[str] = f"experiment-{uuid4()}"
