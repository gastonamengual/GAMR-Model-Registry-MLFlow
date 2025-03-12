from pydantic import BaseModel


class PredictionResult(BaseModel):
    prediction: int
