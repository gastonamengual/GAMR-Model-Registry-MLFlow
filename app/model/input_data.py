from pydantic import BaseModel


class InputData(BaseModel):
    X: list[list[float]]
    y: list[int] | None = None
