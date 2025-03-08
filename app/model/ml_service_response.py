from pydantic import BaseModel


class PipelineResponse(BaseModel):
    result: int
