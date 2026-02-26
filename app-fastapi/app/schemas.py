from pydantic import BaseModel
from typing import Any, Optional, List


class DriftResponse(BaseModel):
    drift_detected: bool
    details: Optional[Any] = None


class DriftRequest(BaseModel):
    inputs: List[dict]


class Health(BaseModel):
    name: str
    api_version: str
    model_version: str
    model_metadata: Optional[Any] = None


class ModelMetadata(BaseModel):
    class Config:
        extra = "allow"


class MultipleDataInputs(BaseModel):
    inputs: List[dict]


class PredictionResults(BaseModel):
    predictions: List[Any]
    class_names: Optional[List[str]] = None
