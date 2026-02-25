from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field
from pydantic import BaseModel


class DataInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enrollee_id: Optional[int] = None
    city: str
    city_development_index: float
    gender: Optional[str] = None
    relevent_experience: str
    enrolled_university: Optional[str] = None
    education_level: Optional[str] = None
    major_discipline: Optional[str] = None
    experience: Optional[str] = None
    company_size: Optional[str] = None
    company_type: Optional[str] = None
    last_new_job: Optional[str] = None
    training_hours: float


class MultipleDataInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: List[DataInput] = Field(min_length=1, max_length=1000)


class PredictionResults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predictions: List[int]
    probabilities: Optional[List[float]] = None
    model_version: str
