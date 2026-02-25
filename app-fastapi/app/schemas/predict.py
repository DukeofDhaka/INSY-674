from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DataInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enrollee_id: Optional[int] = Field(default=None, ge=0, description="Enrollee ID must be non-negative")
    city: str = Field(min_length=1, description="City name cannot be empty")
    city_development_index: float = Field(
        ge=0.0, le=1.0, description="City development index must be between 0 and 1"
    )
    gender: Optional[str] = None
    relevent_experience: str = Field(min_length=1, description="Relevant experience cannot be empty")
    enrolled_university: Optional[str] = None
    education_level: Optional[str] = None
    major_discipline: Optional[str] = None
    experience: Optional[str] = None
    company_size: Optional[str] = None
    company_type: Optional[str] = None
    last_new_job: Optional[str] = None
    training_hours: float = Field(gt=0, le=1000, description="Training hours must be positive and reasonable")


class MultipleDataInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: List[DataInput] = Field(min_length=1, max_length=1000)


class PredictionResults(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predictions: List[int]
    probabilities: Optional[List[float]] = None
    model_version: str
