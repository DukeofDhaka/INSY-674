from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.predict import DataInput

FeatureDriftStatus = Literal["ok", "warn", "drifted"]
OverallDriftStatus = Literal["ok", "warn", "drifted", "insufficient_data"]


class DriftRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inputs: List[DataInput] = Field(min_length=1, max_length=1000)


class DriftThresholds(BaseModel):
    warn: float
    drifted: float


class FeatureDriftReport(BaseModel):
    feature_name: str
    feature_type: Literal["numeric", "categorical"]
    psi: float
    status: FeatureDriftStatus
    expected_missing_rate: float
    observed_missing_rate: float
    missing_rate_delta: float
    expected_distribution: Dict[str, float]
    observed_distribution: Dict[str, float]


class DriftResponse(BaseModel):
    model_version: str
    baseline_version: str
    n_records: int
    overall_status: OverallDriftStatus
    thresholds: DriftThresholds
    top_drifting_features: List[str]
    feature_reports: List[FeatureDriftReport]
    message: Optional[str] = None
