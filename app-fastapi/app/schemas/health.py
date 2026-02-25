from typing import Dict, List, Optional

from pydantic import BaseModel


class ModelMetrics(BaseModel):
    accuracy: float
    roc_auc: float


class ModelMetadata(BaseModel):
    model_version: str
    trained_at: str
    git_sha: str
    metrics: ModelMetrics
    n_rows: int
    n_features: int
    feature_names: List[str]


class Health(BaseModel):
    name: str
    api_version: str
    model_version: str
    model_metadata: Optional[ModelMetadata] = None
