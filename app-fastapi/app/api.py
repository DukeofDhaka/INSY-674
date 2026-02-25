from __future__ import annotations
import logging

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder

from src import __version__ as model_version
from src.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()
logger = logging.getLogger(__name__)


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    health_response = schemas.Health(
        name=settings.PROJECT_NAME,
        api_version=__version__,
        model_version=model_version,
    )
    return health_response.model_dump()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    logger.info("Making prediction for %s row(s)", len(input_df))
    results = make_prediction(input_data=input_df.replace({np.nan: None}))
    return results
