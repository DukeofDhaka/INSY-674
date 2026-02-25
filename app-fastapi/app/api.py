from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder

from src import __version__ as model_version
from src.predict import make_prediction
from src.processing.data_manager import load_metadata

from app import __version__, schemas
from app.config import get_logger, settings

api_router = APIRouter()
logger = get_logger(__name__)


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    logger.info("Health endpoint requested")
    
    # Load model metadata if available
    metadata_dict = load_metadata()
    model_metadata = None
    if metadata_dict:
        try:
            model_metadata = schemas.ModelMetadata(**metadata_dict)
        except Exception as e:
            logger.warning(f"Failed to parse model metadata: {e}")
    
    health_response = schemas.Health(
        name=settings.PROJECT_NAME,
        api_version=__version__,
        model_version=model_version,
        model_metadata=model_metadata,
    )
    return health_response.model_dump()


@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs) -> Any:
    try:
        input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
        logger.info(f"Prediction requested for {len(input_df)} row(s)")
        results = make_prediction(input_data=input_df.replace({np.nan: None}))
        return results
    except ValueError as exc:
        logger.exception(f"Prediction input validation failure: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"Unexpected prediction failure: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal server error.",
        ) from exc
