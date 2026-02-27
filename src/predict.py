from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

import pandas as pd

from src import __version__ as _version
from src.config.core import config
from src.processing.data_manager import load_pipeline, model_file_name


@lru_cache(maxsize=1)
def _load_trained_pipeline():
    """Load and cache the trained model pipeline."""
    return load_pipeline(file_name=model_file_name())


def make_prediction(input_data: Any) -> Dict[str, Any]:
    """Make predictions from raw input data.

    Accepts a pandas DataFrame or an object convertible to a DataFrame
    (e.g., list[dict], dict[str, list], etc.).
    """
    if isinstance(input_data, pd.DataFrame):
        data = input_data.copy()
    else:
        data = pd.DataFrame(input_data)

    if config.ml_config.target in data.columns:
        data = data.drop(columns=[config.ml_config.target])

    trained_pipe = _load_trained_pipeline()
    if hasattr(trained_pipe, "feature_names_in_"):
        expected_features = list(trained_pipe.feature_names_in_)
        for feature in expected_features:
            if feature not in data.columns:
                data[feature] = None
        unexpected_features = [feature for feature in data.columns if feature not in expected_features]
        if unexpected_features:
            data = data.drop(columns=unexpected_features)
        data = data[expected_features]
    predictions = trained_pipe.predict(data)
    results: Dict[str, Any] = {
        "predictions": [int(pred) for pred in predictions],
        "model_version": _version,
    }

    if hasattr(trained_pipe, "predict_proba"):
        probabilities = trained_pipe.predict_proba(data)[:, 1]
        results["probabilities"] = [float(prob) for prob in probabilities]

    return results
