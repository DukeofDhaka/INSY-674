from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Mapper(BaseEstimator, TransformerMixin):
    """Map categorical values to numeric values for selected variables."""

    def __init__(
        self,
        variables: List[str],
        mappings: Dict[str, int],
        default_value: int | None = None,
    ):
        if not isinstance(variables, list):
            raise ValueError("variables must be provided as a list")
        self.variables = variables
        self.mappings = mappings
        self.default_value = default_value

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> "Mapper":
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x = x.copy()
        for feature in self.variables:
            x[feature] = x[feature].astype(str).map(self.mappings)
            if self.default_value is not None:
                x[feature] = x[feature].fillna(self.default_value)
        return x
