from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src import __version__ as _version

MISSING_TOKEN = "__MISSING__"
OTHER_TOKEN = "__OTHER__"
PSI_EPSILON = 1e-6


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_numeric_edges(edges: List[float]) -> List[float]:
    unique_edges = sorted({float(edge) for edge in edges if pd.notna(edge)})
    if len(unique_edges) >= 2:
        return unique_edges
    if not unique_edges:
        return [0.0, 1.0]
    anchor = unique_edges[0]
    return [anchor - 0.5, anchor + 0.5]


def _prepare_numeric_edges(series: pd.Series, n_quantiles: int = 10) -> List[float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return [0.0, 1.0]
    quantiles = np.linspace(0.0, 1.0, n_quantiles + 1)
    edges = numeric.quantile(quantiles).tolist()
    return _normalize_numeric_edges(edges)


def _build_cut_bins(numeric_edges: List[float]) -> Tuple[List[float], List[str]]:
    normalized_edges = _normalize_numeric_edges(numeric_edges)
    inner_edges = normalized_edges[1:-1]
    cut_bins = [-np.inf, *inner_edges, np.inf]
    labels = [f"bin_{idx}" for idx in range(len(cut_bins) - 1)]
    return cut_bins, labels


def _normalize_categorical(series: pd.Series) -> Tuple[pd.Series, float]:
    as_object = series.astype("object")
    as_str = as_object.astype(str).str.strip()
    missing_mask = as_object.isna() | as_str.eq("")
    missing_rate = float(missing_mask.mean()) if len(series) else 0.0
    normalized = as_object.where(~missing_mask, MISSING_TOKEN).astype(str)
    return normalized, missing_rate


def _numeric_distribution(series: pd.Series, numeric_edges: List[float]) -> Tuple[Dict[str, float], float]:
    numeric = pd.to_numeric(series, errors="coerce")
    missing_rate = float(numeric.isna().mean()) if len(numeric) else 0.0
    non_missing = numeric.dropna()

    cut_bins, labels = _build_cut_bins(numeric_edges)
    distribution = {label: 0.0 for label in labels}

    if not non_missing.empty:
        bucketed = pd.cut(non_missing, bins=cut_bins, labels=labels, include_lowest=True)
        non_missing_distribution = bucketed.value_counts(normalize=True).to_dict()
        scale = 1.0 - missing_rate
        for label, value in non_missing_distribution.items():
            distribution[str(label)] = float(value) * scale

    distribution[MISSING_TOKEN] = missing_rate
    return distribution, missing_rate


def _categorical_distribution(
    series: pd.Series, top_categories: List[str]
) -> Tuple[Dict[str, float], float]:
    normalized, missing_rate = _normalize_categorical(series)
    top_category_set = set(top_categories)

    mapped = normalized.map(
        lambda value: value
        if value == MISSING_TOKEN or value in top_category_set
        else OTHER_TOKEN
    )

    distribution = {category: 0.0 for category in top_categories}
    distribution[OTHER_TOKEN] = 0.0
    distribution[MISSING_TOKEN] = 0.0

    observed_distribution = mapped.value_counts(normalize=True).to_dict()
    for key, value in observed_distribution.items():
        distribution[key] = float(value)

    return distribution, missing_rate


def _calculate_psi(
    expected_distribution: Dict[str, float],
    observed_distribution: Dict[str, float],
    epsilon: float = PSI_EPSILON,
) -> float:
    all_buckets = set(expected_distribution) | set(observed_distribution)
    psi_value = 0.0
    for bucket in all_buckets:
        expected = max(_to_float(expected_distribution.get(bucket, 0.0)), epsilon)
        observed = max(_to_float(observed_distribution.get(bucket, 0.0)), epsilon)
        psi_value += (observed - expected) * math.log(observed / expected)
    return float(psi_value)


def _status_from_psi(psi: float, warn_threshold: float, alert_threshold: float) -> str:
    if psi < warn_threshold:
        return "ok"
    if psi < alert_threshold:
        return "warn"
    return "drifted"


def build_drift_baseline(x_train: pd.DataFrame, config: Any) -> Dict[str, Any]:
    """Build reference distributions from training data for drift monitoring."""
    feature_baselines: Dict[str, Dict[str, Any]] = {}
    max_categories = int(getattr(config, "drift_max_categories", 20))

    for feature_name in x_train.columns:
        feature_series = x_train[feature_name]
        is_numeric = pd.api.types.is_numeric_dtype(feature_series)

        if is_numeric:
            numeric_edges = _prepare_numeric_edges(feature_series)
            distribution, missing_rate = _numeric_distribution(feature_series, numeric_edges)
            non_missing = pd.to_numeric(feature_series, errors="coerce").dropna()
            feature_baselines[feature_name] = {
                "feature_type": "numeric",
                "missing_rate": missing_rate,
                "bin_edges": numeric_edges,
                "distribution": distribution,
                "summary": {
                    "mean": _to_float(non_missing.mean()) if not non_missing.empty else 0.0,
                    "std": _to_float(non_missing.std()) if not non_missing.empty else 0.0,
                    "min": _to_float(non_missing.min()) if not non_missing.empty else 0.0,
                    "max": _to_float(non_missing.max()) if not non_missing.empty else 0.0,
                },
            }
            continue

        normalized, missing_rate = _normalize_categorical(feature_series)
        non_missing = normalized[normalized != MISSING_TOKEN]
        top_categories = non_missing.value_counts().head(max_categories).index.tolist()
        distribution, _ = _categorical_distribution(feature_series, top_categories)
        feature_baselines[feature_name] = {
            "feature_type": "categorical",
            "missing_rate": missing_rate,
            "top_categories": top_categories,
            "distribution": distribution,
            "summary": {
                "n_unique_non_missing": int(non_missing.nunique()),
                "top_categories": top_categories,
            },
        }

    return {
        "model_version": _version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(x_train)),
        "n_features": int(x_train.shape[1]),
        "features": feature_baselines,
    }


def compute_drift_report(
    input_df: pd.DataFrame,
    baseline: Dict[str, Any],
    config: Any,
) -> Dict[str, Any]:
    """Compare incoming batch distributions against baseline and return PSI report."""
    warn_threshold = float(getattr(config, "drift_warn_threshold", 0.1))
    alert_threshold = float(getattr(config, "drift_alert_threshold", 0.25))
    min_samples = int(getattr(config, "drift_min_samples", 50))

    n_records = int(len(input_df))
    feature_reports: List[Dict[str, Any]] = []

    baseline_features = baseline.get("features", {})
    for feature_name, feature_baseline in baseline_features.items():
        if feature_name in input_df.columns:
            observed_series = input_df[feature_name]
        else:
            observed_series = pd.Series([None] * n_records)

        feature_type = feature_baseline.get("feature_type", "categorical")
        expected_distribution = {
            str(key): _to_float(value)
            for key, value in feature_baseline.get("distribution", {}).items()
        }

        if feature_type == "numeric":
            numeric_edges = [
                _to_float(edge)
                for edge in feature_baseline.get("bin_edges", [0.0, 1.0])
            ]
            observed_distribution, observed_missing_rate = _numeric_distribution(
                observed_series,
                numeric_edges,
            )
        else:
            top_categories = [str(value) for value in feature_baseline.get("top_categories", [])]
            observed_distribution, observed_missing_rate = _categorical_distribution(
                observed_series,
                top_categories,
            )

        psi_value = _calculate_psi(expected_distribution, observed_distribution)
        status = _status_from_psi(psi_value, warn_threshold, alert_threshold)
        expected_missing_rate = _to_float(feature_baseline.get("missing_rate", 0.0))

        feature_reports.append(
            {
                "feature_name": feature_name,
                "feature_type": feature_type,
                "psi": float(psi_value),
                "status": status,
                "expected_missing_rate": expected_missing_rate,
                "observed_missing_rate": float(observed_missing_rate),
                "missing_rate_delta": float(observed_missing_rate - expected_missing_rate),
                "expected_distribution": expected_distribution,
                "observed_distribution": observed_distribution,
            }
        )

    top_drifting_features = [
        report["feature_name"]
        for report in sorted(feature_reports, key=lambda item: item["psi"], reverse=True)
        if report["status"] in {"warn", "drifted"}
    ][:5]

    if n_records < min_samples:
        overall_status = "insufficient_data"
        message = (
            f"Received {n_records} records. Minimum required for drift evaluation is {min_samples}."
        )
    elif any(report["status"] == "drifted" for report in feature_reports):
        overall_status = "drifted"
        message = None
    elif any(report["status"] == "warn" for report in feature_reports):
        overall_status = "warn"
        message = None
    else:
        overall_status = "ok"
        message = None

    return {
        "model_version": _version,
        "baseline_version": str(baseline.get("model_version", "unknown")),
        "n_records": n_records,
        "overall_status": overall_status,
        "thresholds": {
            "warn": warn_threshold,
            "drifted": alert_threshold,
        },
        "top_drifting_features": top_drifting_features,
        "feature_reports": feature_reports,
        "message": message,
    }
