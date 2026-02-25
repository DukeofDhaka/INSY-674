from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, cast

import numpy as np
import pandas as pd

from src.monitoring.drift import (
    OTHER_TOKEN,
    build_drift_baseline,
    compute_drift_report,
    _status_from_psi,
)


def _drift_config() -> SimpleNamespace:
    return SimpleNamespace(
        drift_warn_threshold=0.1,
        drift_alert_threshold=0.25,
        drift_min_samples=50,
        drift_max_categories=2,
    )


def _feature_report(drift_report: Dict[str, Any], feature_name: str) -> Dict[str, Any]:
    for feature_report in drift_report["feature_reports"]:
        if feature_report["feature_name"] == feature_name:
            return cast(Dict[str, Any], feature_report)
    raise AssertionError(f"Feature report not found for: {feature_name}")


def test_numeric_psi_increases_with_distribution_shift():
    config = _drift_config()
    train_df = pd.DataFrame({"score": np.linspace(0.0, 1.0, 200)})
    baseline = build_drift_baseline(train_df, config)

    near_baseline_report = compute_drift_report(train_df.copy(), baseline, config)
    shifted_df = pd.DataFrame({"score": np.linspace(0.8, 1.8, 200)})
    shifted_report = compute_drift_report(shifted_df, baseline, config)

    near_psi = _feature_report(near_baseline_report, "score")["psi"]
    shifted_psi = _feature_report(shifted_report, "score")["psi"]

    assert near_psi >= 0.0
    assert shifted_psi > near_psi


def test_categorical_unseen_values_map_to_other_bucket():
    config = _drift_config()
    train_df = pd.DataFrame(
        {
            "city": ["city_a"] * 70 + ["city_b"] * 20 + ["city_c"] * 10,
        }
    )
    baseline = build_drift_baseline(train_df, config)
    incoming_df = pd.DataFrame({"city": ["city_x"] * 60 + ["city_y"] * 40})

    drift_report = compute_drift_report(incoming_df, baseline, config)
    city_report = _feature_report(drift_report, "city")

    assert OTHER_TOKEN in city_report["observed_distribution"]
    assert city_report["observed_distribution"][OTHER_TOKEN] > 0.0


def test_missing_values_are_handled_without_nan_psi():
    config = _drift_config()
    train_df = pd.DataFrame(
        {
            "numeric_feature": [1.0, 2.0, None, 4.0, 5.0] * 30,
            "categorical_feature": ["a", None, "b", "", "c"] * 30,
        }
    )
    baseline = build_drift_baseline(train_df, config)
    incoming_df = pd.DataFrame(
        {
            "numeric_feature": [None, 2.0, 3.0, None, 6.0] * 20,
            "categorical_feature": [None, "a", "", "z", "b"] * 20,
        }
    )

    drift_report = compute_drift_report(incoming_df, baseline, config)
    for feature_report in drift_report["feature_reports"]:
        assert np.isfinite(feature_report["psi"])
        assert 0.0 <= feature_report["observed_missing_rate"] <= 1.0
        assert 0.0 <= feature_report["expected_missing_rate"] <= 1.0


def test_status_threshold_rules_and_insufficient_data_override():
    config = _drift_config()
    assert _status_from_psi(0.05, config.drift_warn_threshold, config.drift_alert_threshold) == "ok"
    assert _status_from_psi(0.12, config.drift_warn_threshold, config.drift_alert_threshold) == "warn"
    assert _status_from_psi(0.30, config.drift_warn_threshold, config.drift_alert_threshold) == "drifted"

    train_df = pd.DataFrame({"x": np.linspace(0.0, 1.0, 100)})
    baseline = build_drift_baseline(train_df, config)
    small_batch = pd.DataFrame({"x": np.linspace(5.0, 6.0, 10)})
    report = compute_drift_report(small_batch, baseline, config)

    assert report["overall_status"] == "insufficient_data"
    assert report["message"] is not None
