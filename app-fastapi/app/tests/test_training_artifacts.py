from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.processing import data_manager
from src.train_pipeline import _build_evaluation_report


def test_evaluation_report_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(data_manager, "TRAINED_MODEL_DIR", tmp_path)
    payload = {
        "model_version": "0.1.0",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "git_sha": "abc123",
        "validation_rows": 10,
        "metrics": {"accuracy": 0.8, "roc_auc": 0.9},
        "target_distribution": {"class_0_rate": 0.6, "class_1_rate": 0.4},
        "classification_report": {
            "0": {"precision": 0.8, "recall": 0.9, "f1-score": 0.85, "support": 6},
            "1": {"precision": 0.75, "recall": 0.5, "f1-score": 0.6, "support": 4},
            "accuracy": 0.8,
        },
        "confusion_matrix": {"labels": [0, 1], "matrix": [[5, 1], [2, 2]]},
        "feature_names": ["city", "training_hours"],
    }

    data_manager.save_evaluation_report(payload)
    loaded = data_manager.load_evaluation_report()
    assert loaded == payload


def test_build_evaluation_report_contains_expected_sections() -> None:
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([0, 1, 0, 0])
    metrics = {"accuracy": 0.75, "roc_auc": 0.8}

    report = _build_evaluation_report(
        y_true=y_true,
        y_pred=y_pred,
        metrics=metrics,
        git_sha="abc123",
        feature_names=["city", "training_hours"],
    )

    assert report["metrics"] == metrics
    assert report["validation_rows"] == 4
    assert report["confusion_matrix"]["labels"] == [0, 1]
    assert report["confusion_matrix"]["matrix"] == [[2, 0], [1, 1]]
    assert "macro avg" in report["classification_report"]
    assert report["feature_names"] == ["city", "training_hours"]
