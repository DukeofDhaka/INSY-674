from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Any, Dict

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from src import __version__ as _version
from src.config.core import config
from src.monitoring.drift import build_drift_baseline
from src.pipeline import pipe
from src.processing.data_manager import (
    load_dataset,
    save_drift_baseline,
    save_evaluation_report,
    save_metadata,
    save_pipeline,
)


def _get_git_sha() -> str:
    """Get current git commit SHA, or 'unknown' if not in a git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha
    except Exception:
        return "unknown"


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _build_evaluation_report(
    *,
    y_true: Any,
    y_pred: Any,
    metrics: Dict[str, float],
    git_sha: str,
    feature_names: list[str],
) -> Dict[str, Any]:
    class_zero_rate = float((y_true == 0).mean())
    class_one_rate = float((y_true == 1).mean())
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "model_version": _version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "validation_rows": int(len(y_true)),
        "metrics": metrics,
        "target_distribution": {
            "class_0_rate": class_zero_rate,
            "class_1_rate": class_one_rate,
        },
        "classification_report": _to_builtin(report),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": _to_builtin(conf_matrix),
        },
        "feature_names": feature_names,
    }


def run_training() -> Dict[str, float]:
    """Train and persist the model pipeline."""

    data = load_dataset(file_name=config.app_config.training_data_file, drop_features=True)
    x = data.drop(columns=[config.ml_config.target])
    y = data[config.ml_config.target].astype(int)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=config.ml_config.test_size,
        random_state=config.ml_config.random_state,
        stratify=y,
    )

    pipe.fit(x_train, y_train)

    y_pred = pipe.predict(x_valid)
    y_proba = pipe.predict_proba(x_valid)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_valid, y_pred)),
        "roc_auc": float(roc_auc_score(y_valid, y_proba)),
    }
    drift_baseline = build_drift_baseline(x_train=x_train, config=config.ml_config)
    git_sha = _get_git_sha()

    # Build model metadata
    feature_names = list(x_train.columns) if hasattr(x_train, "columns") else []
    metadata = {
        "model_version": _version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "metrics": metrics,
        "n_rows": len(data),
        "n_features": len(feature_names),
        "feature_names": feature_names,
    }
    evaluation_report = _build_evaluation_report(
        y_true=y_valid,
        y_pred=y_pred,
        metrics=metrics,
        git_sha=git_sha,
        feature_names=feature_names,
    )

    save_pipeline(pipeline_to_persist=pipe)
    save_metadata(metadata=metadata)
    save_evaluation_report(report=evaluation_report)
    save_drift_baseline(baseline=drift_baseline)
    return metrics


if __name__ == "__main__":
    training_metrics = run_training()
    print(
        f"Training complete | accuracy={training_metrics['accuracy']:.4f} "
        f"roc_auc={training_metrics['roc_auc']:.4f}"
    )
