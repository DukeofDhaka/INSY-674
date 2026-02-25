from __future__ import annotations

from typing import Dict

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config.core import config
from src.monitoring.drift import build_drift_baseline
from src.pipeline import pipe
from src.processing.data_manager import load_dataset, save_drift_baseline, save_pipeline


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
    save_pipeline(pipeline_to_persist=pipe)
    save_drift_baseline(baseline=drift_baseline)
    return metrics


if __name__ == "__main__":
    training_metrics = run_training()
    print(
        f"Training complete | accuracy={training_metrics['accuracy']:.4f} "
        f"roc_auc={training_metrics['roc_auc']:.4f}"
    )
