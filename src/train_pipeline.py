from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import Dict

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src import __version__ as _version
from src.config.core import config
from src.monitoring.drift import build_drift_baseline
from src.pipeline import pipe
from src.processing.data_manager import load_dataset, save_metadata, save_pipeline


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
        # ---------------- SHAP EXPLANATIONS ----------------
    try:
        # Separate preprocessing and model
        preprocessor = pipe[:-1]
        model = pipe.named_steps["logistic_regression"]

        # Transform validation data
        X_valid_transformed = preprocessor.transform(x_valid)

        # Build SHAP explainer for logistic regression
        explainer = shap.LinearExplainer(model, X_valid_transformed)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_valid_transformed)

        # Save summary plot
        shap.summary_plot(
            shap_values,
            X_valid_transformed,
            show=False
        )
        plt.tight_layout()
        plt.savefig("shap_summary.png")
        plt.close()

        print("SHAP summary plot saved as shap_summary.png")

    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        
    # Build model metadata
    feature_names = list(x_train.columns) if hasattr(x_train, "columns") else []
    metadata = {
        "model_version": _version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _get_git_sha(),
        "metrics": metrics,
        "n_rows": len(data),
        "n_features": len(feature_names),
        "feature_names": feature_names,
    }

    save_pipeline(pipeline_to_persist=pipe)
    save_metadata(metadata=metadata)
    return metrics


if __name__ == "__main__":
    training_metrics = run_training()
    print(
        f"Training complete | accuracy={training_metrics['accuracy']:.4f} "
        f"roc_auc={training_metrics['roc_auc']:.4f}"
    )
