# Model Card: Employability Prediction Model

## Model Details
- Model name: `employability_model_v0.1.0`
- Model version: `0.1.0`
- Model type: Scikit-learn `LogisticRegression` in preprocessing pipeline
- Training script: `python -m src.train_pipeline`
- Last recorded training run (UTC): `2026-02-27T04:17:07+00:00`
- Training commit SHA: `72445c3`

## Intended Use
- Predict whether a candidate is likely to look for a new job (`target`).
- Intended for aggregate planning and risk monitoring support.
- Not intended as a sole decision-maker for hiring, promotion, compensation, or termination.

## Data
- Source: HR analytics dataset in `src/data/train.csv`
- Training rows: `19,158`
- Validation rows: `3,832`
- Model input features: `12`

## Performance (Latest Validation Snapshot)
- Accuracy: `0.7309`
- ROC-AUC: `0.7768`
- Class distribution in validation:
  - Class `0`: `75.08%`
  - Class `1`: `24.92%`

### Classification Summary
- Positive class (`1`) precision: `0.4735`
- Positive class (`1`) recall: `0.7120`
- Positive class (`1`) F1: `0.5688`
- Weighted F1: `0.7457`

### Confusion Matrix
- Label order: `[0, 1]`
- Matrix:
  - `[2121, 756]`
  - `[275, 680]`

## Fairness, Risks, and Limitations
- No formal subgroup fairness audit is yet included in this repository.
- Inputs may contain proxy attributes (for example location and background variables) that can introduce bias.
- Performance can degrade under data drift, especially for changing city/job-market distributions.
- This model should be used with human review and periodic bias/performance checks.

## Monitoring and Retraining Policy
- Data drift is monitored with `POST /api/v1/monitor/drift`.
- Recommended action policy:
  - `overall_status = warn`: monitor closely, investigate feature shifts.
  - `overall_status = drifted`: schedule retraining and re-validation.
  - repeated `drifted` signals or material KPI drop: prioritize immediate retraining.
- Retraining output artifacts:
  - model: `src/trained_models/*<version>.pkl`
  - metadata: `src/trained_models/*_metadata.json`
  - evaluation report: `src/trained_models/*_evaluation.json`
  - drift baseline: `src/trained_models/*_drift_baseline.json`

## Evaluation Artifact Contract
- File path pattern: `src/trained_models/<pipeline_save_file><model_version>_evaluation.json`
- Includes:
  - `metrics` (accuracy, roc_auc)
  - `classification_report` (per-class precision/recall/f1/support)
  - `confusion_matrix` (labels + matrix)
  - validation row count, target distribution, feature list, timestamp, git SHA
