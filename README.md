# INSY-674 End-to-End ML Project

[![CI](https://github.com/tyagi14/INSY-674/actions/workflows/ci.yml/badge.svg)](https://github.com/tyagi14/INSY-674/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

This repository contains an end-to-end machine learning workflow for predicting whether a candidate is likely to look for a new job (`target`), using the HR analytics dataset included in `src/data`.

## Live deployment

| Endpoint | URL |
|----------|-----|
| API Docs (Swagger) | [https://insy674-ml-api.onrender.com/docs](https://insy674-ml-api.onrender.com/docs) |
| Health Check | [https://insy674-ml-api.onrender.com/api/v1/health](https://insy674-ml-api.onrender.com/api/v1/health) |
| Predict | `POST` [https://insy674-ml-api.onrender.com/api/v1/predict](https://insy674-ml-api.onrender.com/api/v1/predict) |

> **Note:** The free-tier Render instance spins down after inactivity. The first request after idle may take ~30-60 seconds to cold-start.

## Workflow implemented (Research to Production)
### Phase 1: Research
- Data analysis in Jupyter notebooks
- Model experimentation
- Pipeline creation

### Phase 2: Production Code
- Central config management via `src/config.yml`
- Modularized processing in `src/processing/`
- Reusable training and inference modules (`src/train_pipeline.py`, `src/predict.py`)

### Phase 3: Package Development
- Python package structure under `src/`
- Versioning in `src/VERSION`
- Dependency management with split requirements and packaging metadata (`setup.py`, `pyproject.toml`)

### Phase 4: API Development
- FastAPI REST endpoints (`/api/v1/health`, `/api/v1/predict`)
- Pydantic validation schemas
- Application logging with Loguru (with safe fallback)

### Phase 5: Deployment
- Docker containerization (`Dockerfile`)
- Render blueprint deployment config (`render.yaml`)
- GitHub Actions CI pipeline (`.github/workflows/ci.yml`)

## Stack used
- ML & Data Processing: Scikit-learn, Feature-engine, Pandas, NumPy
- Production & Deployment: FastAPI, Docker, Render blueprint config, PyPI-ready packaging
- Development Tools: Jupyter, Tox, Loguru, Git/GitHub

## Project structure
- `src/`: ML package (config, preprocessing, training, prediction, model artifacts)
- `app-fastapi/`: REST API for model serving
- `requirements/`: split dependency files for dev/research/production
- `notebooks/`: experimentation notebooks

## Quick start
1) Create and activate virtual environment
2) Install dependencies
3) Train model
4) Run API

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements/research.txt
python -m src.train_pipeline
uvicorn app.main:app --app-dir app-fastapi --reload
```

## API endpoints
- `GET /` : basic welcome page
- `GET /api/v1/health` : API + model health/version (+ optional model metadata if available)
- `POST /api/v1/predict` : batch predictions

Example health response with metadata:

```json
{
  "name": "INSY-674 Employability Prediction API",
  "api_version": "0.1.0",
  "model_version": "0.1.0",
  "model_metadata": {
    "model_version": "0.1.0",
    "trained_at": "2026-02-25T19:00:00+00:00",
    "git_sha": "abc1234",
    "metrics": {
      "accuracy": 0.81,
      "roc_auc": 0.88
    },
    "n_rows": 19158,
    "n_features": 12,
    "feature_names": ["city", "city_development_index", "gender"]
  }
}
```

Example payload:

```json
{
  "inputs": [
    {
      "city": "city_41",
      "city_development_index": 0.827,
      "gender": "Male",
      "relevent_experience": "Has relevent experience",
      "enrolled_university": "Full time course",
      "education_level": "Graduate",
      "major_discipline": "STEM",
      "experience": "9",
      "company_size": "<10",
      "company_type": "Pvt Ltd",
      "last_new_job": "1",
      "training_hours": 21
    }
  ]
}
```

## Run tests
```bash
pytest app-fastapi/app/tests -q
```

## Production run (Docker)
```bash
docker build -t insy674-ml-api .
docker run --rm -p 8000:8000 insy674-ml-api
```
## Deploy on Render (Free tier)
1) Push this repository to GitHub.
2) In Render dashboard, choose **New +** → **Blueprint**.
3) Select this repository. Render will read `render.yaml` and create the web service.
4) Confirm deploy; once live, open:
- `https://insy674-ml-api.onrender.com/api/v1/health`
- `https://insy674-ml-api.onrender.com/docs`

The service uses the Docker build defined in `Dockerfile`, including model training during image build.

## API smoke test
```bash
curl -s http://127.0.0.1:8000/api/v1/health
```

## Notes
- The trained model is versioned and saved under `src/trained_models/`.
- Retraining overwrites old model artifacts to keep one active version per package version.
