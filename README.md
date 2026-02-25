# INSY-674 End-to-End ML Project

This repository contains an end-to-end machine learning workflow for predicting whether a candidate is likely to look for a new job (`target`), using the HR analytics dataset included in `src/data`.
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
- `GET /api/v1/health` : API + model health/version
- `POST /api/v1/predict` : batch predictions

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
- `https://<your-render-service>.onrender.com/api/v1/health`
- `https://<your-render-service>.onrender.com/docs`

The service uses the Docker build defined in `Dockerfile`, including model training during image build.

## API smoke test
```bash
curl -s http://127.0.0.1:8000/api/v1/health
```

## Notes
- The trained model is versioned and saved under `src/trained_models/`.
- Retraining overwrites old model artifacts to keep one active version per package version.
