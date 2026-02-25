# INSY-674 End-to-End ML Project

This repository contains an end-to-end machine learning workflow for predicting whether a candidate is likely to look for a new job (`target`), using the HR analytics dataset included in `src/data`.

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

## Notes
- The trained model is versioned and saved under `src/trained_models/`.
- Retraining overwrites old model artifacts to keep one active version per package version.
