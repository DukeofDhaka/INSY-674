from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.monitoring.drift import build_drift_baseline


def _valid_input_row(**overrides):
    row = {
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
        "training_hours": 21.0,
    }
    row.update(overrides)
    return row


def _test_drift_config() -> SimpleNamespace:
    return SimpleNamespace(
        drift_warn_threshold=0.1,
        drift_alert_threshold=0.25,
        drift_min_samples=50,
        drift_max_categories=20,
    )


def _baseline_for_api_tests():
    baseline_train = pd.DataFrame(
        {
            "city": ["city_41"] * 70 + ["city_11"] * 30,
            "city_development_index": [0.82] * 100,
            "training_hours": [21.0] * 100,
        }
    )
    return build_drift_baseline(x_train=baseline_train, config=_test_drift_config())


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"]
    assert payload["api_version"]
    assert payload["model_version"]


def test_predict_endpoint(client):
    payload = {
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
                "training_hours": 21.0,
            }
        ]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert isinstance(body["predictions"], list)
    assert len(body["predictions"]) == 1
    assert body["predictions"][0] in [0, 1]
    assert "model_version" in body
    assert "probabilities" in body
    assert isinstance(body["probabilities"], list)
    assert len(body["probabilities"]) == 1


def test_predict_validation_error(client):
    payload = {
        "inputs": [
            {
                "city": "city_41",
                "city_development_index": 0.827,
                "training_hours": 21.0,
            }
        ]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_training_hours(client):
    """Test that negative training hours are rejected."""
    payload = {
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
                "training_hours": -5.0,
            }
        ]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body


def test_predict_invalid_city_development_index_too_high(client):
    """Test that city_development_index > 1.0 is rejected."""
    payload = {
        "inputs": [
            {
                "city": "city_41",
                "city_development_index": 1.5,
                "gender": "Male",
                "relevent_experience": "Has relevent experience",
                "enrolled_university": "Full time course",
                "education_level": "Graduate",
                "major_discipline": "STEM",
                "experience": "9",
                "company_size": "<10",
                "company_type": "Pvt Ltd",
                "last_new_job": "1",
                "training_hours": 21.0,
            }
        ]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body


def test_predict_invalid_city_development_index_negative(client):
    """Test that negative city_development_index is rejected."""
    payload = {
        "inputs": [
            {
                "city": "city_41",
                "city_development_index": -0.5,
                "gender": "Male",
                "relevent_experience": "Has relevent experience",
                "enrolled_university": "Full time course",
                "education_level": "Graduate",
                "major_discipline": "STEM",
                "experience": "9",
                "company_size": "<10",
                "company_type": "Pvt Ltd",
                "last_new_job": "1",
                "training_hours": 21.0,
            }
        ]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body


def test_predict_excessive_training_hours(client):
    """Test that unreasonably high training hours are rejected."""
    payload = {
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
                "training_hours": 5000.0,
            }
        ]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body


def test_predict_empty_city_name(client):
    """Test that empty city name is rejected."""
    payload = {
        "inputs": [
            {
                "city": "",
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
                "training_hours": 21.0,
            }
        ]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body


def test_predict_negative_enrollee_id(client):
    """Test that negative enrollee_id is rejected."""
    payload = {
        "inputs": [
            {
                "enrollee_id": -1,
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
                "training_hours": 21.0,
            }
        ]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body


def test_predict_empty_relevent_experience(client):
    """Test that empty relevent_experience is rejected."""
    payload = {
        "inputs": [
            {
                "city": "city_41",
                "city_development_index": 0.827,
                "gender": "Male",
                "relevent_experience": "",
                "enrolled_university": "Full time course",
                "education_level": "Graduate",
                "major_discipline": "STEM",
                "experience": "9",
                "company_size": "<10",
                "company_type": "Pvt Ltd",
                "last_new_job": "1",
                "training_hours": 21.0,
            }
        ]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    response = client.post("/api/v1/monitor/drift", json=payload)
    assert response.status_code == 422
