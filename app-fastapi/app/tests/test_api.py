def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"]
    assert payload["api_version"]
    assert payload["model_version"]
    # model_metadata should be present (may be null if metadata file doesn't exist)
    assert "model_metadata" in payload


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


def test_health_endpoint_with_metadata(client):
    """Test that health endpoint returns model metadata when available."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    
    # If metadata exists, verify its structure
    if payload["model_metadata"] is not None:
        metadata = payload["model_metadata"]
        assert "model_version" in metadata
        assert "trained_at" in metadata
        assert "git_sha" in metadata
        assert "metrics" in metadata
        assert "accuracy" in metadata["metrics"]
        assert "roc_auc" in metadata["metrics"]
        assert "n_rows" in metadata
        assert "n_features" in metadata
        assert "feature_names" in metadata
        assert isinstance(metadata["feature_names"], list)


def test_health_endpoint_without_metadata(client, monkeypatch):
    """Test that health endpoint returns 200 even when metadata is missing."""
    from src.processing import data_manager
    
    # Mock load_metadata to return None (simulating missing metadata file)
    monkeypatch.setattr(data_manager, "load_metadata", lambda: None)
    
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_metadata"] is None
