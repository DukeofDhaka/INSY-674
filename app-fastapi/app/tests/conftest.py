from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[3]
API_ROOT = PROJECT_ROOT / "app-fastapi"

if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402
from src.processing.data_manager import model_artifact_path  # noqa: E402
from src.train_pipeline import run_training  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def ensure_trained_model() -> Path:
    artifact = model_artifact_path()
    if not artifact.exists():
        run_training()
    return artifact


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)
