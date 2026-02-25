from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel, Field, ValidationError

import src

PACKAGE_ROOT = Path(src.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "data"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    target: str
    drop_features: List[str] = Field(default_factory=list)
    categorical_missing_with_constant: List[str] = Field(default_factory=list)
    categorical_missing_with_frequent: List[str] = Field(default_factory=list)
    numeric_features: List[str] = Field(default_factory=list)
    yeojohnson_features: List[str] = Field(default_factory=list)
    ordinal_features: List[str] = Field(default_factory=list)
    arbitrary_ordinal_features: List[str] = Field(default_factory=list)
    one_hot_features: List[str] = Field(default_factory=list)
    count_frequency_features: List[str] = Field(default_factory=list)
    experience_features: List[str] = Field(default_factory=list)
    experience_map: Dict[str, int] = Field(default_factory=dict)
    last_new_job_features: List[str] = Field(default_factory=list)
    last_new_job_map: Dict[str, int] = Field(default_factory=dict)
    company_size_features: List[str] = Field(default_factory=list)
    company_size_map: Dict[str, int] = Field(default_factory=dict)
    test_size: float = 0.2
    random_state: int = 42
    model_max_iter: int = 500
    class_weight: str | None = "balanced"


class Config(BaseModel):
    app_config: AppConfig
    ml_config: ModelConfig


def find_config_file() -> Path:
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise FileNotFoundError(f"Config file not found at {CONFIG_FILE_PATH}")


def fetch_config_from_yaml(cfg_path: Path | None = None) -> Dict:
    config_path = cfg_path or find_config_file()
    with config_path.open("r", encoding="utf-8") as conf_file:
        parsed_config = yaml.safe_load(conf_file)
    if not isinstance(parsed_config, dict):
        raise ValueError(f"Config file at {config_path} is not a valid YAML mapping.")
    return parsed_config


def create_and_validate_config(parsed_config: Dict | None = None) -> Config:
    raw_config = parsed_config or fetch_config_from_yaml()
    try:
        validated_config = Config(
            app_config=AppConfig(**raw_config),
            ml_config=ModelConfig(**raw_config),
        )
    except ValidationError as exc:
        raise ValueError("Configuration validation failed.") from exc
    return validated_config


config = create_and_validate_config()
