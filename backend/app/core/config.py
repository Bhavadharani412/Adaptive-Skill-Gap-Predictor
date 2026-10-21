from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    APP_NAME: str = "Adaptive Skill Gap Predictor (ASGP)"
    APP_ENV: str = "development"
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    DEBUG: bool = True

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    DATA_SYNTHETIC_DIR: Path = BASE_DIR / "data" / "synthetic"
    MODELS_DIR: Path = BASE_DIR / "models"
    REPORTS_DIR: Path = BASE_DIR / "reports"
    EXPERIMENTS_DIR: Path = BASE_DIR / "experiments"

    # Database
    DATABASE_URL: str = f"sqlite:///{(BASE_DIR / 'data' / 'asgp.db').as_posix()}"

    # ML & Simulation Baseline Parameters (matching paper)
    DEFAULT_RANDOM_SEED: int = 42
    NUM_OCCUPATIONS: int = 40
    NUM_SKILLS: int = 220
    NUM_CLUSTERS: int = 8
    GENERIC_SKILL_RATIO: float = 0.12
    NUM_EMPLOYEES: int = 600
    READINESS_TARGET: float = 0.80
    DEFAULT_TRAINING_BUDGET: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
