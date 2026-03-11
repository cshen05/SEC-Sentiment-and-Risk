

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """Application configuration for the SEC sentiment API."""

    project_root: Path = PROJECT_ROOT
    model_dir: Path = PROJECT_ROOT / "models" / "finbert_risk_classifier"
    model_name: str = "ProsusAI/finbert"
    max_length: int = int(os.getenv("MAX_LENGTH", "256"))
    api_title: str = "SEC Sentiment and Risk API"
    api_version: str = "1.0.0"
    db_host: str = os.getenv("DB_HOST", "localhost")
    db_port: int = int(os.getenv("DB_PORT", "5432"))
    db_name: str = os.getenv("DB_NAME", "sec_sentiment")
    db_user: str = os.getenv("DB_USER", "sec_user")
    db_password: str = os.getenv("DB_PASSWORD", "sec_password")


settings = Settings()