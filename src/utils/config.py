"""
Configuration management for the application
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # Model paths
    SENTIMENT_MODEL_PATH: str = Field(
        default="models/sentiment_model",
        description="Path to Persian sentiment model"
    )
    DIALECT_MODEL_PATH: str = Field(
        default="models/dialect_model.joblib",
        description="Path to dialect detection model"
    )
    MULTILINGUAL_MODEL_PATH: str = Field(
        default="models/multilingual_model",
        description="Path to multilingual sentiment model"
    )

    # Model configurations
    PERSIAN_MODEL_NAME: str = Field(
        default="HooshvareLab/bert-fa-base-uncased",
        description="Pretrained model for Persian"
    )
    MULTILINGUAL_MODEL_NAME: str = Field(
        default="xlm-roberta-base",
        description="Pretrained multilingual model"
    )
    NUM_LABELS: int = Field(default=3, description="Number of sentiment labels")
    MAX_LENGTH: int = Field(default=128, description="Maximum sequence length")

    # Training configurations
    BATCH_SIZE: int = Field(default=16, description="Training batch size")
    LEARNING_RATE: float = Field(default=2e-5, description="Learning rate")
    NUM_EPOCHS: int = Field(default=3, description="Number of training epochs")
    WARMUP_STEPS: int = Field(default=500, description="Warmup steps")
    WEIGHT_DECAY: float = Field(default=0.01, description="Weight decay")
    TEST_SIZE: float = Field(default=0.2, description="Test set size")
    RANDOM_STATE: int = Field(default=42, description="Random seed")

    # API configurations
    API_HOST: str = Field(default="0.0.0.0", description="API host")
    API_PORT: int = Field(default=8000, description="API port")
    API_RELOAD: bool = Field(default=False, description="Enable auto-reload")
    API_WORKERS: int = Field(default=1, description="Number of workers")
    CORS_ORIGINS: list = Field(
        default=["*"],
        description="Allowed CORS origins"
    )

    # Logging configurations
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    LOG_FILE: Optional[str] = Field(
        default="logs/app.log",
        description="Log file path"
    )

    # Streamlit configurations
    STREAMLIT_PORT: int = Field(default=8501, description="Streamlit port")

    # Performance configurations
    USE_GPU: bool = Field(default=True, description="Use GPU if available")
    MAX_BATCH_SIZE: int = Field(
        default=100,
        description="Maximum batch size for API requests"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def ensure_directories():
    """Ensure required directories exist"""
    for directory in [
        settings.DATA_DIR,
        settings.MODELS_DIR,
        settings.LOGS_DIR
    ]:
        directory.mkdir(parents=True, exist_ok=True)
