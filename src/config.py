from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):

    APP_NAME: str = "Multi-Task NLP API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "multitask_nlp_experiment"

    MODEL_NAME: str = "distilbert-base-uncased"
    MAX_SEQ_LENGTH: int = 128
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 2e-5
    EPOCHS: int = 3

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8")

@lru_cache()
def get_settings():
    return Settings()
