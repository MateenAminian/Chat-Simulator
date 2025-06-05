import os
from pydantic import BaseSettings, Field
from typing import List, Dict, Optional, Any

class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "Llama Chat"
    DEBUG: bool = Field(default=False, env="DEBUG")
    SECRET_KEY: str = Field(default="your-secret-key-for-dev", env="SECRET_KEY")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///llama_chat.db", env="DATABASE_URL")
    
    # Authentication
    JWT_SECRET_KEY: str = Field(default="your-jwt-secret-key", env="JWT_SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    CORS_ORIGINS: List[str] = Field(default=["http://localhost:3000", "https://llamachat.app"])
    
    # File storage
    UPLOAD_DIR: str = Field(default="static/uploads")
    OUTPUT_DIR: str = Field(default="static/outputs")
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100MB
    
    # Model settings
    VISION_MODEL: str = Field(default="Salesforce/blip-image-captioning-base")
    AUDIO_MODEL: str = Field(default="openai/whisper-base")
    OLLAMA_HOST: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    MODEL_CACHE_DIR: str = Field(default="/app/model_cache", env="MODEL_CACHE_DIR")
    
    # Caching
    CACHE_DIR: str = Field(default="cache")
    CACHE_TTL: int = 24 * 60 * 60  # 24 hours
    
    # Worker settings
    WORKER_CONCURRENCY: int = Field(default=3, env="WORKER_CONCURRENCY")
    
    # Monitoring
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        
settings = Settings() 