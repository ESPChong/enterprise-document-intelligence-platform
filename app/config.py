"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with type validation."""
    
    # ---- Application Settings ----
    APP_NAME: str = "Enterprise Document Intelligence"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # ---- API Settings ----
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    
    # ---- Ollama Settings ----
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_LLM_MODEL: str = "llama3:8b-instruct-q4_0"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_TIMEOUT: int = 30
    
    # ---- Vector Database Settings ----
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "enterprise_documents"
    CHROMA_TELEMETRY: bool = False
    
    # ---- Document Processing ----
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_DOCUMENTS: int = 1000
    MAX_FILE_SIZE_MB: int = 50
    
    # ---- Monitoring ----
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "http://localhost:3000"
    LANGFUSE_ENVIRONMENT: str = "development"
    
    # ---- Security ----
    SECRET_KEY: str = "your-secret-key-change-this"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # ---- Logging ----
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    # ---- Performance ----
    MAX_WORKERS: int = 4
    REQUEST_TIMEOUT: int = 60
    RATE_LIMIT_PER_MINUTE: int = 60
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()