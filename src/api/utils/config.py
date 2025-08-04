"""
API configuration management.
"""
import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class APIConfig(BaseSettings):
    """API configuration settings."""
    
    # Basic settings
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(default="development", description="Environment (development, production)")
    
    # Server settings
    host: str = Field(default="127.0.0.1", description="API host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=1, description="Number of worker processes")
    
    # Security settings
    secret_key: str = Field(default="your-secret-key-change-this", description="Secret key for JWT tokens")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration in minutes")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins"
    )
    
    # Database settings
    database_url: str = Field(
        default="sqlite+aiosqlite:///./agents.db",
        description="Database URL"
    )
    
    # Redis settings (for caching and sessions)
    redis_url: Optional[str] = Field(default=None, description="Redis URL for caching")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=100, description="Rate limit per minute per IP")
    
    # File upload settings
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Maximum file size in bytes (10MB)")
    upload_dir: str = Field(default="./uploads", description="Upload directory")
    
    # Agent settings
    max_concurrent_agents: int = Field(default=10, description="Maximum concurrent agent instances")
    agent_timeout_seconds: int = Field(default=300, description="Agent execution timeout")
    
    # LLM API Keys (inherited from environment)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, env="GROQ_API_KEY")
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"
    
    def get_database_url(self) -> str:
        """Get database URL with proper configuration."""
        if self.is_production() and not self.database_url.startswith("postgresql"):
            # In production, prefer PostgreSQL
            return os.getenv("DATABASE_URL", self.database_url)
        return self.database_url
    
    def get_cors_origins(self) -> List[str]:
        """Get CORS origins based on environment."""
        if self.is_production():
            # In production, use environment variable or secure defaults
            return os.getenv("CORS_ORIGINS", "").split(",") or ["https://yourdomain.com"]
        return self.cors_origins


# Global config instance
config = APIConfig()