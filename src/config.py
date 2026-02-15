"""
Production Configuration Management
Handles environment variables and production settings
"""
import os
from typing import Optional, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Project root (parent of src/)
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load .env file from project root
load_dotenv(os.path.join(_BASE_DIR, '.env'))

def _model_path(env_key: str, default_filename: str) -> str:
    """Resolve model path: use env if set, else project_root/models/filename."""
    val = os.getenv(env_key)
    if val:
        return val
    return os.path.join(_BASE_DIR, 'models', default_filename)

@dataclass
class Config:
    """Application configuration"""
    
    # API Settings
    HOST: str = os.getenv('API_HOST', '0.0.0.0')
    PORT: int = int(os.getenv('API_PORT', '5000'))
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # CORS Settings
    CORS_ORIGINS: str = os.getenv('CORS_ORIGINS', '*')
    
    # Model Settings (paths resolved from project root so app works from any cwd)
    MODEL_PATH: str = field(default_factory=lambda: _model_path('MODEL_PATH', 'best_model_random_forest.pkl'))
    PREPROCESSOR_PATH: str = field(default_factory=lambda: _model_path('PREPROCESSOR_PATH', 'preprocessor.pkl'))
    
    # LLM Settings
    GROQ_API_KEY: Optional[str] = os.getenv('GROQ_API_KEY')
    LLM_MODEL: str = os.getenv('LLM_MODEL', 'llama-3.3-70b-versatile')
    LLM_ENABLED: bool = field(default_factory=lambda: 
        os.getenv('LLM_ENABLED', 'true' if os.getenv('GROQ_API_KEY') else 'false').lower() == 'true')
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = os.getenv('RATE_LIMIT_ENABLED', 'True').lower() == 'true'
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.getenv('LOG_FILE', 'logs/api.log')
    
    # Risk scoring thresholds (0–100)
    RISK_LOW_THRESHOLD: float = float(os.getenv('RISK_LOW_THRESHOLD', '30'))
    RISK_MEDIUM_THRESHOLD: float = float(os.getenv('RISK_MEDIUM_THRESHOLD', '70'))

    # Security
    API_KEY_REQUIRED: bool = os.getenv('API_KEY_REQUIRED', 'False').lower() == 'true'
    API_KEYS: List[str] = field(default_factory=lambda: os.getenv('API_KEYS', '').split(',') if os.getenv('API_KEYS') else [])
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        config = cls()
        errors = []
        
        # Check if model files exist
        if not os.path.exists(config.MODEL_PATH):
            errors.append(f"Model file not found: {config.MODEL_PATH}")
        
        if not os.path.exists(config.PREPROCESSOR_PATH):
            errors.append(f"Preprocessor file not found: {config.PREPROCESSOR_PATH}")
        
        # Check LLM config
        if config.LLM_ENABLED and not config.GROQ_API_KEY:
            errors.append("LLM_ENABLED is True but GROQ_API_KEY is not set")
        
        if errors:
            raise ValueError(f"Configuration errors:\\n" + "\\n".join(errors))
        
        return config

# Global config instance
config = Config()
