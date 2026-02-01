"""
Production Configuration Management
Handles environment variables and production settings
"""
import os
from typing import Optional, List
from dataclasses import dataclass, field

@dataclass
class Config:
    """Application configuration"""
    
    # API Settings
    HOST: str = os.getenv('API_HOST', '0.0.0.0')
    PORT: int = int(os.getenv('API_PORT', '5000'))
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # CORS Settings
    CORS_ORIGINS: str = os.getenv('CORS_ORIGINS', '*')
    
    # Model Settings
    MODEL_PATH: str = os.getenv('MODEL_PATH', '../models/best_model_random_forest.pkl')
    PREPROCESSOR_PATH: str = os.getenv('PREPROCESSOR_PATH', '../models/preprocessor.pkl')
    
    # LLM Settings
    GROQ_API_KEY: Optional[str] = os.getenv('GROQ_API_KEY')
    LLM_MODEL: str = os.getenv('LLM_MODEL', 'llama-3.3-70b-versatile')
    LLM_ENABLED: bool = os.getenv('LLM_ENABLED', 'False').lower() == 'true'
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = os.getenv('RATE_LIMIT_ENABLED', 'True').lower() == 'true'
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv('RATE_LIMIT_PER_MINUTE', '60'))
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE: str = os.getenv('LOG_FILE', 'logs/api.log')
    
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
