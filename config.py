import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INSTANCE_DIR = BASE_DIR / "instance"
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

class Config:
    """Base application configuration."""
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-fallback-secret-key-replace-in-production")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", 
        f"sqlite:///{(INSTANCE_DIR / 'resume_ai.db').as_posix()}"
    )
    
    # Model Artifact
    MODEL_PATH = os.getenv(
        "MODEL_PATH", 
        str((BASE_DIR / "model" / "resume_classifier.joblib").as_posix())
    )
    
    # Upload limits (5MB default)
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 5 * 1024 * 1024))
    ALLOWED_EXTENSIONS = {"pdf", "docx"}
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", str((BASE_DIR / "uploads").as_posix()))
    
    # Security & session flags
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    TESTING = False


class TestingConfig(Config):
    """Testing environment configuration."""
    DEBUG = False
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    WTF_CSRF_ENABLED = False
    MODEL_PATH = str((BASE_DIR / "model" / "resume_classifier.joblib").as_posix())


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    TESTING = False
    SESSION_COOKIE_SECURE = False  # Set to True if TLS/HTTPS termination is enabled


config_by_name = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
