"""
ResumeAI - SQLAlchemy Data Models
=================================
Defines User and Prediction entities with password hashing and relationships.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from werkzeug.security import generate_password_hash, check_password_hash

from database.db import Base


class User(Base):
    """User account entity for authentication and prediction tracking."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # One-to-many relationship with predictions
    predictions = relationship(
        "Prediction",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="desc(Prediction.created_at)"
    )

    def set_password(self, password: str) -> None:
        """Hashes password with Werkzeug's default pbkdf2/scrypt algorithm."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Verifies supplied plaintext password against stored hash."""
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<User {self.email}>"


class Prediction(Base):
    """Record of a resume classification request."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    original_filename = Column(String(255), nullable=False)
    predicted_category = Column(String(100), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    extracted_text_preview = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Back reference to User
    user = relationship("User", back_populates="predictions")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "original_filename": self.original_filename,
            "predicted_category": self.predicted_category,
            "confidence": round(self.confidence, 4),
            "confidence_percentage": round(self.confidence * 100, 1),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<Prediction {self.id}: {self.predicted_category} ({self.confidence:.2f})>"
