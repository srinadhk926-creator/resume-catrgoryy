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


import json


class Prediction(Base):
    """Record of a resume classification request."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    original_filename = Column(String(255), nullable=False)
    predicted_category = Column(String(100), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    extracted_text_preview = Column(Text, nullable=True)
    probabilities_json = Column(Text, nullable=True)
    top_keywords_json = Column(Text, nullable=True)
    text_length = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Back reference to User
    user = relationship("User", back_populates="predictions")

    @property
    def confidence_percentage(self) -> float:
        """Returns confidence as a rounded percentage."""
        return round(float(self.confidence) * 100, 1)

    @property
    def probabilities(self) -> list:
        """Safely parses stored JSON probabilities or returns empty list."""
        if not self.probabilities_json:
            return [{
                "category": self.predicted_category,
                "probability": round(float(self.confidence), 4),
                "percentage": self.confidence_percentage,
            }]
        try:
            return json.loads(self.probabilities_json)
        except Exception:
            return [{
                "category": self.predicted_category,
                "probability": round(float(self.confidence), 4),
                "percentage": self.confidence_percentage,
            }]

    @property
    def top_keywords(self) -> list:
        """Safely parses stored JSON keywords or returns empty list."""
        if not self.top_keywords_json:
            return []
        try:
            return json.loads(self.top_keywords_json)
        except Exception:
            return []

    @property
    def margin(self) -> float:
        """Calculates margin between 1st and 2nd category probabilities."""
        probs = self.probabilities
        if len(probs) >= 2:
            return round(float(probs[0].get("probability", 0.0) - probs[1].get("probability", 0.0)), 4)
        return round(float(self.confidence), 4)

    @property
    def is_low_confidence(self) -> bool:
        """Flags whether the prediction has low confidence or narrow runner-up margin."""
        return bool(self.confidence < 0.22 or self.margin < 0.04)

    @property
    def confidence_warning(self) -> str | None:
        """Human-readable ambiguity warning when confidence is low."""
        if self.is_low_confidence:
            return (
                "Low confidence prediction. This resume contains multi-disciplinary skills across "
                "multiple domains or limited domain-specific keywords. Review alternative category probabilities."
            )
        return None

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "original_filename": self.original_filename,
            "predicted_category": self.predicted_category,
            "confidence": round(float(self.confidence), 4),
            "confidence_percentage": self.confidence_percentage,
            "margin": self.margin,
            "is_low_confidence": self.is_low_confidence,
            "confidence_warning": self.confidence_warning,
            "probabilities": self.probabilities,
            "top_keywords": self.top_keywords,
            "text_length": self.text_length or (len(self.extracted_text_preview) if self.extracted_text_preview else 0),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return f"<Prediction {self.id}: {self.predicted_category} ({self.confidence:.2f})>"
