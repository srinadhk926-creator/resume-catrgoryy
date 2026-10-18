"""
ResumeAI - Prediction Service
=============================
Handles model artifact loading, input text validation, category inference,
confidence score calculation, and skill/keyword extraction.
"""

import os
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import joblib

from services.resume_parser import redact_pii

logger = logging.getLogger("ResumeAI-PredictionService")

DEFAULT_MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "model" / "resume_classifier.joblib")
)

MIN_TEXT_LENGTH = 20
MAX_TEXT_LENGTH = 100_000

# High-value technical and professional skills domain vocabulary for keyword extraction
DOMAIN_SKILLS = {
    # Data Science & AI
    "python", "machine learning", "deep learning", "nlp", "tensorflow", "pytorch",
    "scikit-learn", "pandas", "numpy", "sql", "tableau", "powerbi", "r", "keras",
    "bigquery", "data analysis", "feature engineering", "transformers", "bert",
    # Python & Backend
    "django", "flask", "fastapi", "celery", "redis", "postgresql", "mysql",
    "sqlite", "rest api", "microservices", "asyncio", "sqlalchemy", "jwt",
    # Java & Enterprise
    "java", "spring boot", "spring mvc", "hibernate", "kafka", "maven",
    "junit", "j2ee", "jpa", "microservices", "multithreading", "oracle",
    # DevOps & Cloud
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible",
    "jenkins", "ci/cd", "linux", "bash", "prometheus", "grafana", "helm",
    # Web & UI/UX
    "html5", "css3", "javascript", "react", "vue", "tailwind", "bootstrap",
    "figma", "typescript", "ui/ux", "sass", "responsive design", "node.js",
    # Software Engineering
    "git", "data structures", "algorithms", "c++", "c#", "system design",
    "oop", "agile", "scrum", "rest", "unit testing",
    # HR & Recruiting
    "recruitment", "talent acquisition", "onboarding", "hris", "payroll",
    "employee relations", "compliance", "interviewing", "workday",
    # Business Analysis
    "jira", "confluence", "requirements gathering", "user stories", "gap analysis",
    "business analysis", "sdlc", "uat", "bpmn", "stakeholder management",
    # Database Administration
    "database administrator", "dba", "indexing", "sharding", "replication",
    "backup", "disaster recovery", "performance tuning", "pgbouncer",
    # Security
    "cybersecurity", "firewall", "ids/ips", "siem", "splunk", "wireshark",
    "penetration testing", "vulnerability assessment", "cissp", "soc", "vpn"
}


class PredictionService:
    """Service to load fitted pipeline and classify resume texts."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self._model = None

    def is_model_available(self) -> bool:
        """Returns True if the serialized model file exists on disk."""
        return self.model_path.is_file()

    def load_model(self, force_reload: bool = False):
        """Loads or reloads the serialized scikit-learn pipeline from disk."""
        if self._model is not None and not force_reload:
            return self._model

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Trained model artifact not found at '{self.model_path.resolve()}'. "
                f"Please train the model first by running: python train_model.py"
            )

        try:
            logger.info(f"Loading model pipeline from: {self.model_path}")
            self._model = joblib.load(self.model_path)
            return self._model
        except Exception as err:
            logger.error(f"Failed to load model artifact: {err}")
            raise RuntimeError(f"Error deserializing model from {self.model_path}: {err}") from err

    @staticmethod
    def clean_text(text: str) -> str:
        """Sanitizes text by removing control characters and collapsing whitespace."""
        if not text:
            return ""
        # Remove non-printable / control characters except newlines
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
        # Collapse multiple spaces and newlines
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def validate_text(self, text: Any) -> str:
        """Validates input resume text, raising descriptive ValueErrors if invalid."""
        if text is None or not isinstance(text, str):
            raise ValueError("Input resume text must be a non-null string.")

        cleaned = self.clean_text(text)
        if len(cleaned) == 0:
            raise ValueError("Resume text cannot be empty or contain only whitespace.")

        if len(cleaned) < MIN_TEXT_LENGTH:
            raise ValueError(
                f"Resume text is too short to classify ({len(cleaned)} chars). "
                f"A minimum of {MIN_TEXT_LENGTH} characters is required."
            )

        if len(cleaned) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Resume text exceeds maximum supported size ({len(cleaned)} chars). "
                f"Maximum allowed is {MAX_TEXT_LENGTH} characters."
            )

        return cleaned

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extracts significant keywords and recognized technical skills from the text."""
        lower_text = text.lower()
        extracted: List[str] = []

        # 1. Match known domain skills
        for skill in DOMAIN_SKILLS:
            # Word boundary match
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, lower_text):
                extracted.append(skill.title())

        # 2. If model has fitted TF-IDF vectorizer, extract top-weighted terms
        if self._model and hasattr(self._model, "named_steps") and "tfidf" in self._model.named_steps:
            try:
                tfidf = self._model.named_steps["tfidf"]
                response = tfidf.transform([text])
                feature_names = np.array(tfidf.get_feature_names_out())
                row = response.tocoo()
                sorted_items = sorted(zip(row.col, row.data), key=lambda x: (x[1], x[0]), reverse=True)
                for idx, _ in sorted_items[:top_n]:
                    term = feature_names[idx].title()
                    if term not in extracted:
                        extracted.append(term)
            except Exception as e:
                logger.debug(f"TF-IDF keyword extraction skipped: {e}")

        # Return unique keywords up to top_n + 5
        seen = set()
        deduped = []
        for kw in extracted:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                deduped.append(kw)
        return deduped[:15]

    def predict(self, text: str) -> Dict[str, Any]:
        """Runs end-to-end inference on raw resume text."""
        cleaned_text = self.validate_text(text)
        model = self.load_model()

        predicted_category = str(model.predict([cleaned_text])[0])

        probabilities_list = []
        confidence = 0.0

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([cleaned_text])[0]
            classes = model.classes_
            sorted_indices = np.argsort(probs)[::-1]

            confidence = round(float(np.max(probs)), 4)
            for idx in sorted_indices:
                cat_name = str(classes[idx])
                prob_val = round(float(probs[idx]), 4)
                probabilities_list.append({
                    "category": cat_name,
                    "probability": prob_val,
                    "percentage": round(prob_val * 100, 1),
                })
        else:
            confidence = 1.0
            probabilities_list.append({
                "category": predicted_category,
                "probability": 1.0,
                "percentage": 100.0,
            })

        keywords = self.extract_keywords(cleaned_text)

        # Calculate runner-up margin and confidence quality
        margin = confidence
        if len(probabilities_list) >= 2:
            margin = round(float(probabilities_list[0]["probability"] - probabilities_list[1]["probability"]), 4)

        # In a 10-class problem (uniform chance = 10%), confidence < 22% or margin < 4% indicates high ambiguity
        is_low_confidence = bool(confidence < 0.22 or margin < 0.04)
        confidence_warning = (
            "Low confidence prediction. The resume text has multi-disciplinary skills across multiple domains "
            "or limited domain-specific keywords. Please review alternative category probabilities below."
            if is_low_confidence else None
        )

        return {
            "predicted_category": predicted_category,
            "confidence": confidence,
            "confidence_percentage": round(confidence * 100, 1),
            "is_low_confidence": is_low_confidence,
            "confidence_warning": confidence_warning,
            "margin": margin,
            "margin_percentage": round(margin * 100, 1),
            "probabilities": probabilities_list,
            "top_keywords": keywords,
            "text_length": len(cleaned_text),
            "text_preview": redact_pii(cleaned_text, max_chars=400),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }


# Global singleton instance for use across the application
_prediction_service_instance: Optional[PredictionService] = None


def get_prediction_service(model_path: Optional[str] = None) -> PredictionService:
    """Returns singleton instance of PredictionService."""
    global _prediction_service_instance
    if _prediction_service_instance is None or model_path is not None:
        _prediction_service_instance = PredictionService(model_path=model_path)
    return _prediction_service_instance
