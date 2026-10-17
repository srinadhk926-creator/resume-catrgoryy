"""
Pytest Fixtures for ResumeAI Application Testing
"""

import io
import os
from pathlib import Path
import pytest
import fitz  # PyMuPDF
import docx

from app import create_app
from database.db import init_db, get_db, close_db_session
from database.models import User, Prediction
from services.prediction_service import get_prediction_service


@pytest.fixture
def app(tmp_path):
    """Creates a configured test application with in-memory database."""
    test_db_url = f"sqlite:///{(tmp_path / 'test_app.db').as_posix()}"
    model_path = str((Path(__file__).resolve().parent.parent / "model" / "resume_classifier.joblib").as_posix())

    # If the default model does not exist, use test model or trained model
    os.environ["DATABASE_URL"] = test_db_url
    os.environ["MODEL_PATH"] = model_path
    os.environ["FLASK_ENV"] = "testing"

    app = create_app("testing")
    app.config.update({
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,
        "SQLALCHEMY_DATABASE_URI": test_db_url,
        "MODEL_PATH": model_path,
    })

    with app.app_context():
        init_db(test_db_url, create_tables=True)
        yield app
        close_db_session()


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def auth_client(client, app):
    """Test client logged in as a test user."""
    with app.app_context():
        db = get_db()
        user = User(name="Test Candidate", email="candidate@test.com")
        user.set_password("securepassword123")
        db.add(user)
        db.commit()
        user_id = user.id

    with client.session_transaction() as sess:
        sess["user_id"] = user_id
        sess["user_name"] = "Test Candidate"
        sess["user_email"] = "candidate@test.com"

    return client


@pytest.fixture
def sample_pdf_bytes():
    """Generates a valid in-memory PDF containing technical resume text."""
    doc = fitz.open()
    page = doc.new_page()
    text = (
        "DevOps Engineer and Cloud Architect with 5 years experience in Docker, "
        "Kubernetes, AWS EC2, S3, Terraform, Jenkins, and CI/CD automation pipelines. "
        "Proficient in Linux, Prometheus monitoring, and infrastructure reliability."
    )
    page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


@pytest.fixture
def sample_docx_bytes():
    """Generates a valid in-memory DOCX containing technical resume text."""
    doc = docx.Document()
    doc.add_heading("Senior Python Developer", level=1)
    doc.add_paragraph(
        "Backend Python Developer with 4 years experience designing REST APIs with Flask and Django. "
        "Proficient in PostgreSQL, Redis, Celery worker queues, Docker containers, and automated pytest testing."
    )
    bio = io.BytesIO()
    doc.save(bio)
    return bio.getvalue()


@pytest.fixture
def empty_pdf_bytes():
    """Generates a valid PDF with zero text content."""
    doc = fitz.open()
    doc.new_page()  # Blank page
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes
