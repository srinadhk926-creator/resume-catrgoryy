"""
Tests for main Flask application routes, web UI, database persistence, and authorization.
"""

import io
import pytest
from database.db import get_db
from database.models import User, Prediction


def test_landing_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"ResumeAI" in response.data
    assert b"Resume Category Classification" in response.data


def test_prediction_guest_redirect(client):
    resp_get = client.get("/prediction")
    assert resp_get.status_code == 302
    assert "/login" in resp_get.headers["Location"]
    assert "next=" in resp_get.headers["Location"]

    resp_post = client.post("/prediction", data={"resume_text": "Sample text here"})
    assert resp_post.status_code == 302
    assert "/login" in resp_post.headers["Location"]


def test_prediction_authenticated_get(auth_client):
    response = auth_client.get("/prediction")
    assert response.status_code == 200
    assert b"Classify Candidate Resume" in response.data
    assert b"drop-zone" in response.data


def test_prediction_post_pdf_authenticated(auth_client, sample_pdf_bytes):
    data = {
        "resume_file": (io.BytesIO(sample_pdf_bytes), "sample_resume.pdf"),
    }
    response = auth_client.post("/prediction", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Predicted Resume Category" in response.data
    assert b"sample_resume.pdf" in response.data
    assert b"DevOps Engineer" in response.data


def test_prediction_post_docx_authenticated(auth_client, app, sample_docx_bytes):
    data = {
        "resume_file": (io.BytesIO(sample_docx_bytes), "python_dev.docx"),
    }
    response = auth_client.post("/prediction", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Saved to Your History" in response.data

    # Verify record was stored in database
    with app.app_context():
        db = get_db()
        pred = db.query(Prediction).filter_by(original_filename="python_dev.docx").first()
        assert pred is not None
        assert pred.user_id is not None
        assert pred.confidence > 0.0


def test_prediction_post_empty_file(auth_client):
    data = {
        "resume_file": (io.BytesIO(b""), "empty.pdf"),
    }
    response = auth_client.post("/prediction", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"empty" in response.data.lower()


def test_prediction_post_invalid_extension(auth_client):
    data = {
        "resume_file": (io.BytesIO(b"Fake content"), "resume.txt"),
    }
    response = auth_client.post("/prediction", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"unsupported" in response.data.lower()


def test_dashboard_authenticated(auth_client, app):
    # Add a prediction for this user
    with app.app_context():
        db = get_db()
        user = db.query(User).filter_by(email="candidate@test.com").first()
        pred = Prediction(
            user_id=user.id,
            original_filename="my_resume.pdf",
            predicted_category="Data Science",
            confidence=0.88,
        )
        db.add(pred)
        db.commit()

    response = auth_client.get("/dashboard")
    assert response.status_code == 200
    assert b"Welcome back, Test Candidate" in response.data
    assert b"Total Resumes Analyzed" in response.data
    assert b"Data Science" in response.data


def test_history_page_and_delete(auth_client, app):
    # Create prediction
    with app.app_context():
        db = get_db()
        user = db.query(User).filter_by(email="candidate@test.com").first()
        pred = Prediction(
            user_id=user.id,
            original_filename="to_delete.pdf",
            predicted_category="Java Developer",
            confidence=0.91,
        )
        db.add(pred)
        db.commit()
        pred_id = pred.id

    # View history
    history_resp = auth_client.get("/history")
    assert history_resp.status_code == 200
    assert b"to_delete.pdf" in history_resp.data
    assert b"Java Developer" in history_resp.data

    # Delete prediction
    del_resp = auth_client.post(f"/history/delete/{pred_id}", follow_redirects=True)
    assert del_resp.status_code == 200
    assert b"removed from your history" in del_resp.data

    with app.app_context():
        db = get_db()
        deleted = db.query(Prediction).filter_by(id=pred_id).first()
        assert deleted is None


def test_history_delete_unauthorized_user(client, app):
    # Create two users
    with app.app_context():
        db = get_db()
        owner = User(name="Owner", email="owner@test.com")
        owner.set_password("pass12345")
        other = User(name="Attacker", email="attacker@test.com")
        other.set_password("pass12345")
        db.add_all([owner, other])
        db.commit()

        pred = Prediction(
            user_id=owner.id,
            original_filename="secret_resume.pdf",
            predicted_category="HR",
            confidence=0.85,
        )
        db.add(pred)
        db.commit()
        pred_id = pred.id
        attacker_id = other.id

    # Log in as attacker
    with client.session_transaction() as sess:
        sess["user_id"] = attacker_id
        sess["user_name"] = "Attacker"

    # Attempt to delete owner's prediction
    del_resp = client.post(f"/history/delete/{pred_id}")
    assert del_resp.status_code == 403
    data = del_resp.get_json()
    assert "Forbidden" in data["error"]


def test_404_error_page(client):
    response = client.get("/non-existent-page-url")
    assert response.status_code == 404
    assert b"Page Not Found" in response.data or b"404" in response.data


def test_profile_authenticated(auth_client):
    response = auth_client.get("/profile")
    assert response.status_code == 200
    assert b"Account Profile" in response.data
    assert b"candidate@test.com" in response.data


def test_profile_unauthenticated(client):
    response = client.get("/profile")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]
    assert "next=" in response.headers["Location"]


def test_landing_page_logged_in(auth_client):
    response = auth_client.get("/")
    assert response.status_code == 200
    assert b"Dashboard" in response.data
    assert b"Sign Out" in response.data
