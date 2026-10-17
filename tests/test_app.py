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
    assert b"loading-overlay" in response.data


def test_prediction_post_pdf_authenticated(auth_client, app, sample_pdf_bytes):
    data = {
        "resume_file": (io.BytesIO(sample_pdf_bytes), "sample_resume.pdf"),
    }
    response = auth_client.post("/prediction", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert b"Predicted Resume Category" in response.data
    assert b"sample_resume.pdf" in response.data
    assert b"DevOps Engineer" in response.data

    # Verify report was stored with complete metadata
    with app.app_context():
        db = get_db()
        pred = db.query(Prediction).filter_by(original_filename="sample_resume.pdf").first()
        assert pred is not None
        assert pred.user_id is not None
        assert pred.probabilities_json is not None
        assert len(pred.probabilities) > 0
        assert pred.top_keywords_json is not None
        assert pred.text_length is not None


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


def test_prediction_post_empty_file(auth_client, app):
    # Verify failed request does not create a report record
    initial_count = 0
    with app.app_context():
        initial_count = get_db().query(Prediction).count()

    data = {
        "resume_file": (io.BytesIO(b""), "empty.pdf"),
    }
    response = auth_client.post("/prediction", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"empty" in response.data.lower()

    with app.app_context():
        assert get_db().query(Prediction).count() == initial_count


def test_prediction_post_invalid_extension(auth_client, app):
    initial_count = 0
    with app.app_context():
        initial_count = get_db().query(Prediction).count()

    data = {
        "resume_file": (io.BytesIO(b"Fake content"), "resume.txt"),
    }
    response = auth_client.post("/prediction", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"unsupported" in response.data.lower()

    with app.app_context():
        assert get_db().query(Prediction).count() == initial_count


def test_dashboard_authenticated_with_report(auth_client, app):
    # Add a prediction for this user
    with app.app_context():
        db = get_db()
        user = db.query(User).filter_by(email="candidate@test.com").first()
        pred = Prediction(
            user_id=user.id,
            original_filename="my_resume.pdf",
            predicted_category="Data Science",
            confidence=0.88,
            probabilities_json='[{"category": "Data Science", "probability": 0.88, "percentage": 88.0}]',
            top_keywords_json='["Python", "Machine Learning"]',
            text_length=450,
        )
        db.add(pred)
        db.commit()

    response = auth_client.get("/dashboard")
    assert response.status_code == 200
    assert b"Welcome back, Test Candidate" in response.data
    assert b"Total Completed Analyses" in response.data
    assert b"Latest Completed Analysis Report" in response.data
    assert b"my_resume.pdf" in response.data
    assert b"Data Science" in response.data
    assert b"View Full Report" in response.data


def test_dashboard_empty_state(auth_client):
    response = auth_client.get("/dashboard")
    assert response.status_code == 200
    assert b"No Analysis Reports Yet" in response.data
    assert b"Analyze Your First Resume" in response.data


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


def test_report_detail_authenticated(auth_client, app):
    with app.app_context():
        db = get_db()
        user = db.query(User).filter_by(email="candidate@test.com").first()
        pred = Prediction(
            user_id=user.id,
            original_filename="audit_report.pdf",
            predicted_category="DevOps Engineer",
            confidence=0.94,
            probabilities_json='[{"category": "DevOps Engineer", "probability": 0.94, "percentage": 94.0}]',
            top_keywords_json='["Docker", "Kubernetes", "AWS"]',
            text_length=620,
            extracted_text_preview="Experienced DevOps engineer with Docker and Kubernetes skills.",
        )
        db.add(pred)
        db.commit()
        pred_id = pred.id

    resp = auth_client.get(f"/reports/{pred_id}")
    assert resp.status_code == 200
    assert b"audit_report.pdf" in resp.data
    assert b"DevOps Engineer" in resp.data
    assert b"94.0%" in resp.data
    assert b"Docker" in resp.data
    assert b"Kubernetes" in resp.data
    assert b"Back to History" in resp.data


def test_report_detail_unauthenticated(client):
    resp = client.get("/reports/1")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]
    assert "next=" in resp.headers["Location"]


def test_report_detail_unauthorized_user(client, app):
    # User 1 owns report, User 2 tries to access
    with app.app_context():
        db = get_db()
        user1 = User(name="User1", email="u1@test.com")
        user1.set_password("pass123")
        user2 = User(name="User2", email="u2@test.com")
        user2.set_password("pass123")
        db.add_all([user1, user2])
        db.commit()

        pred = Prediction(
            user_id=user1.id,
            original_filename="private_doc.pdf",
            predicted_category="Security",
            confidence=0.9,
        )
        db.add(pred)
        db.commit()
        pred_id = pred.id
        u2_id = user2.id

    # Log in as user 2
    with client.session_transaction() as sess:
        sess["user_id"] = u2_id
        sess["user_name"] = "User2"

    resp = client.get(f"/reports/{pred_id}")
    assert resp.status_code == 403
    data = resp.get_json()
    assert "Forbidden" in data["error"]


def test_report_survives_db_reload(app):
    # Verify records in SQLite can be queried after session teardown
    with app.app_context():
        db = get_db()
        user = User(name="PersistenceTester", email="persist@test.com")
        user.set_password("pass1234")
        db.add(user)
        db.commit()

        pred = Prediction(
            user_id=user.id,
            original_filename="persisted_record.pdf",
            predicted_category="Java Developer",
            confidence=0.92,
            probabilities_json='[{"category": "Java Developer", "probability": 0.92, "percentage": 92.0}]',
            top_keywords_json='["Java", "Spring Boot"]',
            text_length=510,
        )
        db.add(pred)
        db.commit()
        pred_id = pred.id

    # New context simulating a subsequent request / restart
    with app.app_context():
        db2 = get_db()
        loaded_pred = db2.query(Prediction).filter_by(id=pred_id).first()
        assert loaded_pred is not None
        assert loaded_pred.original_filename == "persisted_record.pdf"
        assert loaded_pred.predicted_category == "Java Developer"
        assert loaded_pred.confidence == 0.92
        assert loaded_pred.top_keywords == ["Java", "Spring Boot"]
        assert loaded_pred.probabilities[0]["category"] == "Java Developer"
