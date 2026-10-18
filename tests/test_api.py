"""
Tests for REST API and /health endpoint.
"""

import io
import pytest


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["service"] == "resume-category-classifier"
    assert data["model_loaded"] is True
    assert "timestamp" in data


def test_api_predict_json_text_success(client):
    payload = {
        "text": "Experienced Python Backend Developer with 5 years building REST APIs, Flask, Django, PostgreSQL, and Docker."
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "predicted_category" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "is_low_confidence" in data
    assert "margin" in data
    assert "confidence_warning" in data
    assert isinstance(data["top_keywords"], list)
    assert isinstance(data["probabilities"], list)
    assert len(data["probabilities"]) > 0


def test_api_predict_pdf_upload_success(client, sample_pdf_bytes):
    data = {
        "file": (io.BytesIO(sample_pdf_bytes), "devops_resume.pdf"),
    }
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res = response.get_json()
    assert res["status"] == "success"
    assert res["filename"] == "devops_resume.pdf"
    assert res["predicted_category"] == "DevOps Engineer"
    assert "confidence" in res
    assert "Kubernetes" in res["top_keywords"] or "Docker" in res["top_keywords"]


def test_api_predict_docx_upload_success(client, sample_docx_bytes):
    data = {
        "file": (io.BytesIO(sample_docx_bytes), "python_developer_resume.docx"),
    }
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res = response.get_json()
    assert res["status"] == "success"
    assert res["filename"] == "python_developer_resume.docx"
    assert "Python Developer" in [res["predicted_category"], res["probabilities"][0]["category"]]


def test_api_predict_empty_payload(client):
    response = client.post("/api/predict", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"
    assert "Missing 'text' field" in data["message"]


def test_api_predict_unsupported_file_extension(client):
    data = {
        "file": (io.BytesIO(b"Fake content"), "malicious_script.sh"),
    }
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 415
    data = response.get_json()
    assert data["status"] == "error"
    assert "unsupported" in data["message"].lower()


def test_api_predict_empty_file(client):
    data = {
        "file": (io.BytesIO(b""), "empty.pdf"),
    }
    response = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"


def test_api_predict_text_too_short(client):
    response = client.post("/api/predict", json={"text": "Short"})
    assert response.status_code == 400
    data = response.get_json()
    assert "too short" in data["message"].lower()


def test_api_predict_missing_model_returns_503(client, app, monkeypatch):
    from services.prediction_service import PredictionService
    monkeypatch.setattr(app.config, "get", lambda k, default=None: "non_existent_model.joblib" if k == "MODEL_PATH" else default)
    
    # Force service to point to nonexistent file
    from services import prediction_service
    prediction_service._prediction_service_instance = PredictionService("non_existent_model.joblib")

    response = client.post("/api/predict", json={"text": "Valid long text for testing model error handling."})
    assert response.status_code == 503
    data = response.get_json()
    assert data["status"] == "error"
    assert "unavailable" in data["message"].lower() or "missing" in data["message"].lower()

    # Reset singleton
    prediction_service._prediction_service_instance = None
