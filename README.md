# ResumeAI — Resume Category Classification System

**Subject Code**: 2CEAI702 — MLOps  
**System Architecture**: Machine Learning Pipeline + Flask REST API + Docker + AWS EC2 + Google Cloud Platform

---

## 1. Project Overview & Objectives

**ResumeAI** is an end-to-end Machine Learning Operations (MLOps) platform that automatically processes resumes in PDF or DOCX format, cleans and extracts raw text, and classifies the candidate into appropriate professional domains (e.g., *Data Science*, *DevOps Engineer*, *Python Developer*, *Web Designing*, *Java Developer*, etc.) using TF-IDF vectorization and a scikit-learn classifier pipeline.

### Core MLOps Practicals Covered:
1. **Practical 1**: Transforming ML notebook workflow into a modular Flask application with input validation, SQLAlchemy database, user auth, and unit tests.
2. **Practical 2**: Containerizing the ML application with Docker, multi-stage lightweight Python 3.11 base image, and Gunicorn.
3. **Practical 3**: Production deployment on an Amazon Web Services (AWS) EC2 Ubuntu instance with security groups.
4. **Practical 4**: Production deployment on a Google Cloud Platform (GCP) Compute Engine VM with VPC firewall ingress rules.

---

## 2. Technology Stack

- **Machine Learning**: `scikit-learn`, `pandas`, `numpy`, `joblib`, TF-IDF Vectorizer, Logistic Regression.
- **Backend & API**: Python 3.10 / 3.11, `Flask`, `Flask-WTF` (CSRF), `Werkzeug` (Password Hashing).
- **Database**: SQLite (local development / testing), `SQLAlchemy` ORM (extensible to PostgreSQL).
- **Document Processing**: `PyMuPDF` (fitz) for PDF files, `python-docx` for Word DOCX files.
- **Frontend**: Modern responsive HTML5, CSS3, Vanilla JavaScript, Jinja2 Templates.
- **DevOps & Containers**: `Docker`, `docker-compose`, `Gunicorn` WSGI server (`0.0.0.0:5000`).
- **Cloud Infrastructure**: AWS EC2 (Ubuntu 24.04/22.04 LTS), GCP Compute Engine (Ubuntu LTS).
- **Testing**: `pytest`, `pytest-cov`, Flask test client.

---

## 3. Project Directory Structure

```
D:\MLOPS/
├── app.py                      # Flask factory & application entrypoint
├── config.py                   # Environment configuration classes
├── train_model.py              # ML training, validation, & serialization script
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Development and testing dependencies
├── Dockerfile                  # Container build instructions
├── docker-compose.yml          # Container orchestration
├── .dockerignore               # Container build exclusions
├── .gitignore                  # Git exclusions (credentials, databases, venvs)
├── .env.example                # Example environment variables
├── README.md                   # Complete system documentation
├── PRACTICAL_EVIDENCE.md       # Lab submission guide & screenshot checklist
├── data/
│   ├── README.md               # Dataset documentation & Kaggle download guide
│   └── sample_resumes.csv      # Seed dataset for smoke testing
├── model/
│   └── resume_classifier.joblib# Serialized TF-IDF + Logistic Regression pipeline
├── services/
│   ├── resume_parser.py        # Safe PDF & DOCX text extraction
│   └── prediction_service.py   # Model inference, probability scores & keyword extraction
├── routes/
│   ├── auth.py                 # Registration, login, logout, user sessions
│   ├── main.py                 # UI routes: Landing, Dashboard, Upload, Result, History
│   └── api.py                  # JSON REST API (/health, /api/predict)
├── database/
│   ├── db.py                   # SQLAlchemy instance
│   └── models.py               # User and Prediction database models
├── templates/                  # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── upload.html
│   ├── result.html
│   ├── history.html
│   └── 404.html
├── static/
│   ├── css/style.css           # SaaS design system & responsive styling
│   └── js/app.js               # Drag-and-drop file upload & dynamic preview
└── tests/
    ├── conftest.py             # Pytest fixtures & isolated in-memory DB
    ├── test_app.py             # UI routing & template tests
    ├── test_api.py             # REST API & health endpoint tests
    ├── test_auth.py            # Authentication & access control tests
    └── test_model.py           # ML inference & parser tests
```

---

## 4. Local Setup & Installation

### Step 1: Clone or navigate to the workspace
```powershell
cd D:\MLOPS
```

### Step 2: Create and activate Python virtual environment
```powershell
# Create venv with Python 3.10
py -3.10 -m venv venv

# Activate venv on Windows PowerShell
.\venv\Scripts\Activate.ps1
```
*(On Ubuntu/macOS: `python3 -m venv venv && source venv/bin/activate`)*

### Step 3: Install dependencies
```powershell
pip install -r requirements-dev.txt
```

### Step 4: Configure environment variables
```powershell
Copy-Item .env.example .env
```

---

## 5. Machine Learning Pipeline (`train_model.py`)

Train the model and save the fitted scikit-learn pipeline to `model/resume_classifier.joblib`:

```powershell
# Train using the smoke-test seed dataset
python train_model.py --dataset data/sample_resumes.csv --output model/resume_classifier.joblib

# Or train using the full Kaggle dataset
python train_model.py --dataset data/UpdatedResumeDataSet.csv --output model/resume_classifier.joblib
```

### 5.1 Programmatic Inference via PredictionService
```python
from services.prediction_service import get_prediction_service

service = get_prediction_service()
result = service.predict("Experienced DevOps engineer with Docker, Kubernetes, Terraform, and AWS.")

print("Predicted Category:", result["predicted_category"])
print("Confidence:", result["confidence"])
print("Keywords:", result["top_keywords"])
print("Probabilities:", result["probabilities"][:3])
```

---

## 6. Running the Web Application Locally

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Start the Flask application
python app.py
```
- Web Application: `http://127.0.0.1:5000`
- System Health Check: `http://127.0.0.1:5000/health`
- Resume Upload Interface: `http://127.0.0.1:5000/prediction`
- User Registration: `http://127.0.0.1:5000/register`
- User Sign In: `http://127.0.0.1:5000/login`

---

## 7. REST API Reference

The application provides a public REST API for programmatic predictions without CSRF friction.

### Health Check Endpoint
- **URL**: `GET /health`
- **Response**: `200 OK` (or `503 Service Unavailable` if model artifact is missing)
```json
{
  "status": "ok",
  "service": "resume-category-classifier",
  "model_loaded": true,
  "timestamp": "2026-10-18T00:00:00.000000Z"
}
```

### Prediction Endpoint
- **URL**: `POST /api/predict`
- **Content-Type**: `multipart/form-data` OR `application/json`

#### Option A: Multipart File Upload (PDF or DOCX)
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -F "file=@/path/to/candidate_resume.pdf"
```

#### Option B: JSON Text Payload
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Senior DevOps Engineer with 6 years experience in Kubernetes, Docker, Terraform, AWS EC2, Jenkins, and Linux."}'
```

#### Successful JSON Response (`200 OK`)
```json
{
  "status": "success",
  "filename": "candidate_resume.pdf",
  "predicted_category": "DevOps Engineer",
  "confidence": 0.885,
  "confidence_percentage": 88.5,
  "top_keywords": [
    "Kubernetes",
    "Docker",
    "Terraform",
    "Aws",
    "Linux",
    "Jenkins"
  ],
  "probabilities": [
    {"category": "DevOps Engineer", "probability": 0.885, "percentage": 88.5},
    {"category": "Python Developer", "probability": 0.052, "percentage": 5.2},
    {"category": "Java Developer", "probability": 0.031, "percentage": 3.1}
  ],
  "text_length": 1420,
  "timestamp": "2026-10-18T00:00:00.000000Z"
}
```

---

## 8. Running Automated Tests

Run the complete test suite (unit tests, API integration tests, parser tests, and auth tests):

```powershell
pytest -v
```
All tests use temporary SQLite databases and generated in-memory PDF/DOCX documents.

---

## 8. Docker & Cloud Deployment Commands

### Practical 2: Local Docker Container
```powershell
# Build Docker image
docker build -t resume-classifier:1.0 .

# Run container
docker run -d --name resume-classifier-app -p 5000:5000 resume-classifier:1.0

# Verify
docker ps
curl http://localhost:5000/health
```

### Practical 3: AWS EC2 Deployment
```bash
# Connect to EC2
ssh -i "your-key.pem" ubuntu@YOUR_EC2_PUBLIC_IP

# Install Docker on EC2
sudo apt update && sudo apt install -y docker.io
sudo systemctl enable --now docker

# Run application container
sudo docker run -d --name resume-classifier-app --restart unless-stopped -p 5000:5000 resume-classifier:1.0
```

### Practical 4: Google Cloud Platform (GCP) Deployment
```bash
# Connect via SSH in browser or gcloud
# Configure firewall rule allow-ml-app-5000 for TCP:5000
# Run container on GCP VM
sudo docker run -d --name resume-classifier-app --restart unless-stopped -p 5000:5000 resume-classifier:1.0
```
