# ResumeAI — Resume Category Classification System

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Flask%203.0-green.svg)](https://flask.palletsprojects.com/)
[![ML Pipeline](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Containerized](https://img.shields.io/badge/Docker-enabled-2496ED.svg)](https://www.docker.com/)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2%20Deployed-FF9900.svg)](https://aws.amazon.com/ec2/)
[![Tests](https://img.shields.io/badge/tests-58%20passed-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)]()

**Course / Subject**: 2CEAI702 — Machine Learning Operations (MLOps)  
**Live AWS Deployment**: [http://13.60.105.66:5000](http://13.60.105.66:5000) *(AWS EC2 Ubuntu Host)*

---

## 1. Project Overview & Problem Statement

In modern recruitment and talent management pipelines, technical recruiters and automated applicant tracking systems (ATS) receive thousands of resumes daily in varying unstructured formats (PDF, Word DOCX, raw text). Manually screening and tagging candidates into appropriate engineering disciplines is slow, error-prone, and inconsistent.

**ResumeAI** is an end-to-end, production-minded Machine Learning Operations (MLOps) system that:
1. Ingests resumes in PDF and DOCX formats or through direct text input.
2. Extracts and sanitizes textual content using PyMuPDF and python-docx.
3. Automatically classifies candidate resumes into 10 target industry domains using a TF-IDF vectorization and Logistic Regression pipeline.
4. Generates calibrated class probability distributions and extracts domain skills.
5. Employs real-time **Ambiguity & Low-Confidence Detection** to alert users when resumes span multi-disciplinary roles.
6. Persists audit trails in SQLite with user authentication, CSRF security, and individual analysis reports.
7. Packages the entire solution into a production Docker container deployed on AWS EC2.

---

## 2. Key Features

- **Document Processing**: Fast, secure parsing of PDF and DOCX files with magic byte validation, file size limits (5 MB max), and PII masking (emails, phone numbers, addresses).
- **Machine Learning Inference**: TF-IDF feature extraction with resume boilerplate stop words, sublinear term frequency scaling, and regularized Logistic Regression.
- **Model Comparison**: Automated training-time comparison between Logistic Regression and Multinomial Naive Bayes with cross-validation.
- **Ambiguity Detection**: Automatically flags multi-disciplinary candidate profiles with low posterior confidence (< 22%) or narrow top-2 category margins (< 4%).
- **SaaS Web Workspace**: Modern responsive UI with split-screen authentication, interactive dashboard, drag-and-drop file uploader, and persistent audit reports.
- **Programmatic REST API**: Public `/health` probe and `/api/predict` endpoints supporting JSON and multipart/form-data.
- **Containerized & Cloud-Ready**: Non-root Docker execution with Gunicorn WSGI server, healthchecks, and volume-mounted SQLite persistence deployed on AWS EC2.
- **Rigorous Verification**: 58 automated unit, integration, parser, API, and regression tests with 100% pass rate.

---

## 3. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Machine Learning** | `scikit-learn 1.5.2`, `pandas 2.2.3`, `numpy 2.0.2`, `joblib 1.4.2` |
| **Backend & Web Framework** | Python 3.10 / 3.11, `Flask 3.0.3`, `Werkzeug 3.0.3`, `Flask-WTF 1.2.1` |
| **Database & ORM** | SQLite 3, `SQLAlchemy 2.0.36` (with automated schema upgrades) |
| **Document Processing** | `PyMuPDF 1.24.11` (fitz), `python-docx 1.1.2` |
| **WSGI Server & Containers** | `Gunicorn 23.0.0`, `Docker`, `docker-compose` |
| **Cloud Hosting** | AWS EC2 (Ubuntu 24.04 LTS), Security Groups |
| **Testing & Quality** | `pytest 8.3.3`, `pytest-cov 5.0.0` |

---

## 4. Project Directory Structure

```text
ResumeAI (D:\MLOPS)
├── data/
│   ├── README.md               # Dataset documentation & production guide
│   └── sample_resumes.csv      # 150 balanced benchmark resumes across 10 classes
├── database/
│   ├── db.py                   # Engine initialization, scoped sessions & schema upgrades
│   └── models.py               # User and Prediction SQLAlchemy data models
├── instance/
│   └── resume_ai.db            # Persistent SQLite database (auto-created)
├── model/
│   ├── evaluation_metrics.json # Confusion matrix, classification report & metadata
│   └── resume_classifier.joblib# Serialized TF-IDF + Logistic Regression pipeline
├── routes/
│   ├── api.py                  # Public REST API (/health, /api/predict)
│   ├── auth.py                 # User authentication (register, login, logout)
│   └── main.py                 # UI routes (index, dashboard, prediction, reports, history)
├── services/
│   ├── prediction_service.py   # Model loading, inference, confidence & skill extraction
│   └── resume_parser.py        # PDF & DOCX document parsing and PII redaction
├── static/
│   ├── css/
│   │   └── style.css           # SaaS design system, responsive layout & dark/light themes
│   └── js/
│       └── app.js              # File upload drag-and-drop & loading overlays
├── templates/
│   ├── 404.html                # Custom error page
│   ├── base.html               # Base layout template
│   ├── dashboard.html          # User metrics & recent activity
│   ├── history.html            # Audit history & report deletion
│   ├── index.html              # Public SaaS landing page
│   ├── login.html              # Sign-in form
│   ├── profile.html            # Account profile overview
│   ├── register.html           # User registration form
│   ├── report.html             # Printable analysis report detail
│   ├── result.html             # Live classification results & probability bars
│   └── upload.html             # Document upload & text analysis interface
├── tests/
│   ├── conftest.py             # Pytest fixtures & isolated in-memory DB
│   ├── test_api.py             # REST API & healthcheck tests
│   ├── test_app.py             # UI routing & report tests
│   ├── test_auth.py            # User authentication & session tests
│   └── test_model.py           # ML training, inference & regression tests
├── .dockerignore               # Container build exclusions
├── .env.example                # Template environment variables
├── .gitignore                  # Git exclusions (secrets, venv, databases, tars)
├── app.py                      # Flask Application Factory & WSGI entrypoint
├── config.py                   # Development, testing, and production configs
├── docker-compose.yml          # Local container orchestration
├── Dockerfile                  # Production container definition
├── PRACTICAL_EVIDENCE.md       # College MLOps practical report evidence
├── PROJECT_STRUCTURE.txt       # Compact plain-text directory tree
├── pytest.ini                  # Pytest configuration
├── README.md                   # Complete system documentation
├── requirements.txt            # Pinned runtime dependencies
├── requirements-dev.txt        # Development & testing dependencies
└── train_model.py              # ML training, cross-validation & evaluation script
```

---

## 5. Machine Learning Pipeline & Evaluation

### 5.1 Architecture & Approach

The classification pipeline is built with scikit-learn:

```
[Raw Resume Text / PDF / DOCX]
             │
             ▼
   [Text Preprocessing]
   • Non-printable character removal
   • Whitespace normalization
   • PII redaction (email, phone, address)
             │
             ▼
    [TF-IDF Vectorizer]
   • max_features = 8,000
   • ngram_range = (1, 2) (unigrams & bigrams)
   • sublinear_tf = True (dampens repeated tokens)
   • max_df = 0.85 (ignores corpus-wide ubiquitous words)
   • min_df = 1
   • stop_words = RESUME_STOP_WORDS (English stops + 40+ resume formatting terms)
             │
             ▼
 [Classifier Comparison & Fit]
   • Primary: Logistic Regression (C=3.0, class_weight='balanced', max_iter=1000)
   • Comparison: Multinomial Naive Bayes (alpha=0.1)
             │
             ▼
  [Inference & Ambiguity Engine]
   • Category posterior probability distribution
   • Low-confidence flag: is_low_confidence = True (if confidence < 22% or margin < 4%)
   • Technical skill keyword extraction (24+ industry domains)
```

### 5.2 Supported Categories (10 Domains)

1. `Data Science`
2. `Python Developer`
3. `Java Developer`
4. `DevOps Engineer`
5. `Web Designing`
6. `Software Engineer`
7. `HR`
8. `Business Analyst`
9. `Database Administrator`
10. `Network Security Engineer`

### 5.3 Verified Evaluation Metrics

*Source: `model/evaluation_metrics.json` (evaluated on stratified 75/25 split of 150 benchmark samples)*

| Metric | Score |
| :--- | :--- |
| **Total Samples** | 150 records (15 balanced records per class) |
| **Train / Test Split** | 112 Train / 38 Test (Stratified) |
| **Overall Accuracy** | **86.84%** (0.8684) |
| **Weighted Precision** | **87.81%** (0.8781) |
| **Weighted Recall** | **86.84%** (0.8684) |
| **Weighted F1-Score** | **86.80%** (0.8680) |
| **5-Fold Cross-Validation Accuracy** | **96.00%** (Macro-F1: 95.95%) |

#### Per-Class Performance (Holdout Test Set)

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Business Analyst** | 1.00 | 1.00 | 1.00 | 4 |
| **Data Science** | 1.00 | 1.00 | 1.00 | 3 |
| **Database Administrator** | 0.80 | 1.00 | 0.89 | 4 |
| **DevOps Engineer** | 0.67 | 0.50 | 0.57 | 4 |
| **HR** | 1.00 | 1.00 | 1.00 | 4 |
| **Java Developer** | 1.00 | 0.75 | 0.86 | 4 |
| **Network Security Engineer** | 1.00 | 1.00 | 1.00 | 4 |
| **Python Developer** | 0.75 | 0.75 | 0.75 | 4 |
| **Software Engineer** | 0.50 | 0.67 | 0.57 | 3 |
| **Web Designing** | 1.00 | 1.00 | 1.00 | 4 |

> **Notice on Benchmark Metrics**: These evaluation metrics were computed on the curated 150-sample benchmark dataset (`sample_resumes.csv`). While they verify model convergence, feature calibration, and pipeline integrity, they reflect the distribution of this benchmark corpus and are not a guarantee of real-world production performance across external arbitrary document styles.

---

## 6. Installation & Local Setup (Windows PowerShell)

### Prerequisites

- **Python**: 3.10.x or 3.11.x
- **Git**: Installed and configured
- **Docker Desktop**: (Optional, for containerized execution)

### Step 1: Clone Repository & Navigate

```powershell
git clone https://github.com/gallajagadeesh17/Resume-Category-Classification.git
cd Resume-Category-Classification
```

### Step 2: Create Virtual Environment

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

*(On macOS/Linux: `python3 -m venv venv && source venv/bin/activate`)*

### Step 3: Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### Step 4: Environment Variables

```powershell
Copy-Item .env.example .env
```

---

## 7. Model Training & Serialization

To retrain the model and regenerate `model/resume_classifier.joblib` and `model/evaluation_metrics.json`:

```powershell
python train_model.py --dataset data/sample_resumes.csv --output model/resume_classifier.joblib
```

Command-line arguments:
- `--dataset`: Path to training CSV (default: `data/sample_resumes.csv`)
- `--output`: Path to serialized model joblib (default: `model/resume_classifier.joblib`)
- `--metrics-path`: Path to evaluation JSON (default: `model/evaluation_metrics.json`)
- `--test-size`: Test split ratio (default: `0.25`)
- `--random-seed`: Random seed for reproducibility (default: `42`)
- `--no-compare`: Skip Naive Bayes comparison and train Logistic Regression only

---

## 8. Running the Web Application

Start the local Flask development server:

```powershell
python app.py
```

Access the application in your browser:
- **Landing Page**: `http://127.0.0.1:5000/`
- **Health Check**: `http://127.0.0.1:5000/health`
- **Resume Upload & Inference**: `http://127.0.0.1:5000/prediction`
- **User Dashboard**: `http://127.0.0.1:5000/dashboard`
- **Audit History**: `http://127.0.0.1:5000/history`
- **Sign In / Sign Up**: `http://127.0.0.1:5000/login` | `http://127.0.0.1:5000/register`

---

## 9. REST API Reference

The application exposes RESTful endpoints for programmatic integration:

### 9.1 Health Check Endpoint
- **URL**: `GET /health`
- **Response**: `200 OK` (or `503 Service Unavailable`)

```json
{
  "model_loaded": true,
  "service": "resume-category-classifier",
  "status": "ok",
  "timestamp": "2026-10-18T10:00:00.000000Z"
}
```

### 9.2 Resume Prediction Endpoint
- **URL**: `POST /api/predict`
- **Content-Type**: `application/json` OR `multipart/form-data`

#### Option A: JSON Payload
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Senior Enterprise Java Developer with 8 years building distributed microservices using Spring Boot, Hibernate, and Apache Kafka."}'
```

#### Option B: Multipart Document Upload (PDF / DOCX)
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -F "file=@/path/to/resume.pdf"
```

#### Sample Response (`200 OK`)
```json
{
  "status": "success",
  "filename": "direct_text_input",
  "predicted_category": "Java Developer",
  "confidence": 0.6678,
  "confidence_percentage": 66.8,
  "is_low_confidence": false,
  "confidence_warning": null,
  "margin": 0.6207,
  "probabilities": [
    {"category": "Java Developer", "percentage": 66.8, "probability": 0.6678},
    {"category": "Network Security Engineer", "percentage": 4.7, "probability": 0.0471},
    {"category": "Python Developer", "percentage": 4.1, "probability": 0.0413}
  ],
  "top_keywords": ["Spring Boot", "Hibernate", "Java", "Microservices", "Kafka", "Spring"],
  "text_length": 145,
  "timestamp": "2026-10-18T10:00:00.000000Z"
}
```

---

## 10. Docker Containerization (Practical 2)

### 10.1 Build Docker Image

```powershell
docker build -t resumeai:local .
```

### 10.2 Run Container with SQLite Persistence

```powershell
docker run -d `
  --name resumeai `
  --restart unless-stopped `
  -p 5000:5000 `
  -v resumeai_data:/app/instance `
  -v resumeai_uploads:/app/uploads `
  resumeai:local
```

### 10.3 Run with Docker Compose

```powershell
docker compose up -d
docker compose ps
docker compose logs -f
```

---

## 11. AWS EC2 Cloud Deployment (Practical 3)

The application is deployed on an **AWS EC2 Ubuntu 24.04 LTS** instance:

- **Public Endpoint**: `http://13.60.105.66:5000`
- **Public Health Probe**: `http://13.60.105.66:5000/health`
- **Container Name**: `resumeai` (Restart policy: `unless-stopped`)
- **Persistent Bind Mounts**:
  - `/home/ubuntu/resumeai/data` $\rightarrow$ `/app/instance` (holds `resume_ai.db`)
  - `/home/ubuntu/resumeai/uploads` $\rightarrow$ `/app/uploads`

### Deployment Steps:
1. Provision EC2 Ubuntu instance with Security Group allowing ports `22` (SSH, restricted to your IP) and `5000` (Custom TCP).
2. Transfer image archive: `scp -i <key.pem> resumeai-image.tar ubuntu@<ip>:~/resumeai/`
3. Load image: `sudo docker load -i resumeai-image.tar`
4. Run container with persistent host volumes.
5. Verify container health: `curl -s http://localhost:5000/health`

---

## 12. Automated Testing

The project includes 58 automated unit, integration, and regression tests:

```powershell
# Run all tests with verbose output
pytest -v

# Run with test coverage report
pytest --cov=. --cov-report=term-missing
```

Test suite coverage:
- `tests/test_api.py`: Health probes, JSON/multipart prediction, bad payload error codes (400, 413, 415, 503).
- `tests/test_app.py`: Route navigation, login redirection, template rendering, report persistence, unauthorized access checks.
- `tests/test_auth.py`: Registration, password hashing, session management, protected route guards.
- `tests/test_model.py`: Data validator, column detection, TF-IDF pipeline, confusion matrix integrity, multi-class distinctiveness regression tests, ambiguity detection.

---

## 13. Limitations & Future Roadmap

1. **Benchmark Corpus Size**: The seed dataset (`sample_resumes.csv`) contains 150 records for laboratory smoke testing. For production deployments, training on Kaggle's 962-resume dataset (`UpdatedResumeDataSet.csv`) is recommended.
2. **Multi-Role Overlap**: Students with multi-disciplinary backgrounds (e.g., Computer Engineering in AI with web development skills) correctly receive shared probabilities. Future versions can incorporate multi-label classification to assign multiple secondary tags.
3. **Deep Learning Embeddings**: Transition from TF-IDF n-grams to transformer embeddings (e.g., `all-MiniLM-L6-v2`) for semantic similarity matching.
4. **Automated CI/CD**: Implement GitHub Actions pipeline to test, build, and deploy container images to AWS ECR/ECS automatically.

---

## 14. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
Developed by srinadh K..
