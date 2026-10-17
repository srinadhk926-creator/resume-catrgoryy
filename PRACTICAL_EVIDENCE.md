# College Subject Practical Evidence & Screenshot Guide
## 2CEAI702 — Machine Learning Operations (MLOps)

This document provides the exact commands, execution steps, expected outputs, and screenshot capture instructions required for submission in your 2CEAI702 laboratory record.

---

# PRACTICAL 1: Converting a Machine Learning Notebook into a Flask ML System

### Aim:
To convert a machine learning model developed in a training pipeline into a structured, modular, and deployable Flask-based machine learning application with model serialization, input validation, prediction functionality, and a web interface.

---

### Step-by-Step Execution & Screenshot Checklist

#### 📸 Screenshot 1: Project Folder & Virtual Environment
- **Purpose**: Demonstrate project organization, activated Python 3.10 virtual environment, and dependency installation.
- **Where to Execute**: Local Windows PowerShell inside `D:\MLOPS`.
- **Command to Run**:
  ```powershell
  cd D:\MLOPS
  .\venv\Scripts\Activate.ps1
  python --version
  pip list
  Get-ChildItem -Path . -Exclude venv, __pycache__
  ```
- **What to Capture in Screenshot**:
  1. PowerShell prompt showing `(venv) D:\MLOPS>`.
  2. Python version output `Python 3.10.11`.
  3. Pinned package list showing `Flask`, `scikit-learn`, `PyMuPDF`, `python-docx`, `SQLAlchemy`, etc.
  4. Clean directory structure with `app.py`, `train_model.py`, `routes/`, `services/`, `templates/`, and `tests/`.

---

#### 📸 Screenshot 2: Model Training Output & Serialized Artifacts
- **Purpose**: Prove that the TF-IDF and classifier pipeline was trained, evaluated with authentic metrics, and serialized to disk.
- **Where to Execute**: Windows PowerShell inside `D:\MLOPS`.
- **Command to Run**:
  ```powershell
  .\venv\Scripts\python.exe train_model.py --dataset data/sample_resumes.csv --output model/resume_classifier.joblib
  Get-Item model/resume_classifier.joblib, model/evaluation_metrics.json
  ```
- **What to Capture in Screenshot**:
  1. Category distribution table across the 10 job categories.
  2. Training progress logs, model comparison between Logistic Regression and Naive Bayes.
  3. Classification report showing Precision, Recall, and F1-score.
  4. Confirmation message: `[SUCCESS] Model training and serialization completed successfully.`
  5. File listing showing `resume_classifier.joblib` (~152 KB) and `evaluation_metrics.json`.

---

#### 📸 Screenshot 3: Running Flask Web Server in Terminal
- **Purpose**: Verify that the application initializes cleanly and listens on port 5000.
- **Where to Execute**: Windows PowerShell.
- **Command to Run**:
  ```powershell
  python app.py
  ```
- **What to Capture in Screenshot**:
  1. Startup banner `(ResumeAI-App) ResumeAI initialized successfully in 'development' environment.`
  2. Serving Flask app `app` on `http://127.0.0.1:5000` and `http://0.0.0.0:5000`.
  3. Debug mode active.

---

#### 📸 Screenshot 4: Running Flask Web Interface (Upload Form)
- **Purpose**: Show the live web UI for resume input.
- **Where to Execute**: Web browser.
- **URL to Open**:
  ```
  http://127.0.0.1:5000/prediction
  ```
- **What to Capture in Screenshot**:
  1. Browser navigation bar showing `http://127.0.0.1:5000/prediction`.
  2. Modern header with ResumeAI brand, Home, Classify Resume, and Sign In buttons.
  3. Drag-and-drop upload zone with support for PDF and DOCX documents (Max 5 MB) and the Paste Resume Text tab.

---

#### 📸 Screenshot 5: Prediction Result & Probability Breakdown
- **Purpose**: Verify end-to-end model inference, confidence score, category distribution, and keyword extraction in the UI.
- **Where to Execute**: Web browser.
- **Action**: Upload any resume PDF/DOCX (or paste candidate text) and click **Analyze & Classify Resume**.
- **What to Capture in Screenshot**:
  1. The resulting page at `http://127.0.0.1:5000/prediction`.
  2. Large highlighted badge for **Predicted Resume Category** (e.g. *DevOps Engineer*, *Data Science*, *Python Developer*).
  3. Model confidence progress bar and percentage score.
  4. Extracted technical skills pills (e.g., *Docker*, *Kubernetes*, *AWS*, *Terraform*).
  5. Category Probabilities Breakdown card showing percentage distribution across other domains.
  6. Confidence disclaimer notice.

---

#### 📸 Screenshot 6: System Health Check JSON Endpoint
- **Purpose**: Verify the production monitoring endpoint required by MLOps systems.
- **Where to Execute**: Web browser OR terminal (`curl`).
- **URL / Command**:
  ```powershell
  curl http://127.0.0.1:5000/health
  ```
- **What to Capture in Screenshot**:
  1. Browser or terminal showing clean JSON response:
  ```json
  {
    "model_loaded": true,
    "service": "resume-category-classifier",
    "status": "ok",
    "timestamp": "..."
  }
  ```
  2. HTTP Status `200 OK`.

---

#### 📸 Screenshot 7: Programmatic REST API Prediction Response
- **Purpose**: Prove that external clients and microservices can query the machine learning system via JSON REST API.
- **Where to Execute**: Windows PowerShell or Command Prompt.
- **Command to Run**:
  ```powershell
  curl -X POST http://127.0.0.1:5000/api/predict `
    -H "Content-Type: application/json" `
    -d "{\"text\": \"Senior Data Scientist with 5 years experience in machine learning, deep learning, PyTorch, scikit-learn, NLP, and SQL.\"}"
  ```
- **What to Capture in Screenshot**:
  1. Command sent via PowerShell.
  2. Formatted JSON response returning `"predicted_category": "Data Science"`, confidence score, and top extracted keywords.

---

#### 📸 Screenshot 8: Automated Pytest Test Suite Results
- **Purpose**: Verify that the entire system has automated regression testing.
- **Where to Execute**: Windows PowerShell.
- **Command to Run**:
  ```powershell
  .\venv\Scripts\python.exe -m pytest -v
  ```
- **What to Capture in Screenshot**:
  1. Terminal output showing all 45 automated tests passing (`PASSED [100%]`).
  2. Summary line: `45 passed in X.XXs`.
  3. Coverage across model validation, API endpoints, authentication, and parser fixtures.

---

### Summary Checklist for Practical 1 Submission

| Requirement | Status | Verification Reference |
| :--- | :---: | :--- |
| **Virtual Environment & Dependencies** | ✅ Verified | Screenshot 1 (`pip list`) |
| **Training Pipeline & Serialization** | ✅ Verified | Screenshot 2 (`model/resume_classifier.joblib`) |
| **Flask Application Routes** | ✅ Verified | Screenshot 3 & 4 (`app.py` on `0.0.0.0:5000`) |
| **PDF & DOCX Resume Parsing** | ✅ Verified | Screenshot 5 (PyMuPDF & python-docx) |
| **Prediction UI & Confidence Breakdown** | ✅ Verified | Screenshot 5 (`/prediction` result view) |
| **System Health Endpoint** | ✅ Verified | Screenshot 6 (`/health` JSON) |
| **REST API Prediction Endpoint** | ✅ Verified | Screenshot 7 (`/api/predict` JSON) |
| **Automated Test Suite (45 Tests)** | ✅ Verified | Screenshot 8 (`pytest -v`) |

---

# PRACTICAL 2: Dockerizing the Machine Learning System

### Aim:
To create a production-appropriate Dockerfile for the Flask-based machine learning application, build a Docker image, run the image as an isolated container with persistent SQLite volume storage and non-root execution, and verify the application's functionality.

---

### Step-by-Step Execution & Screenshot Checklist

#### 📸 Screenshot 9: Docker Engine Status & Version
- **Purpose**: Verify that Docker Engine is installed and active on the host.
- **Where to Execute**: Windows PowerShell.
- **Command to Run**:
  ```powershell
  docker --version
  docker info
  ```
- **What to Capture in Screenshot**:
  1. Docker client version (`Docker version 29.3.1` or current).
  2. Server information showing Linux containers / WSL2 engine running.

---

#### 📸 Screenshot 10: Docker Image Build Output
- **Purpose**: Demonstrate successful image build without cache corruption or build errors.
- **Where to Execute**: Windows PowerShell in `D:\MLOPS`.
- **Command to Run**:
  ```powershell
  docker build -t resumeai:local .
  ```
- **What to Capture in Screenshot**:
  1. Build steps executing: Python 3.10-slim base, dependency installation from `requirements.txt`, creation of `appuser`, copying source files and `model/resume_classifier.joblib`.
  2. Final confirmation: `naming to docker.io/library/resumeai:local done` or `Successfully tagged resumeai:local`.

---

#### 📸 Screenshot 11: Docker Images Listing & Image Size
- **Purpose**: Verify that the image was registered locally and inspect its footprint.
- **Where to Execute**: Windows PowerShell.
- **Command to Run**:
  ```powershell
  docker images | Select-String -Pattern "resumeai|REPOSITORY"
  ```
- **What to Capture in Screenshot**:
  1. Repository: `resumeai`.
  2. Tag: `local`.
  3. Image ID and actual compressed virtual size.

---

#### 📸 Screenshot 12: Running Container & Port Mapping (`docker ps`)
- **Purpose**: Confirm container is running in detached mode with port 5000 mapped and healthcheck active.
- **Where to Execute**: Windows PowerShell.
- **Command to Run**:
  ```powershell
  docker run -d --name resumeai-app -p 5000:5000 -v resumeai_data:/app/instance resumeai:local
  docker ps
  ```
- **What to Capture in Screenshot**:
  1. Container ID output from `docker run`.
  2. `docker ps` table showing `NAMES: resumeai-app`, `PORTS: 0.0.0.0:5000->5000/tcp`, `STATUS: Up ... (healthy)`.

---

#### 📸 Screenshot 13: Container Gunicorn Startup Logs
- **Purpose**: Verify that Gunicorn WSGI server spawned workers cleanly under the non-root user.
- **Where to Execute**: Windows PowerShell.
- **Command to Run**:
  ```powershell
  docker logs resumeai-app
  ```
- **What to Capture in Screenshot**:
  1. Gunicorn logging: `[INFO] Starting gunicorn ... Listening at: http://0.0.0.0:5000`.
  2. Worker processes booting without Python exceptions or missing module errors.

---

#### 📸 Screenshot 14: Containerized Web Interface in Browser
- **Purpose**: Demonstrate that the containerized application serves the UI to local clients.
- **Where to Execute**: Web Browser.
- **URL to Open**:
  ```
  http://localhost:5000/prediction
  ```
- **What to Capture in Screenshot**:
  1. Browser showing `http://localhost:5000/prediction`.
  2. Upload interface rendered directly from the running container.

---

#### 📸 Screenshot 15: Containerized Prediction & Health Verification
- **Purpose**: Prove end-to-end inference and health checking through container port forwarding.
- **Where to Execute**: Windows PowerShell.
- **Commands to Run**:
  ```powershell
  # 1. Healthcheck verification
  Invoke-RestMethod -Uri "http://localhost:5000/health" -Method Get

  # 2. Real inference through containerized API
  $body = @{ text = "Senior DevOps Engineer with Kubernetes, Docker, Terraform, AWS EC2, and CI/CD." } | ConvertTo-Json
  Invoke-RestMethod -Uri "http://localhost:5000/api/predict" -Method Post -Body $body -ContentType "application/json"
  ```
- **What to Capture in Screenshot**:
  1. Health check returning `status: ok`, `model_loaded: True`.
  2. API prediction returning `predicted_category: DevOps Engineer`, confidence score, and extracted skills.

---

#### 📸 Screenshot 16: Container Lifecycle & Volume Persistence
- **Purpose**: Demonstrate container restart without data loss using named Docker volume `resumeai_data`.
- **Where to Execute**: Windows PowerShell.
- **Commands to Run**:
  ```powershell
  # Inspect health and volume
  docker inspect --format='{{json .State.Health.Status}}' resumeai-app
  docker volume ls | Select-String "resumeai_data"

  # Restart container to verify persistence
  docker restart resumeai-app
  docker ps
  ```
- **What to Capture in Screenshot**:
  1. Health status reporting `"healthy"`.
  2. Named volume `resumeai_data` attached to `/app/instance`.
  3. Clean container restart.

---

### Summary Checklist for Practical 2 Submission

| Requirement | Status | Verification Reference |
| :--- | :---: | :--- |
| **Dockerfile Created** | ✅ Verified | Multi-stage / Python 3.10-slim, non-root user, Gunicorn on `0.0.0.0:5000` |
| **.dockerignore Configured** | ✅ Verified | Excludes venvs, `.env`, caches, git, and local test DBs |
| **Docker Compose Configured** | ✅ Verified | `docker-compose.yml` with persistent named volume `resumeai_data` |
| **Docker Build Execution** | Ready to capture | Screenshot 10 (`docker build -t resumeai:local .`) |
| **Docker Container Execution** | Ready to capture | Screenshot 12 (`docker ps`) |
| **Gunicorn Logs Inspection** | Ready to capture | Screenshot 13 (`docker logs resumeai-app`) |
| **Container Healthcheck** | Ready to capture | Screenshot 15 (`/health` & `docker inspect`) |
| **Containerized Inference** | Ready to capture | Screenshot 15 (`/api/predict` JSON curl) |
| **Volume Persistence** | Ready to capture | Screenshot 16 (`docker volume ls`, `docker restart`) |

---
*Next: Practical 3 (AWS EC2 Deployment) and Practical 4 (GCP Compute Engine Deployment) instructions will follow once Practical 2 is completed.*
