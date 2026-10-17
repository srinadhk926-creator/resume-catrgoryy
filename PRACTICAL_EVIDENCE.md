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
*Next: Practical 2 (Docker Containerization) instructions will follow once Practical 1 is signed off.*
