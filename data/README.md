# Resume Dataset Guidelines

This directory stores datasets used for training and evaluating the **ResumeAI** category classification model.

---

## 1. Supported Dataset Schema

The training pipeline in `train_model.py` accepts any standard CSV formatted with the following columns:

| Column Name | Alternative Recognized Names | Data Type | Description |
| :--- | :--- | :--- | :--- |
| `Resume` | `Resume_Text`, `resume_text`, `text`, `resume` | String | Cleaned or raw extracted text of the resume. |
| `Category` | `category`, `label`, `Job_Title`, `job_title` | String | Ground-truth professional category or role domain. |

### Categories in Seed Dataset (10 Domains)
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

---

## 2. Seed / Smoke-Testing Dataset (`sample_resumes.csv`)

- **Location**: `data/sample_resumes.csv`
- **Size**: 40 records (4 balanced records per category across 10 categories).
- **Purpose**: Verifies that the training pipeline, TF-IDF vectorization, classifier comparison, stratification, serialization, and test suite execute without errors out of the box in offline and CI environments.
- **Limitation**: While accuracy on the 10-record test split is high (90%), the small sample size (N=40) means this score is purely illustrative for system verification (smoke testing). It must not be cited as real-world production performance.

---

## 3. Recommended Production Dataset

For production benchmarking and model training:

1. **Download Kaggle Dataset**:
   - URL: [https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
   - File: `UpdatedResumeDataSet.csv` (contains 962 real resumes across 25 categories).
2. **Place in Directory**:
   ```
   D:\MLOPS\data\UpdatedResumeDataSet.csv
   ```
3. **Train Pipeline**:
   ```powershell
   python train_model.py --dataset data/UpdatedResumeDataSet.csv --output model/resume_classifier.joblib
   ```
   The pipeline will automatically detect the columns, perform stratified 75/25 train/test splitting, train and compare Logistic Regression with Multinomial Naive Bayes, output classification metrics, and save the fitted model.

---

## 4. Generated Artifacts

- **Trained Model Pipeline**: `model/resume_classifier.joblib`
- **Evaluation Report & Metadata**: `model/evaluation_metrics.json`
