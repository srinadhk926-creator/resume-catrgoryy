# Resume Dataset Guidelines

This directory stores the datasets used for training and evaluating the **ResumeAI** category classification model.

---

## 1. Supported Dataset Schema

The training pipeline in `train_model.py` accepts CSV files formatted with the following columns:

| Column Name | Alternative Recognized Names | Data Type | Description |
| :--- | :--- | :--- | :--- |
| `Resume` | `Resume_Text`, `text`, `resume_text` | Text (String) | Cleaned or raw text content of the candidate's resume. |
| `Category` | `category`, `label`, `Job_Title` | Categorical (String) | Ground-truth professional category or role domain. |

### Example Categories
- `Data Science`
- `Python Developer`
- `Java Developer`
- `DevOps Engineer`
- `Web Designing`
- `Software Engineer`
- `Database Administrator`
- `HR`
- `Business Analyst`
- `Network Security Engineer`

---

## 2. Recommended Production Dataset

For full-scale evaluation and high classification accuracy:

1. Visit Kaggle's **Resume Dataset** page:
   [https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
2. Download `UpdatedResumeDataSet.csv`.
3. Place the file directly in this directory:
   ```
   D:\MLOPS\data\UpdatedResumeDataSet.csv
   ```
4. Run the training script:
   ```powershell
   python train_model.py --dataset data/UpdatedResumeDataSet.csv
   ```

---

## 3. Seed / Smoke-Testing Dataset (`sample_resumes.csv`)

For local testing, continuous integration, and rapid practical verification when offline:
- A seed CSV (`data/sample_resumes.csv`) is provided.
- **Notice**: This dataset contains curated representative resumes intended strictly for smoke testing the end-to-end training, serialization, and prediction pipeline. Production deployments should be trained on the full `UpdatedResumeDataSet.csv`.
