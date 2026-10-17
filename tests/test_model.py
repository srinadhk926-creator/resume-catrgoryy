"""
Unit and integration tests for ML training pipeline and prediction service.
"""

import os
import json
import pytest
import pandas as pd
from pathlib import Path

from train_model import (
    load_and_validate_data,
    build_pipeline,
    split_data,
    train_and_evaluate,
    detect_columns,
)
from services.prediction_service import (
    PredictionService,
    MIN_TEXT_LENGTH,
    MAX_TEXT_LENGTH,
)


@pytest.fixture
def sample_csv_path(tmp_path):
    """Creates a temporary valid CSV dataset with 10 records across 2 categories."""
    csv_file = tmp_path / "valid_dataset.csv"
    data = {
        "Resume": [
            "Experienced Python developer building REST APIs with Flask and Django.",
            "Senior Python engineer with PostgreSQL, Redis, and Docker experience.",
            "Backend Python programmer writing unit tests and microservices.",
            "Junior Python developer skilled in web scraping and automation.",
            "Python software engineer with AWS EC2 and Docker deployment skills.",
            "Lead Data Scientist building machine learning and NLP models.",
            "Data Analyst with expertise in pandas, numpy, and statistical modeling.",
            "Machine Learning Engineer specializing in deep learning with PyTorch.",
            "Data Science specialist in churn prediction and exploratory data analysis.",
            "Senior AI Engineer with NLP, BERT, and scikit-learn experience.",
        ],
        "Category": [
            "Python Developer", "Python Developer", "Python Developer", "Python Developer", "Python Developer",
            "Data Science", "Data Science", "Data Science", "Data Science", "Data Science",
        ],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return str(csv_file)


@pytest.fixture
def trained_test_model(tmp_path, sample_csv_path):
    """Trains a quick pipeline on the temporary fixture and saves it."""
    model_path = tmp_path / "test_model.joblib"
    metrics_path = tmp_path / "metrics.json"
    train_and_evaluate(
        dataset_path=sample_csv_path,
        output_model_path=str(model_path),
        metrics_path=str(metrics_path),
        test_size=0.3,
        random_seed=42,
        compare_models=False,
    )
    return str(model_path)


# =========================================================================
# 1. Dataset Loading and Column Validation Tests
# =========================================================================

def test_detect_columns_success():
    df = pd.DataFrame({"Resume_Text": ["sample resume text"], "Category": ["Data Science"]})
    text_col, target_col = detect_columns(df)
    assert text_col == "Resume_Text"
    assert target_col == "Category"


def test_detect_columns_missing():
    df = pd.DataFrame({"Title": ["Foo"], "Body": ["Bar"]})
    with pytest.raises(ValueError, match="Dataset must contain text and category columns"):
        detect_columns(df)


def test_load_and_validate_data_success(sample_csv_path):
    df = load_and_validate_data(sample_csv_path)
    assert len(df) == 10
    assert set(df.columns) == {"Resume", "Category"}
    assert df["Category"].nunique() == 2


def test_load_and_validate_empty_dataset(tmp_path):
    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("Resume,Category\n", encoding="utf-8")
    with pytest.raises(ValueError, match="The dataset at .* is empty"):
        load_and_validate_data(str(empty_csv))


def test_load_and_validate_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_and_validate_data("non_existent_directory/fake.csv")


def test_load_and_validate_handles_nulls_and_duplicates(tmp_path):
    csv_file = tmp_path / "dirty_data.csv"
    data = {
        "Resume": [
            "Valid resume text for candidate one here.",
            None,  # Null text
            "   ",  # Whitespace text
            "Valid resume text for candidate two here.",
            "Valid resume text for candidate one here.",  # Duplicate
            "Valid resume text for candidate three here.",
            "Valid resume text for candidate four here.",
            "Valid resume text for candidate five here.",
            "Valid resume text for candidate six here.",
        ],
        "Category": [
            "Python",
            "Python",
            "Python",
            "Java",
            "Python",  # Duplicate row
            None,  # Null category
            "Java",
            "Python",
            "Java",
        ],
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)
    df = load_and_validate_data(str(csv_file))
    # Expected valid rows: candidate one, two, four, five, six -> 5 rows
    assert len(df) == 5
    assert df["Category"].nunique() == 2


def test_load_and_validate_insufficient_classes(tmp_path):
    csv_file = tmp_path / "single_class.csv"
    data = {
        "Resume": [
            "Resume one text long enough to qualify.",
            "Resume two text long enough to qualify.",
            "Resume three text long enough to qualify.",
            "Resume four text long enough to qualify.",
        ],
        "Category": ["Python", "Python", "Python", "Python"],
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)
    with pytest.raises(ValueError, match="at least 2 distinct categories"):
        load_and_validate_data(str(csv_file))


# =========================================================================
# 2. Pipeline Training, Splitting, and Serialization Tests
# =========================================================================

def test_pipeline_build():
    pipeline = build_pipeline(model_type="logistic", random_seed=42)
    assert "tfidf" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_pipeline_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_pipeline(model_type="random_forest")


def test_split_data_fallback_when_class_has_one_sample():
    df = pd.DataFrame({
        "Resume": [
            "Resume for class A 1", "Resume for class A 2",
            "Resume for class A 3", "Resume for class B 1"  # only 1 sample for B
        ],
        "Category": ["ClassA", "ClassA", "ClassA", "ClassB"],
    })
    X_train, X_test, y_train, y_test, was_stratified = split_data(df, test_size=0.25, random_seed=42)
    assert not was_stratified
    assert len(X_train) == 3
    assert len(X_test) == 1


def test_train_and_evaluate_generates_artifacts(tmp_path, sample_csv_path):
    model_output = tmp_path / "output_model.joblib"
    metrics_output = tmp_path / "output_metrics.json"

    meta = train_and_evaluate(
        dataset_path=sample_csv_path,
        output_model_path=str(model_output),
        metrics_path=str(metrics_output),
        test_size=0.3,
        random_seed=42,
        compare_models=True,
    )

    assert model_output.is_file()
    assert metrics_output.is_file()
    assert "metrics" in meta
    assert "accuracy" in meta["metrics"]
    assert meta["total_samples"] == 10
    assert meta["is_smoke_test"] is True


# =========================================================================
# 3. Prediction Service Tests
# =========================================================================

def test_prediction_service_valid_prediction(trained_test_model):
    service = PredictionService(model_path=trained_test_model)
    text = "Senior Python engineer building microservices, REST APIs, and Docker containers."
    result = service.predict(text)

    assert "predicted_category" in result
    assert result["predicted_category"] in ["Python Developer", "Data Science"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "probabilities" in result
    assert len(result["probabilities"]) == 2
    assert "top_keywords" in result
    assert isinstance(result["top_keywords"], list)
    assert result["text_length"] == len(service.clean_text(text))


def test_prediction_service_empty_text_rejection(trained_test_model):
    service = PredictionService(model_path=trained_test_model)
    with pytest.raises(ValueError, match="cannot be empty or contain only whitespace"):
        service.predict("   \n\t  ")


def test_prediction_service_short_text_rejection(trained_test_model):
    service = PredictionService(model_path=trained_test_model)
    with pytest.raises(ValueError, match="too short"):
        service.predict("Python developer")  # 16 chars < 20 chars


def test_prediction_service_missing_model(tmp_path):
    missing_path = tmp_path / "non_existent.joblib"
    service = PredictionService(model_path=str(missing_path))
    assert not service.is_model_available()
    with pytest.raises(FileNotFoundError, match="Trained model artifact not found"):
        service.predict("Valid text that is definitely long enough for testing purposes.")


def test_prediction_service_keyword_extraction(trained_test_model):
    service = PredictionService(model_path=trained_test_model)
    service.load_model()
    text = "Proficient in Python, Docker, Kubernetes, AWS, SQL, and Machine Learning."
    keywords = service.extract_keywords(text)
    assert any(kw in ["Python", "Docker", "Kubernetes", "Aws", "Sql", "Machine Learning"] for kw in keywords)
