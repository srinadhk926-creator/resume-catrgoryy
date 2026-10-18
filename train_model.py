#!/usr/bin/env python3
"""
ResumeAI - Machine Learning Training Pipeline
=============================================
Trains a resume classification model using TF-IDF vectorization and a 
scikit-learn classifier (Logistic Regression with optional Naive Bayes comparison).

Usage:
    python train_model.py --dataset data/sample_resumes.csv --output model/resume_classifier.joblib
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ResumeAI-Trainer")

# General resume boilerplate stop words that appear across domains
RESUME_STOP_WORDS = list(ENGLISH_STOP_WORDS.union({
    "curriculum", "vitae", "resume", "contact", "phone", "email", "linkedin", "github",
    "address", "city", "state", "university", "college", "school", "bachelor", "btech",
    "degree", "gpa", "cgpa", "year", "years", "duration", "date", "present",
    "experience", "work", "job", "project", "projects", "summary", "objective", "profile",
    "skills", "technical", "personal", "details", "responsibilities", "responsible",
    "developed", "implemented", "worked", "created", "built", "using", "used", "team",
    "member", "members", "month", "months", "role", "title", "company", "organization"
}))


TEXT_COLUMN_CANDIDATES = ["Resume", "Resume_Text", "resume_text", "text", "resume", "ResumeText"]
TARGET_COLUMN_CANDIDATES = ["Category", "category", "label", "Job_Title", "job_title", "target"]


def detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Detects text and category columns from candidate column names."""
    text_col = next((c for c in TEXT_COLUMN_CANDIDATES if c in df.columns), None)
    target_col = next((c for c in TARGET_COLUMN_CANDIDATES if c in df.columns), None)

    if not text_col or not target_col:
        raise ValueError(
            f"Dataset must contain text and category columns. Found: {list(df.columns)}. "
            f"Expected text column in {TEXT_COLUMN_CANDIDATES} and category column in {TARGET_COLUMN_CANDIDATES}."
        )
    return text_col, target_col


def load_and_validate_data(dataset_path: str) -> pd.DataFrame:
    """Loads CSV, validates columns, cleans missing values and duplicates."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {dataset_path}")

    logger.info(f"Loading dataset from: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(dataset_path, encoding="latin-1")

    if df.empty:
        raise ValueError(f"The dataset at {dataset_path} is empty.")

    text_col, target_col = detect_columns(df)
    initial_count = len(df)
    logger.info(f"Detected text column='{text_col}' and target column='{target_col}'. Initial records: {initial_count}")

    # Standardize column names
    df = df.rename(columns={text_col: "Resume", target_col: "Category"})

    # Clean missing values
    df = df.dropna(subset=["Resume", "Category"])
    # Strip whitespace and remove empty strings
    df["Resume"] = df["Resume"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()
    df = df[(df["Resume"].str.len() > 0) & (df["Category"].str.len() > 0)]

    missing_cleaned = initial_count - len(df)
    if missing_cleaned > 0:
        logger.info(f"Dropped {missing_cleaned} empty/null rows.")

    # Remove duplicates
    dup_count = df.duplicated(subset=["Resume", "Category"]).sum()
    if dup_count > 0:
        df = df.drop_duplicates(subset=["Resume", "Category"]).reset_index(drop=True)
        logger.info(f"Removed {dup_count} duplicate records. Unique records remaining: {len(df)}")

    unique_classes = df["Category"].nunique()
    if unique_classes < 2:
        raise ValueError(f"Dataset must contain at least 2 distinct categories. Found only {unique_classes}.")

    if len(df) < 4:
        raise ValueError(f"Dataset too small to train. Found only {len(df)} valid records.")

    logger.info(f"Dataset validated successfully: {len(df)} records across {unique_classes} categories.")
    logger.info("Category distribution:\n" + df["Category"].value_counts().to_string())

    return df


def build_pipeline(model_type: str = "logistic", random_seed: int = 42) -> Pipeline:
    """Builds a scikit-learn Pipeline with TF-IDF and specified classifier."""
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words=RESUME_STOP_WORDS,
        max_df=0.85,
        min_df=1,
    )

    if model_type == "logistic":
        classifier = LogisticRegression(
            C=3.0,
            max_iter=1000,
            random_state=random_seed,
            class_weight="balanced",
        )
    elif model_type == "naive_bayes":
        classifier = MultinomialNB(alpha=0.1)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'logistic' or 'naive_bayes'.")

    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def split_data(
    df: pd.DataFrame, 
    test_size: float = 0.25, 
    random_seed: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, bool]:
    """Splits data with stratification if feasible, falling back gracefully if not."""
    X = df["Resume"]
    y = df["Category"]

    class_counts = y.value_counts()
    min_count = class_counts.min()

    # Stratified split requires each class to have enough instances in both splits
    use_stratify = True
    if min_count < 2:
        use_stratify = False
        logger.warning(
            f"Smallest class has only {min_count} record(s). "
            f"Stratified split disabled; using random split instead."
        )
    else:
        # Check if train_test_split with stratify works
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_seed, stratify=y
            )
            return X_train, X_test, y_train, y_test, True
        except ValueError as err:
            logger.warning(
                f"Stratification not possible with current class distributions ({err}). "
                f"Falling back to unstratified split."
            )
            use_stratify = False

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=None
    )
    return X_train, X_test, y_train, y_test, False


def evaluate_pipeline(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
    """Calculates accuracy, precision, recall, F1-score, and classification report."""
    y_pred = pipeline.predict(X_test)
    classes = getattr(pipeline, "classes_", sorted(list(y_test.unique())))

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report_str = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
        "f1_weighted": round(f1, 4),
        "classification_report": report_dict,
        "classification_report_text": report_str,
        "confusion_matrix": cm,
        "confusion_matrix_labels": list(classes),
    }


def train_and_evaluate(
    dataset_path: str,
    output_model_path: str,
    metrics_path: Optional[str] = None,
    test_size: float = 0.25,
    random_seed: int = 42,
    compare_models: bool = True,
) -> Dict[str, Any]:
    """Full training pipeline with validation, evaluation, and serialization."""
    df = load_and_validate_data(dataset_path)
    total_samples = len(df)
    is_smoke_test = total_samples < 100

    if is_smoke_test:
        logger.warning(
            f"[SMOKE-TEST NOTICE] Training on small sample dataset ({total_samples} records). "
            f"Metrics are for pipeline verification and code testing only. "
            f"For production accuracy, train on Kaggle's UpdatedResumeDataSet.csv (900+ resumes)."
        )

    X_train, X_test, y_train, y_test, was_stratified = split_data(
        df, test_size=test_size, random_seed=random_seed
    )
    logger.info(
        f"Data split: Train={len(X_train)} samples, Test={len(X_test)} samples "
        f"(Stratified={was_stratified})"
    )

    # 1. Train primary Logistic Regression pipeline
    logger.info("Training Pipeline (TF-IDF + Logistic Regression)...")
    logistic_pipeline = build_pipeline(model_type="logistic", random_seed=random_seed)
    # Fit preprocessing and classifier exclusively on train data
    logistic_pipeline.fit(X_train, y_train)
    logistic_metrics = evaluate_pipeline(logistic_pipeline, X_test, y_test)

    best_pipeline = logistic_pipeline
    best_model_name = "Logistic Regression"
    chosen_metrics = logistic_metrics

    # 2. Optional Naive Bayes comparison
    nb_metrics = None
    if compare_models:
        logger.info("Training Comparison Pipeline (TF-IDF + Multinomial Naive Bayes)...")
        nb_pipeline = build_pipeline(model_type="naive_bayes", random_seed=random_seed)
        nb_pipeline.fit(X_train, y_train)
        nb_metrics = evaluate_pipeline(nb_pipeline, X_test, y_test)

        logger.info("=== Model Comparison on Test Set ===")
        logger.info(f"Logistic Regression: Accuracy={logistic_metrics['accuracy']:.4f}, F1={logistic_metrics['f1_weighted']:.4f}")
        logger.info(f"Multinomial NB:      Accuracy={nb_metrics['accuracy']:.4f}, F1={nb_metrics['f1_weighted']:.4f}")

        # Choose the best or keep Logistic Regression as primary
        if nb_metrics["f1_weighted"] > logistic_metrics["f1_weighted"]:
            best_pipeline = nb_pipeline
            best_model_name = "Multinomial Naive Bayes"
            chosen_metrics = nb_metrics
            logger.info(f"Selecting {best_model_name} as the superior model.")
        else:
            logger.info("Selecting Logistic Regression as the primary model.")

    logger.info(f"\nEvaluation Report ({best_model_name}):\n" + chosen_metrics["classification_report_text"])

    # 3. Serialize model
    output_path = Path(output_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, output_path)
    logger.info(f"Model pipeline successfully saved to: {output_path.resolve()}")

    # 4. Serialize metrics metadata
    classes = sorted(list(df["Category"].unique()))
    metadata = {
        "dataset_path": str(dataset_path),
        "total_samples": total_samples,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "categories_count": len(classes),
        "categories": classes,
        "selected_model": best_model_name,
        "is_smoke_test": is_smoke_test,
        "was_stratified": was_stratified,
        "random_seed": random_seed,
        "metrics": {
            "accuracy": chosen_metrics["accuracy"],
            "precision_weighted": chosen_metrics["precision_weighted"],
            "recall_weighted": chosen_metrics["recall_weighted"],
            "f1_weighted": chosen_metrics["f1_weighted"],
        },
        "classification_report": chosen_metrics["classification_report"],
        "confusion_matrix": chosen_metrics.get("confusion_matrix", []),
        "confusion_matrix_labels": chosen_metrics.get("confusion_matrix_labels", classes),
        "comparison": {
            "logistic_regression": {
                "accuracy": logistic_metrics["accuracy"],
                "f1_weighted": logistic_metrics["f1_weighted"],
            },
            "naive_bayes": {
                "accuracy": nb_metrics["accuracy"] if nb_metrics else None,
                "f1_weighted": nb_metrics["f1_weighted"] if nb_metrics else None,
            } if nb_metrics else None,
        },
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }

    if not metrics_path:
        metrics_path = str(output_path.parent / "evaluation_metrics.json")
    
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Evaluation metrics JSON saved to: {Path(metrics_path).resolve()}")

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="ResumeAI: Train and serialize resume classification pipeline."
    )
    parser.add_argument(
        "--dataset", "--data",
        dest="dataset",
        type=str,
        default="data/sample_resumes.csv",
        help="Path to training CSV dataset (default: data/sample_resumes.csv)",
    )
    parser.add_argument(
        "--output", "--model-path",
        dest="output",
        type=str,
        default="model/resume_classifier.joblib",
        help="Path to save serialized model (default: model/resume_classifier.joblib)",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=None,
        help="Path to save evaluation metrics JSON (default: model/evaluation_metrics.json)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test set split ratio (default: 0.25)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Disable comparison with Naive Bayes (train only Logistic Regression)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        train_and_evaluate(
            dataset_path=args.dataset,
            output_model_path=args.output,
            metrics_path=args.metrics_path,
            test_size=args.test_size,
            random_seed=args.random_seed,
            compare_models=not args.no_compare,
        )
        print("\n[SUCCESS] Model training and serialization completed successfully.")
        sys.exit(0)
    except Exception as err:
        logger.error(f"[TRAINING FAILED] {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
