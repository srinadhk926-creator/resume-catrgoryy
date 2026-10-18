"""
ResumeAI - Main Web Routes Blueprint
====================================
Handles landing page, health check, resume upload, model prediction,
user dashboard, and personal prediction history.
"""

import json
import logging
from datetime import datetime
from collections import Counter
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    current_app,
    g,
)

from database.db import get_db
from database.models import Prediction
from routes.auth import login_required
from services.resume_parser import (
    parse_resume_file,
    ResumeParserError,
    FileTooLargeError,
    EmptyFileError,
    UnsupportedFileTypeError,
    CorruptedFileError,
)
from services.prediction_service import get_prediction_service

logger = logging.getLogger("ResumeAI-MainRoutes")

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Landing page showcasing system overview, features, and MLOps architecture."""
    return render_template("index.html")


@main_bp.route("/health")
def health():
    """
    Health check endpoint returning JSON status and model readiness.
    Used by load balancers, container healthchecks, and practical verification.
    """
    prediction_service = get_prediction_service(current_app.config.get("MODEL_PATH"))
    model_ok = prediction_service.is_model_available()

    response_data = {
        "status": "ok" if model_ok else "unhealthy",
        "service": "resume-category-classifier",
        "model_loaded": model_ok,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    status_code = 200 if model_ok else 503
    return jsonify(response_data), status_code


@main_bp.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():
    """
    Upload and classify resume.
    GET: Renders upload page with drag-and-drop file interface.
    POST: Processes PDF/DOCX file, predicts category, and renders result.
    """
    if request.method == "GET":
        return render_template("upload.html")

    # POST method: Handle resume upload or direct text input
    uploaded_file = request.files.get("resume_file") or request.files.get("file")
    raw_text = request.form.get("resume_text", "").strip()

    if (not uploaded_file or uploaded_file.filename.strip() == "") and not raw_text:
        flash("Please upload a PDF/DOCX resume file or enter resume text.", "danger")
        return render_template("upload.html"), 400

    max_size = current_app.config.get("MAX_CONTENT_LENGTH", 5 * 1024 * 1024)

    # 1. Obtain extracted text
    if uploaded_file and uploaded_file.filename.strip() != "":
        try:
            extracted_text, safe_filename = parse_resume_file(
                file_stream_or_bytes=uploaded_file.stream,
                filename=uploaded_file.filename,
                max_file_size=max_size,
            )
        except FileTooLargeError as err:
            flash(str(err), "danger")
            return render_template("upload.html"), 413
        except (UnsupportedFileTypeError, EmptyFileError, CorruptedFileError, ResumeParserError) as err:
            flash(str(err), "danger")
            return render_template("upload.html"), 400
        except Exception as err:
            logger.error(f"Unexpected file parsing error: {err}")
            flash("Failed to process the uploaded file. Please ensure it is a valid PDF or Word document.", "danger")
            return render_template("upload.html"), 500
    else:
        extracted_text = raw_text
        safe_filename = "direct_text_input"

    # 2. Run model inference
    prediction_service = get_prediction_service(current_app.config.get("MODEL_PATH"))
    try:
        prediction_result = prediction_service.predict(extracted_text)
    except FileNotFoundError as err:
        flash(f"Model artifact is missing: {err}. Please run: python train_model.py", "danger")
        return render_template("upload.html"), 503
    except ValueError as err:
        flash(str(err), "danger")
        return render_template("upload.html"), 400
    except Exception as err:
        logger.error(f"Prediction inference failed: {err}")
        flash("Model prediction failed. Please check the resume text format.", "danger")
        return render_template("upload.html"), 500

    # 3. Persist prediction in database if user is authenticated
    db = get_db()
    saved_record_id = None
    if g.user is not None:
        try:
            pred_record = Prediction(
                user_id=g.user.id,
                original_filename=safe_filename,
                predicted_category=prediction_result["predicted_category"],
                confidence=prediction_result["confidence"],
                extracted_text_preview=prediction_result["text_preview"],
                probabilities_json=json.dumps(prediction_result.get("probabilities", [])),
                top_keywords_json=json.dumps(prediction_result.get("top_keywords", [])),
                text_length=prediction_result.get("text_length", len(extracted_text)),
            )
            db.add(pred_record)
            db.commit()
            saved_record_id = pred_record.id
        except Exception as err:
            db.rollback()
            logger.warning(f"Could not save prediction to database: {err}")

    return render_template(
        "result.html",
        result=prediction_result,
        filename=safe_filename,
        record_id=saved_record_id,
        saved_to_history=bool(g.user is not None),
    )


@main_bp.route("/dashboard")
@login_required
def dashboard():
    """Displays user dashboard with metrics, latest report, and recent activity."""
    db = get_db()
    user_predictions = (
        db.query(Prediction)
        .filter_by(user_id=g.user.id)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .all()
    )

    total_predictions = len(user_predictions)
    latest_prediction = user_predictions[0] if user_predictions else None
    recent_predictions = user_predictions[:5]

    # Calculate top predicted categories
    categories = [p.predicted_category for p in user_predictions]
    category_counts = Counter(categories).most_common(5)

    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        latest_prediction=latest_prediction,
        recent_predictions=recent_predictions,
        category_counts=category_counts,
    )


@main_bp.route("/history")
@login_required
def history():
    """Lists all prediction history records for the authenticated user."""
    db = get_db()
    predictions = (
        db.query(Prediction)
        .filter_by(user_id=g.user.id)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .all()
    )
    return render_template("history.html", predictions=predictions)


@main_bp.route("/reports/<int:prediction_id>")
@login_required
def report_detail(prediction_id: int):
    """
    Renders the persistent analysis report for a specific saved prediction.
    Strictly verifies ownership to prevent unauthorized access.
    """
    db = get_db()
    pred = db.query(Prediction).filter_by(id=prediction_id).first()

    if pred is None:
        flash("The requested analysis report was not found.", "warning")
        return redirect(url_for("main.history"))

    # Strict authorization: User can only view their own reports
    if pred.user_id != g.user.id:
        logger.warning(
            f"Unauthorized report access attempt: User {g.user.id} tried to view report {prediction_id} belonging to user {pred.user_id}"
        )
        return jsonify({"error": "Forbidden. You cannot view another user's report."}), 403

    # Reconstruct result object from stored database record
    result = {
        "predicted_category": pred.predicted_category,
        "confidence": pred.confidence,
        "confidence_percentage": pred.confidence_percentage,
        "is_low_confidence": pred.is_low_confidence,
        "confidence_warning": pred.confidence_warning,
        "margin": pred.margin,
        "probabilities": pred.probabilities,
        "top_keywords": pred.top_keywords,
        "text_length": pred.text_length or (len(pred.extracted_text_preview) if pred.extracted_text_preview else 0),
        "text_preview": pred.extracted_text_preview or "",
        "timestamp": pred.created_at.isoformat() + "Z",
    }

    return render_template(
        "report.html",
        result=result,
        prediction=pred,
        filename=pred.original_filename,
        record_id=pred.id,
    )


@main_bp.route("/history/delete/<int:prediction_id>", methods=["POST"])
@login_required
def delete_prediction(prediction_id: int):
    """Deletes a prediction record with strict ownership authorization."""
    db = get_db()
    pred = db.query(Prediction).filter_by(id=prediction_id).first()

    if pred is None:
        flash("Record not found.", "warning")
        return redirect(url_for("main.history"))

    # Enforce strict ownership check
    if pred.user_id != g.user.id:
        logger.warning(f"Unauthorized deletion attempt on prediction {prediction_id} by user {g.user.id}")
        return jsonify({"error": "Forbidden. You cannot delete another user's prediction."}), 403

    try:
        db.delete(pred)
        db.commit()
        flash("Prediction record removed from your history.", "success")
    except Exception as err:
        db.rollback()
        logger.error(f"Error deleting prediction {prediction_id}: {err}")
        flash("Failed to delete the record.", "danger")

    return redirect(url_for("main.history"))


@main_bp.route("/profile")
@login_required
def profile():
    """User profile overview showing account details and classification metrics."""
    db = get_db()
    total_predictions = db.query(Prediction).filter_by(user_id=g.user.id).count()
    first_prediction = (
        db.query(Prediction)
        .filter_by(user_id=g.user.id)
        .order_by(Prediction.created_at.asc())
        .first()
    )

    return render_template(
        "profile.html",
        user=g.user,
        total_predictions=total_predictions,
        first_prediction=first_prediction,
    )
