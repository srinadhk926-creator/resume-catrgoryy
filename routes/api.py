"""
ResumeAI - REST API Blueprint
=============================
Provides programmatic endpoints for model prediction and automated integrations.
"""

import logging
from flask import Blueprint, request, jsonify, current_app

from services.resume_parser import (
    parse_resume_file,
    FileTooLargeError,
    EmptyFileError,
    UnsupportedFileTypeError,
    CorruptedFileError,
)
from services.prediction_service import get_prediction_service

logger = logging.getLogger("ResumeAI-API")

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/predict", methods=["POST"])
def predict():
    """
    Programmatic inference endpoint.
    Accepts:
      - multipart/form-data with 'file' or 'resume_file' (PDF or DOCX)
      - application/json with {"text": "resume content here..."}
      - application/x-www-form-urlencoded with 'text'
    """
    extracted_text = None
    filename = "direct_text_input"

    # 1. Check for multipart file upload
    if request.files and ("file" in request.files or "resume_file" in request.files):
        uploaded_file = request.files.get("file") or request.files.get("resume_file")
        if not uploaded_file or uploaded_file.filename.strip() == "":
            return jsonify({"status": "error", "message": "Uploaded file cannot be empty."}), 400

        max_size = current_app.config.get("MAX_CONTENT_LENGTH", 5 * 1024 * 1024)
        try:
            extracted_text, filename = parse_resume_file(
                file_stream_or_bytes=uploaded_file.stream,
                filename=uploaded_file.filename,
                max_file_size=max_size,
            )
        except FileTooLargeError as err:
            return jsonify({"status": "error", "message": str(err)}), 413
        except UnsupportedFileTypeError as err:
            return jsonify({"status": "error", "message": str(err)}), 415
        except (EmptyFileError, CorruptedFileError) as err:
            return jsonify({"status": "error", "message": str(err)}), 400
        except Exception as err:
            logger.error(f"API file parsing error: {err}")
            return jsonify({"status": "error", "message": "Internal error processing uploaded document."}), 500

    # 2. Check for JSON payload
    elif request.is_json:
        data = request.get_json(silent=True)
        if not data or "text" not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'text' field in JSON request body."
            }), 400
        extracted_text = data.get("text")

    # 3. Check for form text payload
    elif "text" in request.form:
        extracted_text = request.form.get("text")

    else:
        return jsonify({
            "status": "error",
            "message": "Invalid request format. Send a PDF/DOCX file as multipart/form-data, or JSON with a 'text' field."
        }), 400

    # 4. Perform model inference
    prediction_service = get_prediction_service(current_app.config.get("MODEL_PATH"))
    try:
        result = prediction_service.predict(extracted_text)
    except FileNotFoundError as err:
        return jsonify({
            "status": "error",
            "message": f"Model artifact unavailable: {err}. Please run: python train_model.py"
        }), 503
    except ValueError as err:
        return jsonify({"status": "error", "message": str(err)}), 400
    except Exception as err:
        logger.error(f"API prediction error: {err}")
        return jsonify({"status": "error", "message": "Inference execution failed."}), 500

    # Format JSON response
    response_payload = {
        "status": "success",
        "filename": filename,
        "predicted_category": result["predicted_category"],
        "confidence": result["confidence"],
        "confidence_percentage": result["confidence_percentage"],
        "is_low_confidence": result.get("is_low_confidence", False),
        "confidence_warning": result.get("confidence_warning"),
        "margin": result.get("margin", result["confidence"]),
        "probabilities": result["probabilities"],
        "top_keywords": result["top_keywords"],
        "text_length": result["text_length"],
        "timestamp": result["timestamp"],
    }
    return jsonify(response_payload), 200
