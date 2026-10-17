"""
ResumeAI - Application Factory & Entrypoint
===========================================
Initializes Flask app, registers blueprints, configures CSRF protection,
and manages database session lifecycle.
"""

import os
import logging
from flask import Flask, render_template, jsonify, request, g
from flask_wtf.csrf import CSRFProtect

from config import config_by_name
from database.db import init_db, close_db_session
from routes.main import main_bp
from routes.auth import auth_bp
from routes.api import api_bp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ResumeAI-App")

csrf = CSRFProtect()


def create_app(config_name: str = None) -> Flask:
    """Application factory for ResumeAI Flask application."""
    if config_name is None:
        config_name = os.getenv("FLASK_ENV", "development")

    config_class = config_by_name.get(config_name, config_by_name["default"])

    app = Flask(__name__)
    app.config.from_object(config_class)

    # Initialize CSRF protection (exempt API endpoints)
    csrf.init_app(app)
    csrf.exempt(api_bp)

    # Initialize database engine and scoped session
    db_uri = app.config.get("SQLALCHEMY_DATABASE_URI")
    init_db(database_url=db_uri, create_tables=True)

    # Register session cleanup teardown
    app.teardown_appcontext(close_db_session)

    # Context processor to make logged-in user available to all Jinja templates
    @app.context_processor
    def inject_user():
        return {"current_user": getattr(g, "user", None)}

    # Register Blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp)

    # Global Error Handlers
    @app.errorhandler(400)
    def bad_request_error(error):
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({"status": "error", "message": "Bad request"}), 400
        return render_template("404.html", error_code=400, error_message="Bad Request"), 400

    @app.errorhandler(403)
    def forbidden_error(error):
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({"status": "error", "message": "Forbidden access"}), 403
        return render_template("404.html", error_code=403, error_message="Access Forbidden"), 403

    @app.errorhandler(404)
    def not_found_error(error):
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({"status": "error", "message": "Resource not found"}), 404
        return render_template("404.html", error_code=404, error_message="Page Not Found"), 404

    @app.errorhandler(413)
    def request_entity_too_large(error):
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({"status": "error", "message": "File exceeds maximum upload size (5MB)"}), 413
        return render_template("404.html", error_code=413, error_message="Uploaded File Too Large (Max 5MB)"), 413

    @app.errorhandler(500)
    def internal_server_error(error):
        logger.error(f"Internal Server Error: {error}")
        if request.path.startswith("/api/") or request.is_json:
            return jsonify({"status": "error", "message": "An internal server error occurred"}), 500
        return render_template("404.html", error_code=500, error_message="Internal Server Error"), 500

    logger.info(f"ResumeAI initialized successfully in '{config_name}' environment.")
    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = app.config.get("DEBUG", False)
    logger.info(f"Starting ResumeAI server on http://{host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)
