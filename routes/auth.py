"""
ResumeAI - Authentication Blueprint
===================================
Handles user registration, login, logout, and route authorization.
"""

import re
import functools
from flask import (
    Blueprint,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    jsonify,
    g,
)

from database.db import get_db
from database.models import User

auth_bp = Blueprint("auth", __name__)


def login_required(view):
    """Decorator to require authenticated user session for view functions."""
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if "user_id" not in session or getattr(g, "user", None) is None:
            if request.is_json or request.path.startswith("/api/"):
                return jsonify({"error": "Unauthorized. Please log in first."}), 401
            flash("Please sign in to access this page.", "warning")
            return redirect(url_for("auth.login", next=request.path))
        return view(**kwargs)
    return wrapped_view


@auth_bp.before_app_request
def load_logged_in_user():
    """Loads logged-in user into flask.g for easy template access."""
    user_id = session.get("user_id")
    if user_id is None:
        g.user = None
    else:
        db = get_db()
        g.user = db.query(User).filter_by(id=user_id).first()


def validate_email_format(email: str) -> bool:
    """Validates basic email format."""
    return bool(re.match(r"^[\w\.\+\-]+@[a-zA-Z0-9\-]+(\.[a-zA-Z0-9\-]+)+$", email.strip()))


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """User registration route."""
    if g.user is not None:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        errors = []
        if not name or len(name) < 2:
            errors.append("Please enter your full name (minimum 2 characters).")
        if not email or not validate_email_format(email):
            errors.append("Please enter a valid email address.")
        if not password or len(password) < 6:
            errors.append("Password must be at least 6 characters long.")
        if password != confirm_password:
            errors.append("Passwords do not match.")

        if errors:
            for error in errors:
                flash(error, "danger")
            return render_template("register.html", name=name, email=email), 400

        db = get_db()
        existing_user = db.query(User).filter_by(email=email).first()
        if existing_user is not None:
            flash("An account with that email already exists. Please log in.", "warning")
            return render_template("register.html", name=name, email=email), 400

        new_user = User(name=name, email=email)
        new_user.set_password(password)

        try:
            db.add(new_user)
            db.commit()
            # Automatically establish user session
            session.clear()
            session["user_id"] = new_user.id
            session["user_name"] = new_user.name
            session["user_email"] = new_user.email
            flash(f"Welcome to ResumeAI, {new_user.name}! Your account has been created.", "success")
            return redirect(url_for("main.dashboard"))
        except Exception as err:
            db.rollback()
            flash("An unexpected error occurred while creating your account. Please try again.", "danger")
            return render_template("register.html", name=name, email=email), 500

    return render_template("register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """User login route."""
    if g.user is not None:
        return redirect(url_for("main.dashboard"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email and password are required.", "danger")
            return render_template("login.html", email=email), 400

        db = get_db()
        user = db.query(User).filter_by(email=email).first()

        if user is None or not user.check_password(password):
            flash("Invalid email or password. Please verify your credentials.", "danger")
            return render_template("login.html", email=email), 401

        session.clear()
        session["user_id"] = user.id
        session["user_name"] = user.name
        session["user_email"] = user.email

        flash(f"Welcome back, {user.name}!", "success")
        next_url = request.args.get("next")
        # Validate that next_url is a relative path to prevent open redirect vulnerabilities
        if next_url and next_url.startswith("/") and not next_url.startswith("//"):
            return redirect(next_url)
        return redirect(url_for("main.dashboard"))

    return render_template("login.html")


@auth_bp.route("/logout", methods=["GET", "POST"])
def logout():
    """Clears user session and logs out."""
    session.clear()
    flash("You have been signed out successfully.", "info")
    return redirect(url_for("main.index"))
