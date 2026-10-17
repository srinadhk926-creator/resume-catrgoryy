"""
Tests for user authentication, registration, login, logout, and route protection.
"""

import pytest
from database.db import get_db
from database.models import User


def test_register_success(client, app):
    response = client.post("/register", data={
        "name": "Alice Developer",
        "email": "alice@example.com",
        "password": "mypassword123",
        "confirm_password": "mypassword123",
    }, follow_redirects=True)

    assert response.status_code == 200
    assert b"Welcome to ResumeAI" in response.data or b"Dashboard" in response.data

    with app.app_context():
        db = get_db()
        user = db.query(User).filter_by(email="alice@example.com").first()
        assert user is not None
        assert user.name == "Alice Developer"
        assert user.check_password("mypassword123")
        assert not user.check_password("wrongpassword")


def test_register_duplicate_email(client, app):
    # Register first time
    client.post("/register", data={
        "name": "Bob Original",
        "email": "bob@example.com",
        "password": "password123",
        "confirm_password": "password123",
    })

    # Clear session so user is logged out
    with client.session_transaction() as sess:
        sess.clear()

    # Register second time with same email
    response = client.post("/register", data={
        "name": "Bob Duplicate",
        "email": "bob@example.com",
        "password": "password123",
        "confirm_password": "password123",
    })
    assert response.status_code == 400
    assert b"account with that email already exists" in response.data


def test_register_password_mismatch(client):
    response = client.post("/register", data={
        "name": "Charlie",
        "email": "charlie@example.com",
        "password": "password123",
        "confirm_password": "different123",
    })
    assert response.status_code == 400
    assert b"Passwords do not match" in response.data


def test_register_short_password(client):
    response = client.post("/register", data={
        "name": "Dave",
        "email": "dave@example.com",
        "password": "123",
        "confirm_password": "123",
    })
    assert response.status_code == 400
    assert b"Password must be at least 6 characters" in response.data


def test_register_invalid_email(client):
    response = client.post("/register", data={
        "name": "Eve",
        "email": "notanemail",
        "password": "password123",
        "confirm_password": "password123",
    })
    assert response.status_code == 400
    assert b"valid email address" in response.data


def test_login_success(client, app):
    # Setup user
    with app.app_context():
        db = get_db()
        user = User(name="Frank", email="frank@example.com")
        user.set_password("frankpass123")
        db.add(user)
        db.commit()

    response = client.post("/login", data={
        "email": "frank@example.com",
        "password": "frankpass123",
    }, follow_redirects=True)
    assert response.status_code == 200
    assert b"Welcome back" in response.data


def test_login_invalid_password(client, app):
    with app.app_context():
        db = get_db()
        user = User(name="Grace", email="grace@example.com")
        user.set_password("gracepass123")
        db.add(user)
        db.commit()

    response = client.post("/login", data={
        "email": "grace@example.com",
        "password": "wrongpassword",
    })
    assert response.status_code == 401
    assert b"Invalid email or password" in response.data


def test_login_nonexistent_user(client):
    response = client.post("/login", data={
        "email": "nobody@example.com",
        "password": "password123",
    })
    assert response.status_code == 401
    assert b"Invalid email or password" in response.data


def test_logout(client):
    # Set a session
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["user_name"] = "LoggedInUser"

    response = client.post("/logout", follow_redirects=True)
    assert response.status_code == 200
    assert b"signed out successfully" in response.data

    with client.session_transaction() as sess:
        assert "user_id" not in sess


def test_protected_routes_require_login(client):
    for route in ["/dashboard", "/history", "/prediction", "/profile", "/reports/1"]:
        resp = client.get(route)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]
        assert "next=" in resp.headers["Location"]


def test_login_safe_next_redirect(client, app):
    with app.app_context():
        db = get_db()
        user = User(name="RedirectUser", email="redirect@example.com")
        user.set_password("pass12345")
        db.add(user)
        db.commit()

    # Safe relative next redirect
    resp = client.post("/login?next=%2Fprediction", data={
        "email": "redirect@example.com",
        "password": "pass12345",
    })
    assert resp.status_code == 302
    assert resp.headers["Location"] == "/prediction"

    # Unsafe external next redirect should fallback to dashboard
    resp_unsafe = client.post("/login?next=https%3A%2F%2Fevil.com", data={
        "email": "redirect@example.com",
        "password": "pass12345",
    })
    assert resp_unsafe.status_code == 302
    assert "/dashboard" in resp_unsafe.headers["Location"]
