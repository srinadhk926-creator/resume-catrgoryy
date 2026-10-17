"""
ResumeAI - Database Engine & Session Management
===============================================
SQLAlchemy 2.0 configuration with thread-safe scoped sessions.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

Base = declarative_base()

_engine = None
_current_db_url = None
_SessionFactory = None
db_session = None


def normalize_sqlite_url(database_url: str) -> str:
    """Ensures SQLite parent directories exist and formats POSIX paths."""
    if not database_url.startswith("sqlite"):
        return database_url

    if ":memory:" in database_url:
        return database_url

    # Format: sqlite:///C:/path/to/db.sqlite
    prefix = "sqlite:///"
    if database_url.startswith(prefix):
        raw_path = database_url[len(prefix):]
        p = Path(raw_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{p.resolve().as_posix()}"

    return database_url


def get_engine(database_url: str = None):
    """Creates or returns the active SQLAlchemy engine."""
    global _engine, _current_db_url

    if database_url is None:
        db_dir = Path(__file__).resolve().parent.parent / "instance"
        db_dir.mkdir(parents=True, exist_ok=True)
        database_url = f"sqlite:///{(db_dir / 'resume_ai.db').as_posix()}"

    database_url = normalize_sqlite_url(database_url)

    # If engine doesn't exist or URL changed (e.g. switching to test database), recreate
    if _engine is None or _current_db_url != database_url:
        if _engine is not None:
            _engine.dispose()

        connect_args = {}
        if database_url.startswith("sqlite"):
            connect_args = {"check_same_thread": False}

        _engine = create_engine(database_url, connect_args=connect_args, pool_pre_ping=True)
        _current_db_url = database_url

    return _engine


def init_db(database_url: str = None, create_tables: bool = True):
    """Initializes engine, sessionmaker, scoped session, and creates schema."""
    global _engine, _SessionFactory, db_session
    _engine = get_engine(database_url)
    _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    db_session = scoped_session(_SessionFactory)

    if create_tables:
        from database.models import User, Prediction  # Ensure models are imported
        Base.metadata.create_all(bind=_engine)

    return db_session


def get_db():
    """Returns the thread-local scoped database session."""
    global db_session
    if db_session is None:
        init_db()
    return db_session


def close_db_session(exception=None):
    """Closes the current thread's database session (attached to Flask teardown)."""
    global db_session
    if db_session is not None:
        db_session.remove()
