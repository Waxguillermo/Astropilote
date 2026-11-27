"""SQLAlchemy 2 engine, SQLite WAL pragmas, PostgreSQL pooling, and session helpers."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from astropilote.config import settings
from astropilote.db.models import Base, User

LOCAL_USER_ID = "local-solo"


def _attach_sqlite_pragmas(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()


def create_db_engine(database_url: str | None = None) -> Engine:
    """Create a configured SQLAlchemy 2 engine for SQLite WAL or PostgreSQL."""
    url = (database_url or settings.resolved_database_url()).strip()
    if url.startswith("sqlite"):
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            future=True,
        )
        _attach_sqlite_pragmas(engine)
        return engine
    return create_engine(url, pool_pre_ping=True, future=True)


@lru_cache(maxsize=8)
def get_engine(database_url: str | None = None) -> Engine:
    """Return a cached SQLAlchemy engine for `database_url`."""
    resolved = (database_url or settings.resolved_database_url()).strip()
    return create_db_engine(resolved)


def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    """Return a `sessionmaker` bound to `database_url`."""
    return sessionmaker(bind=get_engine(database_url), autoflush=False, expire_on_commit=False)


def init_db(database_url: str | None = None) -> Engine:
    """Create all tables idempotently and return the engine."""
    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)
    return engine


@contextmanager
def db_session(database_url: str | None = None) -> Generator[Session, None, None]:
    """Transactional context manager yielding a SQLAlchemy `Session`."""
    factory = get_session_factory(database_url)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a request-scoped SQLAlchemy `Session`."""
    init_db()
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ensure_local_user(session: Session) -> User:
    """Return the auto-provisioned solo local user (`AUTH_MODE=local`)."""
    user = session.get(User, LOCAL_USER_ID)
    if user is None:
        user = User(
            id=LOCAL_USER_ID,
            email=None,
            display_name="Local Observer",
            provider="local",
            provider_subject="local-solo",
        )
        session.add(user)
        session.flush()
    return user
