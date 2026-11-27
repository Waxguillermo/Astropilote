"""SQLAlchemy 2 ORM models for Astropilote (SQLite WAL locally, PostgreSQL in cloud)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _uuid() -> str:
    return uuid4().hex


class Base(DeclarativeBase):
    """Shared SQLAlchemy 2 declarative base."""


class User(Base):
    """Application user (auto-provisioned solo local profile or authenticated cloud account)."""

    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("provider", "provider_subject", name="uq_users_provider_subject"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    email: Mapped[str | None] = mapped_column(String(255), unique=True, nullable=True, index=True)
    display_name: Mapped[str] = mapped_column(String(120), default="Observer")
    provider: Mapped[str] = mapped_column(String(32), default="local")
    provider_subject: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    api_keys: Mapped[list[UserApiKey]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    rigs: Mapped[list[Rig]] = relationship(back_populates="user", cascade="all, delete-orphan")
    sites: Mapped[list[Site]] = relationship(back_populates="user", cascade="all, delete-orphan")
    sessions: Mapped[list[ObservingSession]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    catalog_progress: Mapped[list[CatalogProgress]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    chat_threads: Mapped[list[ChatThread]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class UserApiKey(Base):
    """Fernet-encrypted BYOK provider credentials scoped per user."""

    __tablename__ = "user_api_keys"
    __table_args__ = (UniqueConstraint("user_id", "provider", name="uq_user_api_keys_provider"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    provider: Mapped[str] = mapped_column(String(32))  # anthropic | openrouter | astrometry
    encrypted_key: Mapped[str] = mapped_column(Text)
    key_hint: Mapped[str] = mapped_column(String(32), default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    user: Mapped[User] = relationship(back_populates="api_keys")


class Rig(Base):
    """Persistent optical setup (telescope/lens + sensor + mount) owned by a user."""

    __tablename__ = "rigs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(120))
    preset_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    focal_length_mm: Mapped[float] = mapped_column(Float, default=250.0)
    aperture_mm: Mapped[float | None] = mapped_column(Float, nullable=True)
    sensor_format: Mapped[str] = mapped_column(String(48), default="APS-C")
    sensor_width_mm: Mapped[float] = mapped_column(Float, default=23.5)
    sensor_height_mm: Mapped[float] = mapped_column(Float, default=15.6)
    mount_type: Mapped[str] = mapped_column(String(32), default="tracker")
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    user: Mapped[User] = relationship(back_populates="rigs")


class Site(Base):
    """Saved observing location with coordinates and Bortle sky darkness."""

    __tablename__ = "sites"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    name: Mapped[str] = mapped_column(String(120))
    lat: Mapped[float] = mapped_column(Float)
    lon: Mapped[float] = mapped_column(Float)
    bortle_class: Mapped[int | None] = mapped_column(Integer, nullable=True)
    elevation_m: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    user: Mapped[User] = relationship(back_populates="sites")


class ObservingSession(Base):
    """Planned or completed real-world astronomy observing/imaging session."""

    __tablename__ = "observing_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(160))
    obs_date: Mapped[str] = mapped_column(String(16), index=True)  # ISO YYYY-MM-DD
    status: Mapped[str] = mapped_column(
        String(24), default="planned"
    )  # planned|completed|cancelled
    site_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("sites.id", ondelete="SET NULL"), nullable=True
    )
    site_name: Mapped[str] = mapped_column(String(120), default="")
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    rig_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("rigs.id", ondelete="SET NULL"), nullable=True
    )
    rig_name: Mapped[str] = mapped_column(String(120), default="")
    cloud_cover_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    moon_illumination_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    user: Mapped[User] = relationship(back_populates="sessions")
    targets: Mapped[list[SessionTarget]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class SessionTarget(Base):
    """Individual deep-sky target scheduled or logged inside an `ObservingSession`."""

    __tablename__ = "session_targets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("observing_sessions.id", ondelete="CASCADE"), index=True
    )
    object_id: Mapped[str] = mapped_column(String(32), index=True)
    object_name: Mapped[str] = mapped_column(String(120), default="")
    ra_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    dec_deg: Mapped[float | None] = mapped_column(Float, nullable=True)
    exposure_minutes: Mapped[float] = mapped_column(Float, default=0.0)
    sub_exposure_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    frames_count: Mapped[int] = mapped_column(Integer, default=0)
    filter_name: Mapped[str | None] = mapped_column(String(48), nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="planned")  # planned|observed|imaged
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")

    session: Mapped[ObservingSession] = relationship(back_populates="targets")


class CatalogProgress(Base):
    """Per-user Messier / Caldwell / NGC / IC completion record."""

    __tablename__ = "catalog_progress"
    __table_args__ = (
        UniqueConstraint("user_id", "object_id", name="uq_catalog_progress_user_obj"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    object_id: Mapped[str] = mapped_column(String(32), index=True)
    catalog: Mapped[str] = mapped_column(String(32), default="Messier")
    status: Mapped[str] = mapped_column(String(24), default="observed")  # wishlist|observed|imaged
    first_observed_at: Mapped[str | None] = mapped_column(String(16), nullable=True)
    total_exposure_minutes: Mapped[float] = mapped_column(Float, default=0.0)
    best_session_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("observing_sessions.id", ondelete="SET NULL"), nullable=True
    )
    notes: Mapped[str] = mapped_column(Text, default="")

    user: Mapped[User] = relationship(back_populates="catalog_progress")


class ChatThread(Base):
    """Server-persisted conversation thread belonging to a user."""

    __tablename__ = "chat_threads"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(200), default="New observing chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    user: Mapped[User] = relationship(back_populates="chat_threads")
    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="thread",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class ChatMessage(Base):
    """Single user or assistant message persisted inside a `ChatThread`."""

    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    thread_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("chat_threads.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user | assistant
    content: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    thread: Mapped[ChatThread] = relationship(back_populates="messages")


class MagicLinkToken(Base):
    """Single-use SHA-256 hashed Magic Link login token (15-minute TTL)."""

    __tablename__ = "magic_link_tokens"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    consumed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class PendingActionApproval(Base):
    """Single-use SHA-256 capability token authorizing a destructive HITL action."""

    __tablename__ = "pending_action_approvals"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    action_type: Mapped[str] = mapped_column(String(64))
    target_id: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    consumed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
