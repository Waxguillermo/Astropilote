"""Hybrid authentication (Local zero-login + Cloud OAuth/Magic-Link) and encrypted BYOK service."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Literal

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from astropilote.auth.crypto import (
    SESSION_COOKIE_NAME,
    create_session_jwt,
    decrypt_secret,
    encrypt_secret,
    generate_raw_token,
    hash_token,
    mask_api_key,
    verify_session_jwt,
)
from astropilote.config import settings
from astropilote.db.models import MagicLinkToken, User, UserApiKey
from astropilote.db.session import ensure_local_user, get_db

ProviderName = Literal["anthropic", "openrouter", "astrometry"]
_VALID_PROVIDERS: frozenset[str] = frozenset({"anthropic", "openrouter", "astrometry"})
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _as_utc(dt: datetime) -> datetime:
    """Normalize naive SQLite timestamps to UTC-aware datetimes."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def normalize_email(email: str) -> str:
    cleaned = email.strip().lower()
    if not _EMAIL_RE.match(cleaned) or len(cleaned) > 254:
        raise ValueError("Invalid email address.")
    return cleaned


def create_magic_link(
    db: Session,
    email: str,
    *,
    ttl_minutes: int = 15,
) -> tuple[str, str]:
    """Create a 15-minute single-use Magic Link token; returns `(raw_token, verify_url)`."""
    norm_email = normalize_email(email)
    raw = generate_raw_token()
    row = MagicLinkToken(
        token_hash=hash_token(raw),
        email=norm_email,
        expires_at=datetime.now(UTC) + timedelta(minutes=ttl_minutes),
        consumed=False,
    )
    db.add(row)
    db.flush()
    base = settings.public_base_url.rstrip("/")
    return raw, f"{base}/api/auth/magic-link/verify?token={raw}"


def consume_magic_link(db: Session, raw_token: str) -> tuple[User, str]:
    """Validate and atomically consume a Magic Link token; returns `(user, jwt_token)`."""
    token_hash = hash_token(raw_token.strip())
    row = db.get(MagicLinkToken, token_hash)
    if row is None or row.consumed or _as_utc(row.expires_at) < datetime.now(UTC):
        raise ValueError("Magic link is invalid, expired, or already used.")
    row.consumed = True

    user = db.scalar(select(User).where(User.email == row.email))
    if user is None:
        display = row.email.split("@", 1)[0]
        user = User(
            email=row.email,
            display_name=display,
            provider="magic_link",
            provider_subject=row.email,
        )
        db.add(user)
    db.flush()
    token = create_session_jwt(user_id=user.id, email=user.email)
    return user, token


def upsert_oauth_user(
    db: Session,
    *,
    provider: Literal["github", "google"],
    subject: str,
    email: str | None,
    display_name: str,
) -> tuple[User, str]:
    """Create or update a GitHub/Google OAuth user and return `(user, jwt_token)`."""
    sub = subject.strip()
    if not sub:
        raise ValueError("OAuth subject cannot be empty.")
    norm_email = normalize_email(email) if email else None
    user = db.scalar(select(User).where(User.provider == provider, User.provider_subject == sub))
    if user is None and norm_email:
        user = db.scalar(select(User).where(User.email == norm_email))
    if user is None:
        user = User(
            email=norm_email,
            display_name=(display_name or "Observer").strip()[:120],
            provider=provider,
            provider_subject=sub,
        )
        db.add(user)
    else:
        if norm_email and not user.email:
            user.email = norm_email
        if display_name.strip():
            user.display_name = display_name.strip()[:120]
    db.flush()
    token = create_session_jwt(user_id=user.id, email=user.email)
    return user, token


def save_user_api_key(
    db: Session,
    *,
    user_id: str,
    provider: ProviderName,
    raw_key: str,
) -> dict[str, str]:
    """Encrypt and store a BYOK provider API key scoped strictly to `user_id`."""
    if provider not in _VALID_PROVIDERS:
        raise ValueError(f"Unsupported provider `{provider}`.")
    cleaned = raw_key.strip()
    if len(cleaned) < 8:
        raise ValueError("API key is too short.")
    ciphertext = encrypt_secret(cleaned)
    hint = mask_api_key(cleaned)

    existing = db.scalar(
        select(UserApiKey).where(UserApiKey.user_id == user_id, UserApiKey.provider == provider)
    )
    if existing is None:
        db.add(
            UserApiKey(
                user_id=user_id,
                provider=provider,
                encrypted_key=ciphertext,
                key_hint=hint,
            )
        )
    else:
        existing.encrypted_key = ciphertext
        existing.key_hint = hint
    db.flush()
    return {"provider": provider, "key_hint": hint}


def get_user_api_key(db: Session, *, user_id: str, provider: ProviderName) -> str | None:
    """Decrypt and return the stored BYOK key for `(user_id, provider)` or `None`."""
    row = db.scalar(
        select(UserApiKey).where(UserApiKey.user_id == user_id, UserApiKey.provider == provider)
    )
    if row is None:
        return None
    return decrypt_secret(row.encrypted_key)


def delete_user_api_key(db: Session, *, user_id: str, provider: ProviderName) -> bool:
    """Delete a stored BYOK key owned by `user_id`."""
    row = db.scalar(
        select(UserApiKey).where(UserApiKey.user_id == user_id, UserApiKey.provider == provider)
    )
    if row is None:
        return False
    db.delete(row)
    db.flush()
    return True


def list_user_api_keys(db: Session, *, user_id: str) -> list[dict[str, str]]:
    """List masked key metadata (`provider`, `key_hint`) for `user_id` without exposing secrets."""
    rows = db.scalars(select(UserApiKey).where(UserApiKey.user_id == user_id)).all()
    return [{"provider": r.provider, "key_hint": r.key_hint} for r in rows]


def resolve_llm_api_key(
    db: Session,
    *,
    user_id: str,
    header_api_key: str | None = None,
) -> str | None:
    """Resolve the active LLM API key enforcing strict BYOK isolation in `cloud` mode.

    - In `cloud` mode: uses `header_api_key` or the user's encrypted DB key (`anthropic` then
      `openrouter`). Never falls back to the server's `.env` key (`$0` server LLM cost).
    - In `local` mode: checks `header_api_key`, then user DB key, then `.env` `ANTHROPIC_API_KEY`.
    """
    if header_api_key and header_api_key.strip():
        return header_api_key.strip()
    for provider in ("anthropic", "openrouter"):
        stored = get_user_api_key(db, user_id=user_id, provider=provider)
        if stored:
            return stored
    if settings.auth_mode == "cloud":
        return None
    env_key = settings.anthropic_api_key.strip()
    return env_key or None


def resolve_astrometry_api_key(db: Session, *, user_id: str) -> str | None:
    """Resolve user's stored `astrometry` key, falling back to `ASTROMETRY_API_KEY` env var."""
    stored = get_user_api_key(db, user_id=user_id, provider="astrometry")
    if stored:
        return stored
    env_key = settings.astrometry_api_key.strip()
    return env_key or None


def _extract_token(request: Request) -> str | None:
    cookie_val = request.cookies.get(SESSION_COOKIE_NAME)
    if cookie_val:
        return cookie_val.strip()
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip() or None
    return None


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    """FastAPI dependency returning the authenticated user (auto-provisioned solo user in local mode)."""
    if settings.auth_mode == "local":
        return ensure_local_user(db)

    raw_jwt = _extract_token(request)
    if not raw_jwt:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        claims = verify_session_jwt(raw_jwt)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    user = db.get(User, str(claims["sub"]))
    if user is None:
        raise HTTPException(status_code=401, detail="Account no longer exists.")
    return user
