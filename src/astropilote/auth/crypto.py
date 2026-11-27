"""Cryptographic primitives for JWT sessions, Fernet BYOK key encryption, and SHA-256 tokens."""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jwt
from cryptography.fernet import Fernet, InvalidToken

from astropilote.config import settings

SESSION_COOKIE_NAME = "astropilote_session"
_MIN_CLOUD_SECRET_LEN = 32


def resolve_app_secret() -> str:
    """Return the signing/encryption secret.

    In `cloud` mode, requires an explicit `APP_SECRET_KEY` (>= 32 chars) and refuses weak defaults.
    In `local` mode, persists a random 48-byte key inside `<data_dir>/.secret_key` (mode `0600`)
    when `APP_SECRET_KEY` is unset so encrypted BYOK keys survive restarts.
    """
    configured = settings.app_secret_key.strip()
    if configured:
        if settings.auth_mode == "cloud" and len(configured) < _MIN_CLOUD_SECRET_LEN:
            raise RuntimeError(
                f"APP_SECRET_KEY must be at least {_MIN_CLOUD_SECRET_LEN} characters in cloud mode."
            )
        return configured

    if settings.auth_mode == "cloud":
        raise RuntimeError(
            "APP_SECRET_KEY is required when AUTH_MODE=cloud (generate with `openssl rand -hex 32`)."
        )

    key_file: Path = (settings.data_dir / ".secret_key").resolve()
    key_file.parent.mkdir(parents=True, exist_ok=True)
    if key_file.exists():
        existing = key_file.read_text(encoding="utf-8").strip()
        if len(existing) >= _MIN_CLOUD_SECRET_LEN:
            return existing
    generated = secrets.token_urlsafe(48)
    key_file.write_text(generated, encoding="utf-8")
    try:
        os.chmod(key_file, 0o600)
    except OSError:
        pass
    return generated


def _fernet(secret: str | None = None) -> Fernet:
    raw_secret = (secret or resolve_app_secret()).encode("utf-8")
    digest = hashlib.sha256(raw_secret).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_secret(plaintext: str, *, secret: str | None = None) -> str:
    """Encrypt a BYOK API key with authenticated Fernet AES-128-CBC + HMAC-SHA256."""
    cleaned = plaintext.strip()
    if not cleaned:
        raise ValueError("Cannot encrypt an empty secret.")
    return _fernet(secret).encrypt(cleaned.encode("utf-8")).decode("ascii")


def decrypt_secret(ciphertext: str, *, secret: str | None = None) -> str:
    """Decrypt a stored Fernet ciphertext; raises `ValueError` on tampering or wrong key."""
    try:
        return _fernet(secret).decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError(
            "Stored API key could not be decrypted with the current APP_SECRET_KEY."
        ) from exc


def mask_api_key(raw_key: str) -> str:
    """Return a non-sensitive display hint such as `sk-ant-...4f2a`."""
    cleaned = raw_key.strip()
    if len(cleaned) <= 10:
        return "****"
    return f"{cleaned[:6]}...{cleaned[-4:]}"


def hash_token(raw_token: str) -> str:
    """Return the hex SHA-256 digest of a one-time capability or magic-link token."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def generate_raw_token() -> str:
    """Generate a high-entropy URL-safe one-time token."""
    return secrets.token_urlsafe(32)


def create_session_jwt(
    *,
    user_id: str,
    email: str | None = None,
    ttl_hours: int = 168,
    secret: str | None = None,
) -> str:
    """Sign an HS256 JWT session token."""
    now = datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": user_id,
        "email": email,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=ttl_hours)).timestamp()),
    }
    return jwt.encode(payload, secret or resolve_app_secret(), algorithm="HS256")


def verify_session_jwt(token: str, *, secret: str | None = None) -> dict[str, Any]:
    """Verify an HS256 JWT session token and return its claims; raises `ValueError` if invalid."""
    try:
        claims = jwt.decode(token, secret or resolve_app_secret(), algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise ValueError("Invalid or expired session token.") from exc
    if not isinstance(claims, dict) or not claims.get("sub"):
        raise ValueError("Session token missing subject claim.")
    return claims
