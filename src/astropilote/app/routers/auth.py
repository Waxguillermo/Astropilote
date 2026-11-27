"""Modular FastAPI router for Hybrid Auth (Local Solo + Cloud Magic Link / OAuth) and BYOK keys."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from astropilote.auth import (
    SESSION_COOKIE_NAME,
    consume_magic_link,
    create_magic_link,
    delete_user_api_key,
    enforce_rate_limit,
    get_current_user,
    list_user_api_keys,
    resolve_astrometry_api_key,
    save_user_api_key,
)
from astropilote.config import settings
from astropilote.db import User, get_db
from astropilote.vision.solvers import detect_available_solvers

router = APIRouter(prefix="/api/auth", tags=["auth"])


class MagicLinkRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=254)


class SaveKeyRequest(BaseModel):
    api_key: str = Field(..., min_length=8, max_length=256)


@router.get("/me")
def auth_me(
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return active user profile, auth mode, masked BYOK keys, and available Vision solvers."""
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    astro_key = resolve_astrometry_api_key(db, user_id=user.id)
    return {
        "auth_mode": settings.auth_mode,
        "user": {
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
            "provider": user.provider,
        },
        "api_keys": list_user_api_keys(db, user_id=user.id),
        "vision_solvers": detect_available_solvers(astrometry_api_key=astro_key),
    }


@router.post("/magic-link")
def auth_request_magic_link(
    req: MagicLinkRequest,
    request: Request,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Create a 15-minute single-use Magic Link login token."""
    enforce_rate_limit(request)
    try:
        raw_token, verify_url = create_magic_link(db, req.email)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    out: dict[str, Any] = {
        "status": "sent",
        "message": "If this email is valid, a 15-minute sign-in link has been generated.",
    }
    # In local/dev preview, expose the verify URL so self-hosted setups work without an SMTP relay
    if settings.auth_mode == "local" or settings.public_base_url.startswith("http://127.0.0.1"):
        out["dev_verify_url"] = verify_url
        out["dev_token"] = raw_token
    return out


@router.get("/magic-link/verify")
def auth_verify_magic_link(
    token: str,
    response: Response,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Consume a Magic Link token and issue an `HttpOnly` JWT session cookie."""
    try:
        user, jwt_tok = consume_magic_link(db, token)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    is_https = settings.public_base_url.startswith("https://")
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=jwt_tok,
        httponly=True,
        secure=is_https,
        samesite="lax",
        max_age=7 * 24 * 3600,
        path="/",
    )
    return {
        "status": "authenticated",
        "user": {"id": user.id, "email": user.email, "display_name": user.display_name},
    }


@router.post("/logout")
def auth_logout(response: Response) -> dict[str, str]:
    """Clear the `HttpOnly` session cookie."""
    response.delete_cookie(key=SESSION_COOKIE_NAME, path="/")
    return {"status": "logged_out"}


@router.put("/keys/{provider}")
def auth_save_byok_key(
    provider: Literal["anthropic", "openrouter", "astrometry"],
    req: SaveKeyRequest,
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Encrypt and save a per-user BYOK key (`anthropic`, `openrouter`, or `astrometry`)."""
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    try:
        saved = save_user_api_key(db, user_id=user.id, provider=provider, raw_key=req.api_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "saved", **saved}


@router.delete("/keys/{provider}")
def auth_delete_byok_key(
    provider: Literal["anthropic", "openrouter", "astrometry"],
    request: Request,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Delete a stored BYOK key owned by the authenticated user."""
    enforce_rate_limit(request, subject_key=f"user:{user.id}")
    deleted = delete_user_api_key(db, user_id=user.id, provider=provider)
    return {"status": "deleted" if deleted else "not_found", "provider": provider}
