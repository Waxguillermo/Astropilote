"""Authentication, BYOK credential encryption, and rate-limiting package."""

from astropilote.auth.crypto import (
    SESSION_COOKIE_NAME,
    create_session_jwt,
    decrypt_secret,
    encrypt_secret,
    generate_raw_token,
    hash_token,
    mask_api_key,
    resolve_app_secret,
    verify_session_jwt,
)
from astropilote.auth.ratelimit import SlidingWindowRateLimiter, enforce_rate_limit
from astropilote.auth.service import (
    consume_magic_link,
    create_magic_link,
    delete_user_api_key,
    get_current_user,
    get_user_api_key,
    list_user_api_keys,
    resolve_astrometry_api_key,
    resolve_llm_api_key,
    save_user_api_key,
    upsert_oauth_user,
)

__all__ = [
    "SESSION_COOKIE_NAME",
    "SlidingWindowRateLimiter",
    "consume_magic_link",
    "create_magic_link",
    "create_session_jwt",
    "decrypt_secret",
    "delete_user_api_key",
    "encrypt_secret",
    "enforce_rate_limit",
    "generate_raw_token",
    "get_current_user",
    "get_user_api_key",
    "hash_token",
    "list_user_api_keys",
    "mask_api_key",
    "resolve_app_secret",
    "resolve_astrometry_api_key",
    "resolve_llm_api_key",
    "save_user_api_key",
    "upsert_oauth_user",
    "verify_session_jwt",
]
