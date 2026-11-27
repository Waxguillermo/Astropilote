"""Tests for Sprint 2: SQLAlchemy 2 + Alembic migrations, Hybrid Auth, Fernet BYOK, rate-limiting, and Observing Logbook."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi import HTTPException
from sqlalchemy import inspect
from starlette.requests import Request

from astropilote.auth import (
    SlidingWindowRateLimiter,
    consume_magic_link,
    create_magic_link,
    decrypt_secret,
    delete_user_api_key,
    encrypt_secret,
    enforce_rate_limit,
    get_current_user,
    get_user_api_key,
    list_user_api_keys,
    resolve_llm_api_key,
    save_user_api_key,
    upsert_oauth_user,
    verify_session_jwt,
)
from astropilote.config import settings
from astropilote.db import (
    LOCAL_USER_ID,
    create_db_engine,
    db_session,
    ensure_local_user,
    init_db,
)
from astropilote.logbook import (
    add_target_to_session,
    append_chat_message,
    canonicalize_object_id,
    consume_action_approval,
    create_session,
    delete_rig,
    delete_session,
    get_catalog_progress_summary,
    get_chat_thread,
    get_session,
    issue_action_approval,
    list_chat_threads,
    list_rigs,
    list_sessions,
    list_sites,
    save_rig,
    save_site,
    update_session,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fake_request(
    *,
    cookie_token: str | None = None,
    bearer: str | None = None,
    client_ip: str = "203.0.113.10",
) -> Request:
    headers: list[tuple[bytes, bytes]] = []
    if cookie_token:
        headers.append((b"cookie", f"astropilote_session={cookie_token}".encode()))
    if bearer:
        headers.append((b"authorization", f"Bearer {bearer}".encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/test",
        "headers": headers,
        "client": (client_ip, 50000),
    }
    return Request(scope)


def test_alembic_upgrade_downgrade_and_wal_mode(tmp_path: Path) -> None:
    db_path = tmp_path / "migrations.sqlite3"
    url = f"sqlite:///{db_path}"
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)

    command.upgrade(cfg, "head")
    eng = create_db_engine(url)
    tables = set(inspect(eng).get_table_names())
    assert {
        "users",
        "user_api_keys",
        "rigs",
        "sites",
        "observing_sessions",
        "session_targets",
        "catalog_progress",
        "chat_threads",
        "chat_messages",
        "magic_link_tokens",
        "pending_action_approvals",
    }.issubset(tables)

    with eng.connect() as conn:
        mode = conn.exec_driver_sql("PRAGMA journal_mode").scalar()
        assert str(mode).lower() == "wal"

    command.downgrade(cfg, "base")
    assert set(inspect(eng).get_table_names()) == {"alembic_version"}
    command.upgrade(cfg, "head")
    assert "observing_sessions" in set(inspect(eng).get_table_names())
    eng.dispose()


def test_hybrid_auth_local_solo_vs_cloud_magic_link_and_oauth(tmp_path: Path, monkeypatch) -> None:
    url = f"sqlite:///{tmp_path / 'auth.sqlite3'}"
    init_db(url)
    monkeypatch.setattr(settings, "app_secret_key", "s" * 48)

    # Local mode: auto-provisions solo user with zero login
    monkeypatch.setattr(settings, "auth_mode", "local")
    with db_session(url) as db:
        local_user = get_current_user(_fake_request(), db)
        assert local_user.id == LOCAL_USER_ID

    # Cloud mode: rejects anonymous requests with 401
    monkeypatch.setattr(settings, "auth_mode", "cloud")
    with db_session(url) as db:
        with pytest.raises(HTTPException) as exc:
            get_current_user(_fake_request(), db)
        assert exc.value.status_code == 401

        raw_token, verify_url = create_magic_link(db, "Alice@Example.com")
        assert "token=" in verify_url
        user, jwt_tok = consume_magic_link(db, raw_token)
        assert user.email == "alice@example.com"
        assert verify_session_jwt(jwt_tok)["sub"] == user.id

        # Single-use: second consume fails
        with pytest.raises(ValueError):
            consume_magic_link(db, raw_token)

        # Authenticated via HttpOnly cookie and Bearer header
        authed_cookie = get_current_user(_fake_request(cookie_token=jwt_tok), db)
        authed_bearer = get_current_user(_fake_request(bearer=jwt_tok), db)
        assert authed_cookie.id == user.id == authed_bearer.id

        # OAuth upsert links cleanly
        oauth_user, oauth_jwt = upsert_oauth_user(
            db,
            provider="github",
            subject="gh-42",
            email="bob@example.com",
            display_name="Bob Astro",
        )
        assert oauth_user.id != user.id
        assert get_current_user(_fake_request(cookie_token=oauth_jwt), db).id == oauth_user.id


def test_byok_fernet_encryption_and_strict_cloud_isolation(tmp_path: Path, monkeypatch) -> None:
    url = f"sqlite:///{tmp_path / 'byok.sqlite3'}"
    init_db(url)
    monkeypatch.setattr(settings, "app_secret_key", "k" * 48)
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-server-secret-do-not-leak")

    ct = encrypt_secret("sk-ant-user-1234567890")
    assert "1234567890" not in ct
    assert decrypt_secret(ct) == "sk-ant-user-1234567890"

    with db_session(url) as db:
        u1, _ = upsert_oauth_user(
            db, provider="github", subject="u1", email="u1@ex.com", display_name="U1"
        )
        u2, _ = upsert_oauth_user(
            db, provider="github", subject="u2", email="u2@ex.com", display_name="U2"
        )
        saved = save_user_api_key(
            db, user_id=u1.id, provider="anthropic", raw_key="sk-ant-alice-9999"
        )
        assert saved["key_hint"] == "sk-ant...9999"
        assert get_user_api_key(db, user_id=u1.id, provider="anthropic") == "sk-ant-alice-9999"
        assert get_user_api_key(db, user_id=u2.id, provider="anthropic") is None
        assert list_user_api_keys(db, user_id=u1.id) == [
            {"provider": "anthropic", "key_hint": "sk-ant...9999"}
        ]

        # Cloud mode: strictly BYOK -> u2 NEVER inherits server's ANTHROPIC_API_KEY
        monkeypatch.setattr(settings, "auth_mode", "cloud")
        assert resolve_llm_api_key(db, user_id=u1.id) == "sk-ant-alice-9999"
        assert resolve_llm_api_key(db, user_id=u2.id) is None
        assert (
            resolve_llm_api_key(db, user_id=u2.id, header_api_key="sk-or-v1-header-key")
            == "sk-or-v1-header-key"
        )

        # Local mode: falls back to .env key
        monkeypatch.setattr(settings, "auth_mode", "local")
        assert resolve_llm_api_key(db, user_id=u2.id) == "sk-ant-server-secret-do-not-leak"

        assert delete_user_api_key(db, user_id=u1.id, provider="anthropic") is True
        assert get_user_api_key(db, user_id=u1.id, provider="anthropic") is None


def test_rate_limiter_enforced_in_cloud_and_bypassed_in_local(monkeypatch) -> None:
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60.0)
    req = _fake_request(client_ip="198.51.100.5")

    monkeypatch.setattr(settings, "auth_mode", "local")
    for _ in range(5):
        enforce_rate_limit(req, subject_key="user:1", limiter=limiter)

    monkeypatch.setattr(settings, "auth_mode", "cloud")
    enforce_rate_limit(req, subject_key="user:1", limiter=limiter)
    enforce_rate_limit(req, subject_key="user:1", limiter=limiter)
    with pytest.raises(HTTPException) as exc:
        enforce_rate_limit(req, subject_key="user:1", limiter=limiter)
    assert exc.value.status_code == 429
    assert "Retry-After" in (exc.value.headers or {})


def test_logbook_crud_catalog_progress_hitl_and_user_isolation(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'logbook.sqlite3'}"
    init_db(url)
    assert canonicalize_object_id("m31") == ("M031", "Messier")
    assert canonicalize_object_id("c20") == ("C020", "Caldwell")

    with db_session(url) as db:
        alice = ensure_local_user(db)
        bob, _ = upsert_oauth_user(
            db, provider="github", subject="bob", email="bob@ex.com", display_name="Bob"
        )

        rig = save_rig(
            db,
            user_id=alice.id,
            name="RedCat 51 + ASI2600MC",
            focal_length_mm=250.0,
            aperture_mm=51.0,
            sensor_format="APS-C",
            is_default=True,
        )
        site = save_site(
            db,
            user_id=alice.id,
            name="Fontainebleau Clearing",
            lat=48.4047,
            lon=2.7016,
            bortle_class=5,
            is_default=True,
        )
        assert len(list_rigs(db, user_id=alice.id)) == 1
        assert len(list_rigs(db, user_id=bob.id)) == 0
        assert len(list_sites(db, user_id=alice.id)) == 1

        sess = create_session(
            db,
            user_id=alice.id,
            title="Autumn Galaxies",
            obs_date="2026-11-28",
            status="planned",
            site_id=site["id"],
            rig_id=rig["id"],
            cloud_cover_pct=12.0,
            targets=[
                {
                    "object_id": "m31",
                    "object_name": "Andromeda Galaxy",
                    "ra_deg": 10.6847,
                    "dec_deg": 41.2687,
                    "exposure_minutes": 90.0,
                    "status": "imaged",
                }
            ],
        )
        add_target_to_session(
            db,
            user_id=alice.id,
            session_id=sess["id"],
            object_id="C20",
            object_name="North America Nebula",
            exposure_minutes=60.0,
            status="observed",
        )
        updated = update_session(db, user_id=alice.id, session_id=sess["id"], status="completed")
        assert updated["status"] == "completed"
        assert len(list_sessions(db, user_id=alice.id)) == 1
        assert get_session(db, user_id=bob.id, session_id=sess["id"]) is None

        # Messier/Caldwell progress auto-synced from logged targets
        prog = get_catalog_progress_summary(db, user_id=alice.id)
        assert prog["messier"]["completed"] == 1
        assert prog["caldwell"]["completed"] == 1
        assert prog["imaged_total"] == 1
        assert prog["total_exposure_minutes"] == 150.0
        assert get_catalog_progress_summary(db, user_id=bob.id)["messier"]["completed"] == 0

        # Chat thread persistence isolated per user
        m1 = append_chat_message(
            db, user_id=alice.id, role="user", content="Plan tonight in Fontainebleau"
        )
        append_chat_message(
            db,
            user_id=alice.id,
            thread_id=m1["thread_id"],
            role="assistant",
            content="Clear skies!",
        )
        assert len(list_chat_threads(db, user_id=alice.id)) == 1
        assert (
            len(get_chat_thread(db, user_id=alice.id, thread_id=m1["thread_id"])["messages"]) == 2
        )
        assert get_chat_thread(db, user_id=bob.id, thread_id=m1["thread_id"]) is None

        # Single-use SHA-256 HITL approval bound to (user_id, action_type, target_id)
        tok = issue_action_approval(
            db, user_id=alice.id, action_type="delete_session", target_id=sess["id"]
        )
        assert (
            consume_action_approval(
                db,
                raw_token=tok,
                user_id=bob.id,
                action_type="delete_session",
                target_id=sess["id"],
            )
            is False
        )
        assert (
            consume_action_approval(
                db,
                raw_token=tok,
                user_id=alice.id,
                action_type="delete_session",
                target_id=sess["id"],
            )
            is True
        )
        assert (
            consume_action_approval(
                db,
                raw_token=tok,
                user_id=alice.id,
                action_type="delete_session",
                target_id=sess["id"],
            )
            is False
        )

        # Bob cannot delete Alice's session or rig
        assert delete_session(db, user_id=bob.id, session_id=sess["id"]) is False
        assert delete_rig(db, user_id=bob.id, rig_id=rig["id"]) is False
        assert delete_session(db, user_id=alice.id, session_id=sess["id"]) is True
        assert get_session(db, user_id=alice.id, session_id=sess["id"]) is None
