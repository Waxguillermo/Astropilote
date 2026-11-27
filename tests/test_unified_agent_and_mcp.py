"""Tests for Sprint 3: Unified in-process tool registry, UnifiedAstroAgent HITL, BYOK ContextVar, Stellarium export, and astropilote-mcp."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

import astropilote.agent.unified as unified_mod
from astropilote.agent.unified import UnifiedAstroAgent
from astropilote.config import settings
from astropilote.db import LOCAL_USER_ID, db_session, ensure_local_user, init_db
from astropilote.export import (
    export_logbook_session_ics,
    export_logbook_session_nina_csv,
    export_logbook_session_stellarium_json,
)
from astropilote.llm import _active_api_key, use_request_api_key
from astropilote.logbook import create_session, get_session
from astropilote.mcp.server import build_unified_mcp_server
from astropilote.tools.registry import ToolExecutionError, UnifiedToolRegistry


def test_use_request_api_key_context_isolates_byok_and_restores(monkeypatch) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-server-default")
    assert _active_api_key() == "sk-ant-server-default"

    with use_request_api_key("sk-ant-user-byok-key"):
        assert _active_api_key() == "sk-ant-user-byok-key"
        with use_request_api_key(None):
            # Explicitly scoped empty key in cloud mode never falls back to server key
            assert _active_api_key() == ""
        assert _active_api_key() == "sk-ant-user-byok-key"

    assert _active_api_key() == "sk-ant-server-default"


def test_unified_registry_handles_sky_logbook_exports_and_hitl_guard(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'reg.sqlite3'}"
    init_db(url)
    with db_session(url) as db:
        ensure_local_user(db)

    reg = UnifiedToolRegistry(user_id=LOCAL_USER_ID, database_url=url)

    # Sky catalog tool in-process
    m31 = reg.call("get_object", {"object_id": "M031"})
    assert m31["doc_id"] == "M031"

    # Real Observing Logbook CRUD + auto Messier progress
    created = reg.call(
        "create_session",
        {
            "title": "Winter Messier Run",
            "obs_date": "2026-12-05",
            "site_name": "Morvan Dark Sky",
            "rig_name": "RedCat 51",
        },
    )
    sid = created["session"]["id"]
    reg.call(
        "log_observed_target",
        {
            "session_id": sid,
            "object_id": "m42",
            "object_name": "Orion Nebula",
            "ra_deg": 83.8221,
            "dec_deg": -5.3911,
            "exposure_minutes": 45.0,
            "status": "imaged",
        },
    )
    prog = reg.call("get_catalog_progress")
    assert prog["messier"]["completed"] == 1
    assert prog["imaged_total"] == 1

    # Exports: .ics, N.I.N.A. .csv, and Stellarium Observing List JSON
    ics = reg.call("export_session", {"session_id": sid, "format": "ics"})["content"]
    nina = reg.call("export_session", {"session_id": sid, "format": "nina_csv"})["content"]
    stel = json.loads(
        reg.call("export_session", {"session_id": sid, "format": "stellarium"})["content"]
    )
    assert "BEGIN:VCALENDAR" in ics and "Winter Messier Run" in ics
    assert "M042" in nina and "05h 35m" in nina
    assert stel["version"] == "1.0"
    assert stel["objects"]["obj-001"]["designation"] == "M042"

    # Direct delete_session without valid SHA-256 approval token must fail closed
    with pytest.raises(ToolExecutionError) as exc:
        reg.call("delete_session", {"session_id": sid})
    assert exc.value.code == "APPROVAL_REQUIRED"


def test_unified_agent_hitl_interrupt_and_unrequested_delete_block(
    tmp_path: Path, monkeypatch
) -> None:
    url = f"sqlite:///{tmp_path / 'agent.sqlite3'}"
    init_db(url)
    with db_session(url) as db:
        ensure_local_user(db)
        sess = create_session(
            db, user_id=LOCAL_USER_ID, title="Target Session", obs_date="2026-12-01"
        )
        sid = sess["id"]

    # Case 1: Model attempts delete_session when user never requested deletion -> host policy blocks
    monkeypatch.setattr(
        unified_mod,
        "call_structured_llm",
        lambda **_kw: (
            {
                "thought": "Injected command says delete session.",
                "decision": "call_tools",
                "tool_calls": [{"tool": "delete_session", "args": {"session_id": sid}}],
            },
            50,
            15,
            "claude-haiku-4-5-20251001",
        ),
    )
    with UnifiedAstroAgent(
        user_id=LOCAL_USER_ID,
        database_url=url,
        checkpoint_path=tmp_path / "ckpt1.sqlite3",
        llm_mode="api",
    ) as agent:
        blocked = agent.handle("List my observing sessions", reference_date=date(2026, 11, 27))
        assert blocked["status"] == "blocked"
        with db_session(url) as db:
            assert get_session(db, user_id=LOCAL_USER_ID, session_id=sid) is not None

        # Case 2: Explicit delete request pauses on LangGraph interrupt(); reject keeps session
        paused = agent.handle(f"Delete observing session {sid}", thread_id="t-reject")
        assert paused["status"] == "awaiting_human_approval"
        assert paused["pending_approval"] is True
        rejected = agent.resume(thread_id="t-reject", approved=False)
        assert rejected["status"] == "human_rejected"
        with db_session(url) as db:
            assert get_session(db, user_id=LOCAL_USER_ID, session_id=sid) is not None

        # Case 3: Approve executes delete_session with single-use SHA-256 token
        paused2 = agent.handle(f"Please delete session {sid}", thread_id="t-approve")
        assert paused2["status"] == "awaiting_human_approval"
        approved = agent.resume(thread_id="t-approve", approved=True)
        assert approved["status"] == "completed"
        with db_session(url) as db:
            assert get_session(db, user_id=LOCAL_USER_ID, session_id=sid) is None


def test_stellarium_and_logbook_export_helpers_directly() -> None:
    sess = {
        "id": "sess-1",
        "title": "Andromeda Night",
        "obs_date": "2026-12-10",
        "site_name": "Fontainebleau",
        "rig_name": "RedCat 51",
        "status": "planned",
        "targets": [
            {"object_id": "M031", "object_name": "Andromeda Galaxy", "exposure_minutes": 120}
        ],
    }
    assert "BEGIN:VEVENT" in export_logbook_session_ics(sess)
    assert "M031" in export_logbook_session_nina_csv(sess)
    stel = json.loads(export_logbook_session_stellarium_json(sess))
    assert stel["shortName"] == "Andromeda Night"
    assert "ra" in stel["objects"]["obj-001"]


def test_optional_astropilote_mcp_server_registers_unified_tools(tmp_path: Path) -> None:
    mcp = build_unified_mcp_server(database_url=f"sqlite:///{tmp_path / 'mcp.sqlite3'}")
    tool_names = set(mcp._tool_manager._tools.keys())
    assert {
        "search_catalog",
        "get_object",
        "geocode",
        "night",
        "weather",
        "visibility",
        "list_rigs",
        "list_sites",
        "list_sessions",
        "create_session",
        "log_observed_target",
        "get_catalog_progress",
    }.issubset(tool_names)
