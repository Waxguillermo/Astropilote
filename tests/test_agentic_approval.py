"""Phase 3: actual MCP calls + durable LangGraph human approval boundary."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcp import Client

from astropilote.agentic.workflow import ObservatoryWorkflow
from astropilote.backend.approvals import CancellationApprovals
from astropilote.backend.store import ObservatoryStore
from astropilote.mcp.observatory_server import create_server

SEED = Path(__file__).resolve().parents[1] / "seeds" / "observatory_demo.json"


class InProcGateway:
    """Exercise real MCP transport in-process to keep unit tests fast/offline."""

    def __init__(self, db: Path):
        self.db = db

    def call(self, server: str, tool: str, args: dict):
        async def go():
            if server == "observatory":
                target = create_server(self.db)
            elif server == "sky":
                from astropilote.mcp.sky_server import mcp
                target = mcp
            else:
                raise ValueError(server)
            async with Client(target) as client:
                await client.list_tools()
                res = await client.call_tool(tool, args)

            # Propagate business errors only after the MCP client has closed.
            # Otherwise AnyIO wraps ToolFailure in an ExceptionGroup.
            payload = res.structured_content
            if res.is_error:
                from astropilote.agentic.workflow import ToolFailure
                raise ToolFailure(payload)
            assert payload["ok"] is True
            return payload["data"]

        return asyncio.run(go())


@pytest.fixture
def ctx(tmp_path: Path):
    db = tmp_path / "observatory.sqlite3"
    store = ObservatoryStore(db)
    store.reset_backend(SEED)
    return db, tmp_path / "checkpoints.sqlite3", InProcGateway(db), store


def wf(ctx):
    db, checkpoints, gateway, _ = ctx
    return ObservatoryWorkflow(db_path=db, checkpoint_path=checkpoints, gateway=gateway)


def test_safe_booking_uses_real_mcp_and_persists(ctx):
    with wf(ctx) as engine:
        result = engine.start(thread_id="book-1", action="book", slot_id="S-01", user_id="alice")
    assert result["status"] == "completed"
    assert result["pending_approval"] is False
    assert [r["tool"] for r in result["tool_trace"]] == ["list_slots", "create_booking"]
    assert ctx[3].get_booking("BK-002")["user_id"] == "alice"


def test_cancellation_interrupt_precedes_every_write(ctx):
    before = ctx[3].get_state()
    with wf(ctx) as engine:
        result = engine.start(thread_id="cancel-1", action="cancel", booking_id="BK-001", user_id="astro-club")
        pending = engine.inspect(thread_id="cancel-1")
    assert result["status"] == "awaiting_human_approval"
    assert pending["pending_approval"] is True
    assert [r["tool"] for r in result["tool_trace"]] == ["get_booking"]
    assert ctx[3].get_state() == before


def test_refusal_after_restart_leaves_backend_unchanged(ctx):
    before = ctx[3].get_state()
    with wf(ctx) as engine:
        engine.start(thread_id="denied", action="cancel", booking_id="BK-001", user_id="astro-club")
    with wf(ctx) as engine:
        result = engine.resume(thread_id="denied", approved=False)
    assert result["status"] == "human_rejected"
    assert result["pending_approval"] is False
    assert result["tool_trace"][-1]["tool"] == "get_booking"
    assert ctx[3].get_state() == before


def test_approved_resume_after_restart_cancels_via_real_mcp(ctx):
    with wf(ctx) as engine:
        engine.start(thread_id="approved", action="cancel", booking_id="BK-001", user_id="astro-club")
    with wf(ctx) as engine:
        result = engine.resume(thread_id="approved", approved=True)
    assert result["status"] == "completed"
    assert [x["tool"] for x in result["tool_trace"]] == ["get_booking", "cancel_booking"]
    assert ctx[3].get_booking("BK-001")["status"] == "cancelled"
    assert ctx[3].get_booking("BK-001")["refunded"] is True
    assert "approval_token" not in str(result)


def test_double_resume_is_rejected_without_second_write(ctx):
    with wf(ctx) as engine:
        engine.start(thread_id="once", action="cancel", booking_id="BK-001", user_id="astro-club")
        engine.resume(thread_id="once", approved=True)
        before = ctx[3].get_state()
        with pytest.raises(ValueError, match="No human approval"):
            engine.resume(thread_id="once", approved=True)
    assert ctx[3].get_state() == before


def test_untrusted_values_cannot_approve(ctx):
    before = ctx[3].get_state()
    with wf(ctx) as engine:
        engine.start(thread_id="bad-decision", action="cancel", booking_id="BK-001", user_id="astro-club")
        with pytest.raises(ValueError, match="explicit boolean"):
            engine.resume(thread_id="bad-decision", approved="false")  # type: ignore[arg-type]
    assert ctx[3].get_state() == before


def test_missing_or_forged_approval_token_rejected_by_mcp(ctx):
    before = ctx[3].get_state()
    gateway = ctx[2]
    from astropilote.agentic.workflow import ToolFailure
    for args in [
        {"booking_id": "BK-001", "user_id": "astro-club"},
        {"booking_id": "BK-001", "user_id": "astro-club", "approval_token": "approved=true"},
    ]:
        with pytest.raises(ToolFailure) as err:
            gateway.call("observatory", "cancel_booking", args)
        assert err.value.payload["error"]["code"] == "APPROVAL_REQUIRED"
    assert ctx[3].get_state() == before


def test_approval_capability_binds_actor_and_booking_and_is_one_time(ctx):
    issuer = CancellationApprovals(ctx[0])
    token = issuer.issue(booking_id="BK-001", user_id="astro-club")
    gateway = ctx[2]
    from astropilote.agentic.workflow import ToolFailure
    wrong = {"booking_id": "BK-001", "user_id": "other", "approval_token": token}
    with pytest.raises(ToolFailure):
        gateway.call("observatory", "cancel_booking", wrong)
    approved = {"booking_id": "BK-001", "user_id": "astro-club", "approval_token": token}
    succeeded = gateway.call("observatory", "cancel_booking", approved)
    assert succeeded["status"] == "cancelled"
    with pytest.raises(ToolFailure) as err:
        gateway.call("observatory", "cancel_booking", approved)
    assert err.value.payload["error"]["code"] == "APPROVAL_REQUIRED"


def test_reset_revokes_outstanding_approval(ctx):
    token = CancellationApprovals(ctx[0]).issue(booking_id="BK-001", user_id="astro-club")
    ctx[3].reset_backend(SEED)
    from astropilote.agentic.workflow import ToolFailure
    with pytest.raises(ToolFailure) as err:
        ctx[2].call("observatory", "cancel_booking", {
            "booking_id": "BK-001", "user_id": "astro-club", "approval_token": token
        })
    assert err.value.payload["error"]["code"] == "APPROVAL_REQUIRED"
    assert ctx[3].get_booking("BK-001")["status"] == "confirmed"


def test_step_budget_stops_before_booking_write(ctx):
    before = ctx[3].get_state()
    with wf(ctx) as engine:
        result = engine.start(thread_id="budget", action="book", slot_id="S-01", user_id="alice", max_steps=2)
    assert result["status"] == "max_steps"
    assert ctx[3].get_state() == before


def test_invalid_or_impossible_requests_cannot_write(ctx):
    before = ctx[3].get_state()
    with wf(ctx) as engine:
        missing = engine.start(thread_id="missing", action="book", user_id="alice")
        impossible = engine.start(thread_id="impossible", action="book", slot_id="S-04", user_id="alice")
    assert missing["status"] == "invalid_request"
    assert impossible["status"] == "impossible"
    assert ctx[3].get_state() == before


def test_gateway_can_call_other_sky_server(ctx):
    from astropilote.config import settings
    settings.eval_offline_mode = True
    result = ctx[2].call("sky", "geocode", {"place_name": "Fontainebleau"})
    assert result["found"] is True
    assert "lat" in result


def test_booking_can_consult_both_mcp_servers(ctx):
    with wf(ctx) as engine:
        result = engine.start(
            thread_id="book-sky", action="book", slot_id="S-01",
            user_id="alice", place_name="Fontainebleau"
        )
    assert result["status"] == "completed"
    assert [t["tool"] for t in result["tool_trace"]] == [
        "list_slots", "geocode", "create_booking"
    ]
    assert ctx[3].get_booking("BK-002")["status"] == "confirmed"
