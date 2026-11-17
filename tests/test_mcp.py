"""Exercise both MCP 2.x servers (`sky` and `observatory`) over the client protocol in-process."""

from __future__ import annotations

import asyncio
import gc
import json
import weakref
from pathlib import Path

import pytest
from mcp import Client
from mcp.types import CallToolResult

import astropilote.agentic.workflow as workflow_mod
from astropilote.agentic.workflow import MCPTransportError, StdioMCPGateway, ToolFailure
from astropilote.backend.store import ObservatoryStore
from astropilote.mcp import sky_server
from astropilote.mcp.common import invoke
from astropilote.mcp.observatory_server import create_server

SEED = Path(__file__).resolve().parents[1] / "seeds" / "observatory_demo.json"


def _run(coro):
    return asyncio.run(coro)


def test_two_independent_mcp_servers_and_typed_schemas(tmp_path: Path) -> None:
    async def check() -> None:
        async with (
            Client(sky_server.mcp) as sky,
            Client(create_server(tmp_path / "obs.sqlite3")) as obs,
        ):
            sky_tools = {t.name: t for t in (await sky.list_tools()).tools}
            obs_tools = {t.name: t for t in (await obs.list_tools()).tools}
            assert set(sky_tools) == {
                "search_catalog",
                "get_object",
                "geocode",
                "night",
                "weather",
                "visibility",
            }
            assert set(obs_tools) == {
                "get_observatory_state",
                "list_telescopes",
                "list_slots",
                "get_booking",
                "create_booking",
                "cancel_booking",
                "refund_all",
            }
            assert sky_tools["night"].input_schema["properties"]["lat"]["type"] == "number"
            assert (
                obs_tools["create_booking"].input_schema["properties"]["slot_id"]["type"]
                == "string"
            )
            assert "slot_id" in obs_tools["create_booking"].input_schema["required"]

    _run(check())


def test_sky_geocode_via_mcp(monkeypatch) -> None:
    monkeypatch.setattr(
        sky_server.world_tools,
        "geocode",
        lambda place_name: {"found": True, "place_name": place_name, "lat": 48.4, "lon": 2.7},
    )

    async def check() -> None:
        async with Client(sky_server.mcp) as client:
            await client.list_tools()
            result = await client.call_tool("geocode", {"place_name": "Fontainebleau"})
            assert not result.is_error
            assert result.structured_content == {
                "ok": True,
                "data": {"found": True, "place_name": "Fontainebleau", "lat": 48.4, "lon": 2.7},
            }

    _run(check())


def test_sky_night_via_mcp(monkeypatch) -> None:
    monkeypatch.setattr(
        sky_server.sky_tools,
        "night",
        lambda **kwargs: {"date": kwargs["obs_date"], "days": kwargs["days"]},
    )

    async def check() -> None:
        async with Client(sky_server.mcp) as client:
            await client.list_tools()
            result = await client.call_tool(
                "night", {"lat": 48.4, "lon": 2.7, "obs_date": "2026-11-17", "days": 2}
            )
            assert result.structured_content == {
                "ok": True,
                "data": {"date": "2026-11-17", "days": 2},
            }

    _run(check())


def test_sky_rejects_invalid_coordinates() -> None:
    async def check() -> None:
        async with Client(sky_server.mcp) as client:
            await client.list_tools()
            result = await client.call_tool(
                "night", {"lat": 120.0, "lon": 2.7, "obs_date": "2026-11-17"}
            )
            assert result.is_error

    _run(check())


def test_observatory_booking_changes_real_sqlite_state(tmp_path: Path) -> None:
    store = ObservatoryStore(tmp_path / "obs.sqlite3")
    store.reset_backend(SEED)
    before = store.get_state()

    async def check() -> None:
        async with Client(create_server(store.db_path)) as client:
            await client.list_tools()
            result = await client.call_tool(
                "create_booking", {"slot_id": "S-01", "user_id": "observer"}
            )
            assert not result.is_error
            assert result.structured_content["ok"] is True
            assert result.structured_content["data"]["slot_id"] == "S-01"
            state = await client.call_tool("get_observatory_state")
            assert state.structured_content["data"] == store.get_state()
            assert state.structured_content["data"] != before

    _run(check())


def test_observatory_structured_domain_error_is_signaled(tmp_path: Path) -> None:
    store = ObservatoryStore(tmp_path / "obs.sqlite3")
    store.reset_backend(SEED)
    before = store.get_state()

    async def check() -> None:
        async with Client(create_server(store.db_path)) as client:
            await client.list_tools()
            result: CallToolResult = await client.call_tool(
                "create_booking", {"slot_id": "S-NOT-THERE", "user_id": "observer"}
            )
            assert result.is_error
            assert result.structured_content["ok"] is False
            assert result.structured_content["error"]["code"] == "SLOT_NOT_FOUND"
            assert json.loads(result.content[0].text) == result.structured_content
            assert store.get_state() == before

    _run(check())


def test_dangerous_cancellation_fails_closed_and_does_not_change_state(tmp_path: Path) -> None:
    store = ObservatoryStore(tmp_path / "obs.sqlite3")
    store.reset_backend(SEED)
    before = store.get_state()

    async def check() -> None:
        async with Client(create_server(store.db_path)) as client:
            await client.list_tools()
            result = await client.call_tool(
                "cancel_booking", {"booking_id": "BK-001", "user_id": "astro-club"}
            )
            assert result.is_error
            assert result.structured_content["error"]["code"] == "APPROVAL_REQUIRED"
            assert store.get_state() == before

    _run(check())


def test_observatory_invalid_args_rejected_without_write(tmp_path: Path) -> None:
    store = ObservatoryStore(tmp_path / "obs.sqlite3")
    store.reset_backend(SEED)
    before = store.get_state()

    async def check() -> None:
        async with Client(create_server(store.db_path)) as client:
            await client.list_tools()
            result = await client.call_tool(
                "create_booking", {"slot_id": "", "user_id": "observer"}
            )
            assert result.is_error
            assert store.get_state() == before

    _run(check())


def test_stdio_gateway_close_ends_its_subprocess_and_releases_the_instance(tmp_path: Path) -> None:
    """`close()` must stop the MCP server and drop the `atexit` hook that would keep it alive."""
    store = ObservatoryStore(tmp_path / "obs.sqlite3")
    store.reset_backend(SEED)
    gateway = StdioMCPGateway(store.db_path)
    assert len(gateway.call("observatory", "list_telescopes", {})) == 2
    session = gateway._sessions["observatory"]

    gateway.close()
    gateway.close()
    assert session.exited.is_set()

    released = weakref.ref(gateway)
    del gateway
    gc.collect()
    assert released() is None


def test_stdio_gateway_reports_transport_failures_as_typed_errors(
    tmp_path: Path, monkeypatch
) -> None:
    store = ObservatoryStore(tmp_path / "obs.sqlite3")
    store.reset_backend(SEED)
    monkeypatch.setitem(
        workflow_mod._SERVER_MODULES, "observatory", "astropilote.mcp.no_such_server"
    )

    with StdioMCPGateway(store.db_path) as gateway:
        with pytest.raises(MCPTransportError) as startup:
            gateway.call("observatory", "list_telescopes", {})
        assert startup.value.code == "MCP_START_FAILED"
        assert gateway._sessions == {}

    with pytest.raises(MCPTransportError) as closed:
        gateway.call("observatory", "list_telescopes", {})
    assert closed.value.code == "MCP_CLOSED"


def test_stdio_gateway_restarts_its_session_after_a_timed_out_call(
    tmp_path: Path, monkeypatch
) -> None:
    store = ObservatoryStore(tmp_path / "obs.sqlite3")
    store.reset_backend(SEED)
    with StdioMCPGateway(store.db_path) as gateway:
        assert len(gateway.call("observatory", "list_telescopes", {})) == 2
        stuck = gateway._sessions["observatory"]

        monkeypatch.setattr(workflow_mod, "_CALL_TIMEOUT_S", 0.0)
        with pytest.raises(MCPTransportError) as timed_out:
            gateway.call("observatory", "list_telescopes", {})
        assert timed_out.value.code == "TIMED_OUT"
        assert stuck.exited.is_set()

        monkeypatch.undo()
        assert len(gateway.call("observatory", "list_telescopes", {})) == 2
        assert gateway._sessions["observatory"] is not stuck


def test_gateway_types_argument_validation_failures(tmp_path: Path) -> None:
    """A violated `Field` bound surfaces as `INVALID_ARGUMENT` naming the field, not `BAD_MCP_RESULT`."""
    store = ObservatoryStore(tmp_path / "obs.sqlite3")
    store.reset_backend(SEED)
    with StdioMCPGateway(store.db_path) as gateway:
        with pytest.raises(ToolFailure) as out_of_range:
            gateway.call("sky", "night", {"lat": 120.0, "lon": 2.7, "obs_date": "2026-11-17"})
        error = out_of_range.value.payload["error"]
        assert error["code"] == "INVALID_ARGUMENT"
        assert "lat" in error["message"]

        with pytest.raises(ToolFailure) as unknown:
            gateway.call("sky", "no_such_tool", {})
        assert unknown.value.payload["error"]["code"] == "UNKNOWN_TOOL"


def test_gateway_rejects_impossible_calendar_dates(tmp_path: Path) -> None:
    """`2026-13-45` matches YYYY-MM-DD syntax but must be rejected before tool execution."""
    store = ObservatoryStore(tmp_path / "obs.sqlite3")
    store.reset_backend(SEED)
    with StdioMCPGateway(store.db_path) as gateway:
        with pytest.raises(ToolFailure) as bad_date:
            gateway.call("sky", "night", {"lat": 45.76, "lon": 4.84, "obs_date": "2026-13-45"})
    error = bad_date.value.payload["error"]
    assert error["code"] == "INVALID_ARGUMENT"
    assert "calendar date" in error["message"]


def test_gateway_invalid_write_arguments_are_typed_and_change_nothing(tmp_path: Path) -> None:
    store = ObservatoryStore(tmp_path / "obs.sqlite3")
    store.reset_backend(SEED)
    before = store.get_state()
    with StdioMCPGateway(store.db_path) as gateway:
        with pytest.raises(ToolFailure) as empty_slot:
            gateway.call("observatory", "create_booking", {"slot_id": "", "user_id": "observer"})
    error = empty_slot.value.payload["error"]
    assert error["code"] == "INVALID_ARGUMENT"
    assert "slot_id" in error["message"]
    assert store.get_state() == before


def test_invoke_types_value_errors_and_hides_unexpected_exceptions() -> None:
    def bad_value() -> None:
        raise ValueError("days must be positive")

    def bad_json() -> None:
        json.loads("{not-json-secret")

    def crash() -> None:
        raise KeyError("internal-secret")

    refused = invoke(bad_value)
    assert refused.is_error
    assert refused.structured_content["error"]["code"] == "INVALID_ARGUMENT"
    assert refused.structured_content["error"]["message"] == "days must be positive"

    decode_failed = invoke(bad_json)
    assert decode_failed.is_error
    assert decode_failed.structured_content["error"]["code"] == "TOOL_EXECUTION_ERROR"
    assert decode_failed.structured_content["error"]["details"]["exception"] == "JSONDecodeError"
    assert "not-json-secret" not in json.dumps(decode_failed.structured_content)

    crashed = invoke(crash)
    assert crashed.is_error
    assert crashed.structured_content["error"]["code"] == "TOOL_EXECUTION_ERROR"
    assert "internal-secret" not in json.dumps(crashed.structured_content)
    assert json.loads(crashed.content[0].text) == crashed.structured_content
