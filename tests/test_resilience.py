"""Bounded recovery from injected MCP timeouts and 429 rate limits (T10)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from astropilote.agentic.resilience import (
    RETRYABLE_TOOLS,
    FaultInjectingGateway,
    FaultSpec,
    RetryingGateway,
    RetryPolicy,
)
from astropilote.agentic.scenario_eval import READ_TOOLS, load_scenarios, run_one
from astropilote.agentic.workflow import MCPTransportError, ToolFailure

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "seeds" / "observatory_demo.json"
RESILIENCE_SCENARIOS = ROOT / "eval" / "scenarios_resilience.jsonl"


class EchoGateway:
    """Healthy MCP stand-in: records each call that reaches it and answers deterministically."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def call(self, server: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(tool)
        return {"server": server, "tool": tool}


class DomainErrorGateway:
    """MCP server that answers every call with the same structured domain error."""

    def __init__(self, code: str) -> None:
        self.code = code
        self.calls: list[str] = []

    def call(self, server: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(tool)
        raise ToolFailure({"ok": False, "error": {"code": self.code}})


def _stack(
    *faults: FaultSpec, policy: RetryPolicy | None = None
) -> tuple[RetryingGateway, FaultInjectingGateway, EchoGateway, list[float]]:
    backend = EchoGateway()
    sleeps: list[float] = []
    injector = FaultInjectingGateway(backend, list(faults))
    gateway = RetryingGateway(injector, policy or RetryPolicy(), sleep=sleeps.append)
    return gateway, injector, backend, sleeps


# --- RetryingGateway ----------------------------------------------------------------------


def test_single_timeout_is_recovered_after_one_backoff():
    gateway, _, backend, sleeps = _stack(FaultSpec(tool="list_slots", kind="timeout"))

    result = gateway.call("observatory", "list_slots", {})

    assert result == {"server": "observatory", "tool": "list_slots"}
    assert backend.calls == ["list_slots"]
    assert sleeps == [0.2]
    assert gateway.log[-1]["outcome"] == "recovered"
    assert [a["code"] for a in gateway.last_attempts] == ["TIMED_OUT", "ok"]


def test_rate_limit_waits_for_retry_after():
    gateway, _, _, sleeps = _stack(
        FaultSpec(tool="list_telescopes", kind="rate_limit", times=2, retry_after_s=0.1)
    )

    gateway.call("observatory", "list_telescopes", {})

    assert sleeps == [0.1, 0.1]
    assert [a["code"] for a in gateway.last_attempts] == ["RATE_LIMITED", "RATE_LIMITED", "ok"]


def test_retry_after_is_capped_by_max_backoff():
    gateway, _, _, sleeps = _stack(
        FaultSpec(tool="weather", kind="rate_limit", retry_after_s=5.0),
        policy=RetryPolicy(max_backoff_s=1.5),
    )

    gateway.call("sky", "weather", {})

    assert sleeps == [1.5]


def test_persistent_timeout_exhausts_attempts_and_keeps_its_code():
    gateway, _, backend, sleeps = _stack(FaultSpec(tool="list_slots", kind="timeout", times=5))

    with pytest.raises(MCPTransportError) as raised:
        gateway.call("observatory", "list_slots", {})

    assert raised.value.code == "TIMED_OUT"
    assert backend.calls == []
    assert sleeps == [0.2, 0.4]
    assert len(gateway.last_attempts) == 3
    assert gateway.log[-1]["outcome"] == "exhausted"


def test_exhausted_rate_limit_surfaces_as_a_transport_error():
    """A 429 that never clears must read as an unavailable tool, not as a domain refusal."""
    gateway, _, _, _ = _stack(FaultSpec(tool="get_booking", kind="rate_limit", times=5))

    with pytest.raises(MCPTransportError) as raised:
        gateway.call("observatory", "get_booking", {"booking_id": "BK-001"})

    assert raised.value.code == "RATE_LIMITED"


def test_total_wait_budget_stops_retrying_early():
    gateway, _, _, sleeps = _stack(
        FaultSpec(tool="list_slots", kind="timeout", times=5),
        policy=RetryPolicy(max_attempts=5, max_total_wait_s=0.5),
    )

    with pytest.raises(MCPTransportError):
        gateway.call("observatory", "list_slots", {})

    assert sleeps == [0.2]
    assert sum(sleeps) <= 0.5


def test_open_circuit_fails_fast_without_calling_the_server():
    gateway, injector, backend, sleeps = _stack(
        FaultSpec(tool="list_slots", kind="timeout", times=3)
    )
    with pytest.raises(MCPTransportError):
        gateway.call("observatory", "list_slots", {})
    sleeps.clear()
    triggered_before = len(injector.triggered)

    with pytest.raises(MCPTransportError) as raised:
        gateway.call("observatory", "list_slots", {})

    assert raised.value.code == "CIRCUIT_OPEN"
    assert len(injector.triggered) == triggered_before
    assert backend.calls == []
    assert sleeps == []
    assert gateway.log[-1]["outcome"] == "circuit_open"


def test_open_circuit_is_scoped_to_the_failing_tool():
    gateway, _, backend, _ = _stack(FaultSpec(tool="list_slots", kind="timeout", times=3))
    with pytest.raises(MCPTransportError):
        gateway.call("observatory", "list_slots", {})

    gateway.call("observatory", "list_telescopes", {})

    assert backend.calls == ["list_telescopes"]


@pytest.mark.parametrize("tool", ["create_booking", "cancel_booking", "refund_all"])
def test_writes_are_never_replayed(tool: str):
    """A timed-out write may have committed: replaying could double-book or reuse a spent token."""
    gateway, _, backend, sleeps = _stack(FaultSpec(tool=tool, kind="timeout"))

    with pytest.raises(MCPTransportError) as raised:
        gateway.call("observatory", tool, {})

    assert raised.value.code == "TIMED_OUT"
    assert backend.calls == []
    assert sleeps == []
    assert gateway.log[-1]["outcome"] == "not_retried"


@pytest.mark.parametrize("code", ["SLOT_UNAVAILABLE", "INVALID_ARGUMENT", "BOOKING_NOT_FOUND"])
def test_domain_errors_are_not_retried(code: str):
    backend = DomainErrorGateway(code)
    sleeps: list[float] = []
    gateway = RetryingGateway(backend, sleep=sleeps.append)

    with pytest.raises(ToolFailure) as raised:
        gateway.call("observatory", "list_slots", {})

    assert raised.value.payload["error"]["code"] == code
    assert backend.calls == ["list_slots"]
    assert sleeps == []
    assert gateway.log[-1]["outcome"] == "failed"


def test_retryable_tools_are_exactly_the_evaluated_read_tools():
    assert RETRYABLE_TOOLS == READ_TOOLS


# --- FaultInjectingGateway ---------------------------------------------------------------


def test_injected_fault_fires_before_reaching_the_server():
    backend = EchoGateway()
    injector = FaultInjectingGateway(backend, [FaultSpec(tool="create_booking", kind="timeout")])

    with pytest.raises(MCPTransportError):
        injector.call("observatory", "create_booking", {"slot_id": "S-01"})
    assert backend.calls == []

    injector.call("observatory", "create_booking", {"slot_id": "S-01"})
    assert backend.calls == ["create_booking"]
    assert injector.triggered == [
        {"server": "observatory", "tool": "create_booking", "kind": "timeout"}
    ]


def test_injected_rate_limit_is_a_structured_tool_error():
    injector = FaultInjectingGateway(
        EchoGateway(), [FaultSpec(tool="weather", kind="rate_limit", retry_after_s=0.3)]
    )

    with pytest.raises(ToolFailure) as raised:
        injector.call("sky", "weather", {})

    error = raised.value.payload["error"]
    assert error["code"] == "RATE_LIMITED"
    assert error["details"]["retry_after_s"] == 0.3


# --- Real stdio MCP transport ---------------------------------------------------------


def test_real_stdio_session_recovers_from_a_real_timeout(tmp_path: Path, monkeypatch):
    """First call really times out (session killed); the retry restarts the server and succeeds."""
    pytest.importorskip("mcp")
    import astropilote.agentic.workflow as workflow_mod
    from astropilote.backend.store import ObservatoryStore

    db_path = tmp_path / "observatory.sqlite3"
    ObservatoryStore(db_path).reset_backend(SEED)
    monkeypatch.setattr(workflow_mod, "_CALL_TIMEOUT_S", 0.0)
    sleeps: list[float] = []

    def restore_timeout_then_record(seconds: float) -> None:
        sleeps.append(seconds)
        monkeypatch.setattr(workflow_mod, "_CALL_TIMEOUT_S", 25.0)

    gateway = RetryingGateway(
        workflow_mod.StdioMCPGateway(db_path, offline_sky=True), sleep=restore_timeout_then_record
    )
    try:
        slots = gateway.call("observatory", "list_slots", {"available_only": False})
    finally:
        gateway.close()

    assert [slot["id"] for slot in slots] == ["S-01", "S-02", "S-03", "S-04"]
    assert [a["code"] for a in gateway.last_attempts] == ["TIMED_OUT", "ok"]
    assert sleeps == [0.2]


def test_web_studio_recovers_from_a_transient_mcp_timeout(tmp_path: Path, monkeypatch):
    pytest.importorskip("mcp")
    from fastapi.testclient import TestClient

    import astropilote.app.main as main_mod

    monkeypatch.setattr(main_mod, "OBSERVATORY_DB_PATH", tmp_path / "observatory.sqlite3")
    monkeypatch.setattr(main_mod, "OBSERVATORY_CHECKPOINT_PATH", tmp_path / "checkpoints.sqlite3")
    monkeypatch.setattr(main_mod, "OBSERVATORY_MEMORY_PATH", tmp_path / "memory.sqlite3")
    shared_gateway = main_mod._get_observatory_gateway
    flaky: dict[str, FaultInjectingGateway] = {}

    def flaky_shared_gateway() -> FaultInjectingGateway:
        if "gateway" not in flaky:
            flaky["gateway"] = FaultInjectingGateway(
                shared_gateway(), [FaultSpec(tool="list_slots", kind="timeout")]
            )
        return flaky["gateway"]

    monkeypatch.setattr(main_mod, "_get_observatory_gateway", flaky_shared_gateway)

    response = TestClient(main_mod.app).post(
        "/api/observatory/run",
        json={
            "text": "Quels télescopes de l'observatoire sont disponibles mardi soir ?",
            "user_id": "alice",
            "llm_mode": "offline",
            "reference_date": "2026-11-14",
        },
    )

    assert response.status_code == 200
    assert response.json()["response"]["status"] == "completed"
    assert flaky["gateway"].triggered == [
        {"server": "observatory", "tool": "list_slots", "kind": "timeout"}
    ]


# --- Scenario harness ---------------------------------------------------------------------


def _availability_scenario(**overrides: Any) -> dict[str, Any]:
    scenario: dict[str, Any] = {
        "id": "R-test",
        "category": "normal",
        "seed": "seeds/observatory_demo.json",
        "turns": [
            {
                "text": "Quels télescopes de l'observatoire sont disponibles mardi soir ?",
                "user_id": "alice",
                "reference_date": "2026-11-14",
            }
        ],
        "max_steps": 6,
        "expect_responses": [
            {"status": "completed", "route": "availability", "data_ids": ["S-01", "S-03"]}
        ],
        "expect_tools": ["list_telescopes", "list_slots"],
        "expect_unchanged": True,
    }
    scenario.update(overrides)
    return scenario


def _write_jsonl(path: Path, *items: dict[str, Any]) -> Path:
    path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "field",
    [
        {"inject_tool_failure": {"tool": "list_slots", "kind": "timeout"}},
        {"inject_tool_failure": [{"tool": "refund_all", "kind": "timeout"}]},
        {"inject_tool_failure": [{"tool": "cancel_booking", "kind": "timeout"}]},
        {"inject_tool_failure": [{"tool": "list_slots", "kind": "explode"}]},
        {"inject_tool_failure": [{"tool": "list_slots", "kind": "timeout", "times": 0}]},
        {"inject_tool_failure": [{"tool": "list_slots", "kind": "timeout", "times": 6}]},
        {"inject_tool_failure": [{"tool": "list_slots", "kind": "timeout", "note": "x"}]},
        {"expect_retries": {"recovered": -1}},
        {"expect_retries": {"retried": 1}},
        {"expect_retries": [1]},
    ],
)
def test_invalid_failure_fixtures_are_rejected_before_execution(tmp_path: Path, field: dict):
    path = _write_jsonl(tmp_path / "bad.jsonl", _availability_scenario(**field))

    with pytest.raises(ValueError):
        load_scenarios(path)


def test_resilience_dataset_covers_recovery_exhaustion_and_writes():
    cases = load_scenarios(RESILIENCE_SCENARIOS)

    assert [case["id"] for case in cases] == ["R01", "R02", "R03", "R04", "R05", "R06"]
    kinds = {fault["kind"] for case in cases for fault in case["inject_tool_failure"]}
    assert kinds == {"timeout", "rate_limit"}


@pytest.mark.parametrize("scenario_id", ["R01", "R02", "R03", "R04", "R05", "R06"])
def test_resilience_scenarios_pass_with_real_mcp(scenario_id: str):
    pytest.importorskip("mcp")
    pytest.importorskip("langgraph")
    scenario = next(c for c in load_scenarios(RESILIENCE_SCENARIOS) if c["id"] == scenario_id)

    result = run_one(scenario, seed_path=SEED)

    assert result["pass"], result["reason"]


def test_eval_record_carries_the_retry_proof_trace():
    pytest.importorskip("mcp")
    scenario = next(c for c in load_scenarios(RESILIENCE_SCENARIOS) if c["id"] == "R01")

    result = run_one(scenario, seed_path=SEED)

    assert result["faults_injected"] == [
        {"server": "observatory", "tool": "list_slots", "kind": "timeout"}
    ]
    list_slots = next(call for call in result["tool_calls"] if call["tool"] == "list_slots")
    assert [a["code"] for a in list_slots["attempts"]] == ["TIMED_OUT", "ok"]
    assert [entry["outcome"] for entry in result["retries"]] == ["ok", "recovered"]


def test_declared_fault_that_never_fires_fails_the_scenario():
    pytest.importorskip("mcp")
    scenario = _availability_scenario(
        inject_tool_failure=[{"tool": "weather", "kind": "timeout", "times": 1}]
    )

    result = run_one(scenario, seed_path=SEED)

    assert result["pass"] is False
    assert "tool-failure fixture was not exercised" in result["reason"]


def test_retry_counters_are_graded():
    pytest.importorskip("mcp")
    scenario = _availability_scenario(
        inject_tool_failure=[{"tool": "list_slots", "kind": "timeout", "times": 1}],
        expect_retries={"recovered": 0},
    )

    result = run_one(scenario, seed_path=SEED)

    assert result["pass"] is False
    assert "expected recovered=0, got 1" in result["reason"]


def test_llm_agent_steps_are_unchanged_by_a_recovered_fault(monkeypatch):
    """Retries happen below the planner: the ReAct loop spends no extra step on them."""
    pytest.importorskip("mcp")
    import astropilote.agentic.llm_agent as llm_agent_mod

    def run(scenario: dict[str, Any]) -> dict[str, Any]:
        plans = iter(
            [
                (
                    {
                        "thought": "Read telescopes and slots.",
                        "decision": "call_tools",
                        "tool_calls": [
                            {"server": "observatory", "tool": "list_telescopes"},
                            {"server": "observatory", "tool": "list_slots", "available_only": True},
                        ],
                    },
                    100,
                    20,
                    "ollama/gemma4",
                ),
                (
                    {
                        "thought": "Return Tuesday's open slots.",
                        "decision": "finish",
                        "final_status": "completed",
                        "route": "availability",
                        "matching_slot_ids": ["S-01", "S-03"],
                    },
                    100,
                    20,
                    "ollama/gemma4",
                ),
            ]
        )
        monkeypatch.setattr(llm_agent_mod, "call_structured_llm", lambda **_kw: next(plans))
        return run_one(scenario, seed_path=SEED, llm_mode="local")

    baseline = run(_availability_scenario(expect_responses=[{"status": "completed"}]))
    faulty = run(
        _availability_scenario(
            expect_responses=[{"status": "completed"}],
            inject_tool_failure=[{"tool": "list_slots", "kind": "timeout", "times": 1}],
            expect_retries={"recovered": 1, "exhausted": 0},
        )
    )

    assert baseline["pass"], baseline["reason"]
    assert faulty["pass"], faulty["reason"]
    assert faulty["steps"] == baseline["steps"]


# --- One retry layer -------------------------------------------------------------------------


def _finish_impossible() -> tuple[dict[str, Any], int, int, str]:
    return (
        {
            "thought": "The slot list is unavailable; stop without guessing.",
            "decision": "finish",
            "final_status": "impossible",
            "route": "action",
            "message": "The observatory could not be reached.",
        },
        100,
        20,
        "ollama/gemma4",
    )


def test_llm_agent_relies_on_the_gateway_retry_alone(monkeypatch):
    """An exhausted read reaches the planner once, with its original code, not a CIRCUIT_OPEN echo."""
    pytest.importorskip("mcp")
    import astropilote.agentic.llm_agent as llm_agent_mod

    plans = iter(
        [
            (
                {
                    "thought": "List the slots.",
                    "decision": "call_tools",
                    "tool_calls": [
                        {"server": "observatory", "tool": "list_slots", "available_only": True}
                    ],
                },
                100,
                20,
                "ollama/gemma4",
            ),
            _finish_impossible(),
        ]
    )
    monkeypatch.setattr(llm_agent_mod, "call_structured_llm", lambda **_kw: next(plans))

    result = run_one(
        _availability_scenario(
            inject_tool_failure=[{"tool": "list_slots", "kind": "timeout", "times": 3}],
        ),
        seed_path=SEED,
        llm_mode="local",
    )

    assert [c["tool"] for c in result["tool_calls"]] == ["list_slots"]
    assert [entry["outcome"] for entry in result["retries"]] == ["exhausted"]


def test_two_agent_sky_planner_relies_on_the_gateway_retry_alone(tmp_path: Path, monkeypatch):
    pytest.importorskip("mcp")
    from datetime import date

    import astropilote.agentic.ablation as ablation_mod
    import astropilote.agentic.llm_agent as llm_agent_mod
    from astropilote.agentic.ablation import TwoAgentObservatoryAgent
    from astropilote.agentic.memory import UserMemory
    from astropilote.agentic.scenario_eval import TracedGateway
    from astropilote.agentic.workflow import StdioMCPGateway
    from astropilote.backend.store import ObservatoryStore

    db_path = tmp_path / "obs.sqlite3"
    ObservatoryStore(db_path).reset_backend(SEED)
    sky_steps = iter(
        [
            (
                {
                    "thought": "Locate Lyon, then check its weather.",
                    "decision": "call_tools",
                    "tool_calls": [
                        {"server": "sky", "tool": "geocode", "place_name": "Lyon"},
                        {
                            "server": "sky",
                            "tool": "weather",
                            "obs_date": "2026-11-17",
                            "place_name": "Lyon",
                        },
                    ],
                    "needs_sky_check": True,
                },
                80,
                20,
                "ollama/gemma4",
            ),
            (
                {
                    "thought": "Weather is unavailable.",
                    "decision": "finish",
                    "needs_sky_check": True,
                    "sky_ok": False,
                    "handoff_summary": "Weather service unavailable.",
                },
                80,
                20,
                "ollama/gemma4",
            ),
        ]
    )
    monkeypatch.setattr(ablation_mod, "call_structured_llm", lambda **_kw: next(sky_steps))
    monkeypatch.setattr(llm_agent_mod, "call_structured_llm", lambda **_kw: _finish_impossible())

    with StdioMCPGateway(db_path, offline_sky=True) as raw:
        retrying = RetryingGateway(
            FaultInjectingGateway(raw, [FaultSpec(tool="weather", kind="timeout", times=3)]),
            RetryPolicy(base_backoff_s=0.0),
        )
        traced = TracedGateway(retrying)
        with TwoAgentObservatoryAgent(
            db_path=db_path,
            checkpoint_path=tmp_path / "ckpt.sqlite3",
            memory=UserMemory(tmp_path / "mem.sqlite3"),
            gateway=traced,
            llm_mode="local",
        ) as agent:
            agent.handle(
                "Check the night weather in Lyon on 2026-11-17 and book S-01 if it is clear",
                user_id="alice",
                reference_date=date(2026, 11, 14),
                max_steps=6,
            )

    assert [c["tool"] for c in traced.calls] == ["geocode", "weather"]
    assert [(entry["tool"], entry["outcome"]) for entry in retrying.log] == [
        ("geocode", "ok"),
        ("weather", "exhausted"),
    ]


def test_llm_agent_wraps_its_own_mcp_session_in_the_retry_layer(tmp_path: Path, monkeypatch):
    """Without an injected gateway (CLI chat), the agent's own MCP session still recovers."""
    import astropilote.agentic.llm_agent as llm_agent_mod

    class TimesOutOnce(EchoGateway):
        def call(self, server: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
            self.calls.append(tool)
            if len(self.calls) == 1:
                raise MCPTransportError("TIMED_OUT", "first call timed out")
            return {"server": server, "tool": tool}

    session = TimesOutOnce()
    monkeypatch.setattr(llm_agent_mod, "StdioMCPGateway", lambda *_a, **_kw: session)

    with llm_agent_mod.LLMObservatoryAgent(
        db_path=tmp_path / "obs.sqlite3",
        checkpoint_path=tmp_path / "ckpt.sqlite3",
        memory_path=tmp_path / "mem.sqlite3",
        llm_mode="local",
    ) as agent:
        answer = agent.gateway.call("observatory", "list_slots", {})

    assert answer == {"server": "observatory", "tool": "list_slots"}
    assert session.calls == ["list_slots", "list_slots"]
