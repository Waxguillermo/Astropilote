"""Tests for the LangGraph `StateGraph`, Hybrid ReAct loop, and Agentic Harness."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

import astropilote.agent.nodes as nodes_mod
import astropilote.tools as tools_mod
from astropilote.agent.graph import get_astro_graph, run_agent, run_agent_stream
from astropilote.agent.harness import (
    HarnessConfig,
    canonical_tool_signature,
    execute_tool_with_timeout,
    gate_tool_calls,
)
from astropilote.app.main import app


def test_compiled_langgraph_topology_and_mermaid_endpoint() -> None:
    """Verify `AstroWorkflow` compiles a genuine `langgraph.graph.StateGraph` and exposes `/api/graph`."""
    workflow = get_astro_graph()
    compiled_cls_mod = type(workflow.compiled_graph).__module__
    assert "langgraph" in compiled_cls_mod

    mermaid = workflow.get_mermaid()
    for node_name in ("guard", "refuse", "agent", "tools", "compose", "validate", "scrub"):
        assert node_name in mermaid

    client = TestClient(app)
    resp = client.get("/api/graph")
    assert resp.status_code == 200
    body = resp.json()
    assert body["engine"] == "langgraph"
    assert "agent" in body["nodes"]
    assert "tools" in body["nodes"]
    assert "graph TD" in body["mermaid"] or "flowchart" in body["mermaid"] or "guard" in body["mermaid"]


def test_canonical_tool_signatures_and_anti_duplicate_gate() -> None:
    """Ensure semantically equivalent tool calls normalize to identical signatures and duplicates/unknowns are blocked."""
    sig_paris_1 = canonical_tool_signature("geocode", {"place_name": "Paris"})
    sig_paris_2 = canonical_tool_signature("geocode", {"place_name": "  paris  "})
    assert sig_paris_1 == sig_paris_2

    sig_m31_1 = canonical_tool_signature("get_object", {"object_id": "M31", "focal_length_mm": 480.00001})
    sig_m31_2 = canonical_tool_signature("get_object", {"focal_length_mm": 480.0, "object_id": "m031"})
    assert sig_m31_1 == sig_m31_2

    sig_vis_1 = canonical_tool_signature(
        "visibility",
        {"object_ids": ["M42", "M31"], "lat": 48.85661, "lon": 2.35222, "obs_date": "2026-10-17"},
    )
    sig_vis_2 = canonical_tool_signature(
        "visibility",
        {"obs_date": "2026-10-17", "lon": 2.3522, "lat": 48.8566, "object_ids": ["M031", "M042"]},
    )
    assert sig_vis_1 == sig_vis_2

    cfg = HarnessConfig(max_tool_calls=5, max_identical_calls=1)
    empty_state: Any = {
        "steps": 0,
        "tool_trace": [],
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "seen_tool_signatures": [],
        "consecutive_tool_errors": 0,
    }
    proposed = [
        {"name": "geocode", "args": {"place_name": "Paris"}, "id": "c1"},
        {"name": "geocode", "args": {"place_name": " paris "}, "id": "c2"},
        {"name": "hallucinated_telescope_laser", "args": {"power": 9000}, "id": "c3"},
        {"name": "get_object", "args": {"object_id": "M31"}, "id": "c4"},
        {"name": "get_object", "args": {"object_id": "M031"}, "id": "c5"},
    ]
    res = gate_tool_calls(
        proposed_calls=proposed,
        state=empty_state,
        config=cfg,
        allowed_tools=set(tools_mod.TOOLS_BY_NAME.keys()),
    )
    assert [c["id"] for c in res.approved_calls] == ["c1", "c4"]
    assert any(ev.startswith("blocked_duplicate_call:geocode:") for ev in res.new_events)
    assert any(ev.startswith("blocked_duplicate_call:get_object:") for ev in res.new_events)
    assert "blocked_unknown_tool:hallucinated_telescope_laser" in res.new_events


def test_non_blocking_per_tool_timeout_and_error_circuit_breaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify hung tools return immediately upon `tool_timeout_s` and consecutive failures trip the circuit-breaker."""

    def _slow_tool(**_kwargs: Any) -> dict[str, Any]:
        time.sleep(1.5)
        return {"ok": True}

    t0 = time.perf_counter()
    outcome = execute_tool_with_timeout("weather", {"lat": 48.8, "lon": 2.3}, _slow_tool, timeout_s=0.06)
    elapsed = time.perf_counter() - t0

    assert outcome.status == "timeout"
    assert outcome.output["ok"] is False
    assert outcome.output["timed_out"] is True
    assert elapsed < 0.5, f"Timeout should not wait for background sleep thread (took {elapsed:.2f}s)"

    # Now test consecutive tool error circuit-breaker inside the full LangGraph workflow
    def _failing_geocode(place_name: str) -> dict[str, Any]:
        raise RuntimeError(f"Simulated network failure for {place_name}")

    def _failing_search(**_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("Simulated BM25 failure")

    patched_tools = {
        **tools_mod.TOOLS_BY_NAME,
        "geocode": _failing_geocode,
        "search_catalog": _failing_search,
    }
    monkeypatch.setattr(nodes_mod, "TOOLS_BY_NAME", patched_tools)

    _ans, state, _row = run_agent(
        "What can I observe tonight from Paris?",
        date_override="2026-10-17",
        harness_config=HarnessConfig(max_consecutive_tool_errors=2, tool_timeout_s=1.0),
    )
    assert state["consecutive_tool_errors"] >= 2
    assert state["harness_budget_tripped"] == "consecutive_tool_errors"
    assert any("circuit_breaker:consecutive_tool_errors" in ev for ev in state["harness_events"])
    assert all(tc.status == "error" for tc in state["tool_trace"])


def test_strict_step_tool_token_and_cost_budgets() -> None:
    """Verify `max_steps`, `max_tool_calls`, `max_tokens_budget`, and `max_cost_usd` budgets behave deterministically."""
    # 1. A 1-step RAG query with max_steps=1 finishes naturally (`answered`, not `max_steps`)
    ans_rag, state_rag, row_rag = run_agent(
        "What is the Andromeda Galaxy (M31)?",
        execution_mode="rag",
        harness_config=HarnessConfig(max_steps=1),
    )
    assert not ans_rag.abstained
    assert state_rag["harness_budget_tripped"] is None
    assert row_rag.stop_reason == "answered"

    # 2. A multi-step located session query with max_steps=1 trips `max_steps` cleanly after step 1
    _ans_s, state_s, row_s = run_agent(
        "What should I look at tonight in Paris?",
        date_override="2026-10-17",
        harness_config=HarnessConfig(max_steps=1),
    )
    assert state_s["harness_budget_tripped"] == "max_steps"
    assert row_s.stop_reason == "max_steps"
    assert any("budget_tripped:max_steps" in ev for ev in state_s["harness_events"])

    # 3. `max_tool_calls=2` truncates excess tool calls and caps `tool_trace` to exactly 2 calls
    _ans_tc, state_tc, row_tc = run_agent(
        "What should I look at tonight in Paris?",
        date_override="2026-10-17",
        harness_config=HarnessConfig(max_tool_calls=2),
    )
    assert len(row_tc.tool_calls) == 2
    assert any("truncated_tool_call_budget" in ev for ev in state_tc["harness_events"])

    # 4. Token and USD cost budgets trip immediately in `gate_tool_calls`
    over_cost_state: Any = {
        "steps": 0,
        "tool_trace": [],
        "tokens_in": 500,
        "tokens_out": 200,
        "cost_usd": 0.05,
        "harness_config": {"max_cost_usd": 0.02},
        "seen_tool_signatures": [],
        "consecutive_tool_errors": 0,
    }
    gate_cost = gate_tool_calls(
        proposed_calls=[{"name": "geocode", "args": {"place_name": "Paris"}}],
        state=over_cost_state,
        config=HarnessConfig(max_cost_usd=0.02),
        allowed_tools=set(tools_mod.TOOLS_BY_NAME.keys()),
    )
    assert gate_cost.approved_calls == []
    assert gate_cost.budget_tripped == "max_cost_usd"


def test_adaptive_multi_site_react_loop_and_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify multi-hop LLM ReAct planning compares Paris vs Fontainebleau and streams thoughts via SSE."""
    captured_compose_prompts: list[str] = []

    def _fake_react_planner(
        ledger: dict[str, Any],
        mode_override: str | None = None,
    ) -> tuple[dict[str, Any], int, int, str]:
        step = int(ledger.get("step", 0))
        geocoded_names = [str(g.get("short_name") or "") for g in ledger.get("geocoded_sites", [])]
        if step == 0:
            return (
                {
                    "thought": "Geocode both Paris and Fontainebleau in parallel to compare sky darkness and cloud cover.",
                    "is_complete": False,
                    "tool_calls": [
                        {"name": "geocode", "args": {"place_name": "Paris"}},
                        {"name": "geocode", "args": {"place_name": "Fontainebleau"}},
                        # Intentional duplicate + unknown tool to verify Harness filtering inside ReAct!
                        {"name": "geocode", "args": {"place_name": " paris "}},
                        {"name": "invented_satellite_tool", "args": {}},
                    ],
                },
                120,
                45,
                "claude-sonnet-4-6",
            )
        if step == 1 and len(geocoded_names) >= 2 and not ledger.get("nights_computed"):
            return (
                {
                    "thought": "Both Paris and Fontainebleau are geocoded; fetching twilight ephemeris and cloud cover for both sites.",
                    "is_complete": False,
                    "tool_calls": [],
                },
                110,
                35,
                "claude-sonnet-4-6",
            )
        return (
            {
                "thought": "Ephemeris, weather, and target altitudes are ready for both sites; proceeding to grounded synthesis.",
                "is_complete": True,
                "tool_calls": [],
            },
            95,
            25,
            "claude-sonnet-4-6",
        )

    orig_composer = nodes_mod.generate_grounded_answer

    def _capturing_composer(
        system_prompt: str,
        user_prompt: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        captured_compose_prompts.append(user_prompt)
        return orig_composer(system_prompt, user_prompt, *args, **kwargs)

    monkeypatch.setattr(nodes_mod, "plan_react_step_with_llm", _fake_react_planner)
    monkeypatch.setattr(nodes_mod, "generate_grounded_answer", _capturing_composer)

    ans, state, _row = run_agent(
        "Compare observing conditions tonight between Paris and Fontainebleau on October 17, 2026.",
        date_override="2026-10-17",
    )
    assert ans.verified
    assert len(state["react_thoughts"]) >= 2
    assert "Geocode both Paris and Fontainebleau" in state["react_thoughts"][0]
    assert any("blocked_duplicate_call:geocode:" in ev for ev in state["harness_events"])
    assert "blocked_unknown_tool:invented_satellite_tool" in state["harness_events"]

    assert captured_compose_prompts, "Composer should have been invoked"
    assert '"compared_sites"' in captured_compose_prompts[-1]
    assert "Fontainebleau" in captured_compose_prompts[-1]
    assert "Paris" in captured_compose_prompts[-1]

    # Also verify `run_agent_stream` emits live ReAct thought stage details + done payload
    stream_events = list(
        run_agent_stream(
            "Compare observing conditions tonight between Paris and Fontainebleau on October 17, 2026.",
            date_override="2026-10-17",
        )
    )
    stage_details = [ev.get("detail", "") for ev in stream_events if ev.get("event") == "stage"]
    assert any("Geocode both Paris and Fontainebleau" in d for d in stage_details)
    done_ev = next(ev for ev in stream_events if ev.get("event") == "done")
    assert done_ev["state"]["react_thoughts"]

    # And verify `/api/chat` exposes the `harness` object
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={
            "question": "Compare observing conditions tonight between Paris and Fontainebleau on October 17, 2026.",
            "mode": "agent",
            "date_override": "2026-10-17",
        },
    )
    assert resp.status_code == 200
    chat_json = resp.json()
    assert chat_json["harness"]["engine"] == "langgraph"
    assert len(chat_json["harness"]["react_thoughts"]) >= 2
    assert any("blocked_unknown_tool:invented_satellite_tool" in e for e in chat_json["harness"]["events"])
    assert all(t["status"] == "ok" for t in chat_json["tool_observations"])
    # Ensure json serialization works cleanly
    assert json.dumps(chat_json["harness"])
