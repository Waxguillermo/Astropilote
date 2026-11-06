"""LangGraph `StateGraph` orchestrator (`START -> guard -> agent <-> tools -> compose -> validate -> END`) + SSE streaming."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from astropilote.agent.harness import resolve_harness_config
from astropilote.agent.nodes import (
    _normalize_conversation_history,
    agent_node,
    compose_node,
    guard_node,
    refuse_node,
    stream_compose_node,
    tools_node,
    validate_node,
)
from astropilote.agent.schemas import (
    AstroAnswer,
    ConversationTurn,
    EquipmentProfile,
    EvalResultRow,
    HarnessConfig,
)
from astropilote.agent.state import AstroState
from astropilote.agent.validator import scrub_unverified_citations
from astropilote.config import settings
from astropilote.tracking.tracker import LatencyTimer


def scrub_node(state: AstroState) -> dict[str, Any]:
    """Fallback citation scrubber node executed when `validate` exhausts `max_compose_retries`."""
    ans = state.get("answer")
    if ans is not None and not ans.verified:
        return {"answer": scrub_unverified_citations(ans, state)}
    return {}


def _guard_step(state: AstroState) -> dict[str, Any]:
    return guard_node(state)


def _refuse_step(state: AstroState) -> dict[str, Any]:
    return refuse_node(state)


def _agent_step(state: AstroState) -> dict[str, Any]:
    return agent_node(state)


def _tools_step(state: AstroState) -> dict[str, Any]:
    return tools_node(state)


def _compose_step(state: AstroState) -> dict[str, Any]:
    return compose_node(state)


def _validate_step(state: AstroState) -> dict[str, Any]:
    return validate_node(state)


def _scrub_step(state: AstroState) -> dict[str, Any]:
    return scrub_node(state)


def _route_after_guard(state: AstroState) -> Literal["refuse", "agent"]:
    """Route off-topic, impossible forecasts, and prompt injections directly to `refuse`."""
    intent = state.get("intent")
    if intent and intent.request_type in ("off_topic", "impossible_forecast", "prompt_injection"):
        return "refuse"
    return "agent"


def _route_after_agent(state: AstroState) -> Literal["tools", "compose"]:
    """Continue the `agent <-> tools` ReAct loop if approved tool calls remain within Harness budgets."""
    if state.get("harness_budget_tripped") is not None:
        return "compose"
    cfg = resolve_harness_config(state)
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        tool_calls = getattr(last_msg, "tool_calls", []) or []
        if tool_calls and state.get("steps", 0) <= cfg.max_steps:
            return "tools"
    return "compose"


def _route_after_agent_pre_compose(state: AstroState) -> Literal["tools", "__end__"]:
    """Conditional edge for the streaming pre-compose ReAct subgraph."""
    return "tools" if _route_after_agent(state) == "tools" else "__end__"


def _route_after_validate(state: AstroState) -> Literal["compose", "scrub", "__end__"]:
    """Trigger up to `max_compose_retries` back to `compose` if verification failed, else `scrub` or `END`."""
    ans = state.get("answer")
    if ans is None or ans.verified:
        return "__end__"
    if state.get("retries", 0) <= settings.max_compose_retries and state.get(
        "harness_budget_tripped"
    ) not in ("max_tokens", "max_cost_usd"):
        return "compose"
    return "scrub"


def _build_full_state_graph() -> Any:
    """Compile the complete AstroPilote `langgraph.graph.StateGraph(AstroState)`."""
    builder: StateGraph = StateGraph(AstroState)
    builder.add_node("guard", _guard_step)
    builder.add_node("refuse", _refuse_step)
    builder.add_node("agent", _agent_step)
    builder.add_node("tools", _tools_step)
    builder.add_node("compose", _compose_step)
    builder.add_node("validate", _validate_step)
    builder.add_node("scrub", _scrub_step)

    builder.add_edge(START, "guard")
    builder.add_conditional_edges(
        "guard",
        _route_after_guard,
        {"refuse": "refuse", "agent": "agent"},
    )
    builder.add_edge("refuse", END)
    builder.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "compose": "compose"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("compose", "validate")
    builder.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"compose": "compose", "scrub": "scrub", "__end__": END},
    )
    builder.add_edge("scrub", END)
    return builder.compile()


def _build_react_pre_compose_graph() -> Any:
    """Compile the `guard -> (refuse | agent <-> tools)` LangGraph subgraph for live SSE streaming."""
    builder: StateGraph = StateGraph(AstroState)
    builder.add_node("guard", _guard_step)
    builder.add_node("refuse", _refuse_step)
    builder.add_node("agent", _agent_step)
    builder.add_node("tools", _tools_step)

    builder.add_edge(START, "guard")
    builder.add_conditional_edges(
        "guard",
        _route_after_guard,
        {"refuse": "refuse", "agent": "agent"},
    )
    builder.add_edge("refuse", END)
    builder.add_conditional_edges(
        "agent",
        _route_after_agent_pre_compose,
        {"tools": "tools", "__end__": END},
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


class AstroWorkflow:
    """Compiled LangGraph `StateGraph` orchestrator with ReAct loop & Agent Harness enforcement."""

    def __init__(self) -> None:
        self._compiled = _build_full_state_graph()
        self._react_compiled = _build_react_pre_compose_graph()

    @property
    def compiled_graph(self) -> Any:
        """Expose the underlying compiled LangGraph `CompiledStateGraph`."""
        return self._compiled

    def _langgraph_config(self, state: AstroState) -> dict[str, Any]:
        cfg = resolve_harness_config(state)
        return {"recursion_limit": max(32, cfg.max_steps * 4 + 16)}

    def invoke(self, initial_state: AstroState) -> AstroState:
        """Execute `START -> guard -> (refuse | agent <-> tools -> compose -> validate -> scrub?) -> END`."""
        out: AstroState = self._compiled.invoke(
            initial_state,
            config=self._langgraph_config(initial_state),
        )
        return out

    def stream_react(self, initial_state: AstroState) -> Iterator[dict[str, Any]]:
        """Stream node updates from the compiled LangGraph `guard -> (refuse | agent <-> tools)` subgraph."""
        yield from self._react_compiled.stream(
            initial_state,
            config=self._langgraph_config(initial_state),
            stream_mode="updates",
        )

    def get_mermaid(self) -> str:
        """Return the Mermaid diagram definition of the compiled LangGraph `StateGraph`."""
        return str(self._compiled.get_graph().draw_mermaid())


def build_astro_graph() -> AstroWorkflow:
    """Construct and compile the AstroPilote LangGraph workflow."""
    return AstroWorkflow()


_compiled_graph: AstroWorkflow | None = None


def get_astro_graph() -> AstroWorkflow:
    """Return a cached compiled `AstroWorkflow` instance."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_astro_graph()
    return _compiled_graph


def _build_initial_state(
    *,
    question: str,
    llm_mode: Literal["api", "local"] | None,
    execution_mode: Literal["agent", "rag"] | None,
    mode: Literal["agent", "rag"] | None,
    date_override: str | None,
    focal_length_mm: float | None,
    aperture_mm: float | None,
    equipment_profile: EquipmentProfile | dict[str, Any] | None,
    strict_corpus_guard: bool | None,
    conversation_history: list[ConversationTurn] | list[dict[str, Any]] | None,
    harness_config: HarnessConfig | dict[str, Any] | None = None,
) -> AstroState:
    """Build a normalized `AstroState` dictionary for `run_agent` and `run_agent_stream`."""
    return {
        "messages": [],
        "question": question,
        "conversation_history": _normalize_conversation_history(conversation_history),
        "llm_mode_override": llm_mode,
        "execution_mode": execution_mode or mode,
        "strict_corpus_guard": strict_corpus_guard,
        "date_override": date_override,
        "focal_length_override": focal_length_mm,
        "aperture_override": aperture_mm,
        "equipment_profile": equipment_profile,
        "harness_config": harness_config,
        "intent": None,
        "retrieved": [],
        "calculated_altitudes": {},
        "night_window": None,
        "tool_trace": [],
        "steps": 0,
        "seen_tool_signatures": [],
        "consecutive_tool_errors": 0,
        "harness_events": [],
        "harness_budget_tripped": None,
        "react_thoughts": [],
        "answer": None,
        "retries": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
    }


def _finalize_agent_result(
    final_state: AstroState,
    question_id: str,
    elapsed_s: float,
) -> tuple[AstroAnswer, AstroState, EvalResultRow]:
    """Assemble `(answer, final_state, eval_row)` with exact per-call accumulated `cost_usd`."""
    answer: AstroAnswer = final_state.get("answer") or AstroAnswer(
        answer="Agent failed to produce an answer.",
        sources=[],
        abstained=True,
        verified=False,
    )
    intent = final_state.get("intent")
    if intent and intent.request_type in ("off_topic", "impossible_forecast", "prompt_injection"):
        stop_reason: Literal["answered", "abstained", "refused", "max_steps", "error"] = "refused"
    elif answer.abstained:
        stop_reason = "abstained"
    elif final_state.get("harness_budget_tripped") == "max_steps":
        stop_reason = "max_steps"
    else:
        stop_reason = "answered"

    tokens_in = int(final_state.get("tokens_in", 0))
    tokens_out = int(final_state.get("tokens_out", 0))
    cost_usd = (
        0.0 if settings.eval_offline_mode else round(float(final_state.get("cost_usd", 0.0)), 6)
    )
    retrieved_ids = list(dict.fromkeys(c.doc_id for c in final_state.get("retrieved", [])))

    eval_row = EvalResultRow(
        id=question_id,
        answer=answer.answer,
        cited_sources=answer.sources,
        retrieved_ids=retrieved_ids,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_s=elapsed_s,
        tool_calls=final_state.get("tool_trace", []),
        stop_reason=stop_reason,
        cost_usd=cost_usd,
    )
    return answer, final_state, eval_row


def run_agent(
    question: str,
    question_id: str = "q000",
    llm_mode: Literal["api", "local"] | None = None,
    execution_mode: Literal["agent", "rag"] | None = None,
    mode: Literal["agent", "rag"] | None = None,
    date_override: str | None = None,
    focal_length_mm: float | None = None,
    aperture_mm: float | None = None,
    equipment_profile: EquipmentProfile | dict[str, Any] | None = None,
    strict_corpus_guard: bool | None = None,
    conversation_history: list[ConversationTurn] | list[dict[str, Any]] | None = None,
    harness_config: HarnessConfig | dict[str, Any] | None = None,
) -> tuple[AstroAnswer, AstroState, EvalResultRow]:
    """Execute the full compiled LangGraph workflow on `question` and return `(answer, final_state, eval_row)`."""
    graph = get_astro_graph()
    initial_state = _build_initial_state(
        question=question,
        llm_mode=llm_mode,
        execution_mode=execution_mode,
        mode=mode,
        date_override=date_override,
        focal_length_mm=focal_length_mm,
        aperture_mm=aperture_mm,
        equipment_profile=equipment_profile,
        strict_corpus_guard=strict_corpus_guard,
        conversation_history=conversation_history,
        harness_config=harness_config,
    )

    with LatencyTimer() as timer:
        final_state: AstroState = graph.invoke(initial_state)

    return _finalize_agent_result(final_state, question_id, timer.elapsed_s)


def run_agent_stream(
    question: str,
    question_id: str = "q000",
    llm_mode: Literal["api", "local"] | None = None,
    execution_mode: Literal["agent", "rag"] | None = None,
    mode: Literal["agent", "rag"] | None = None,
    date_override: str | None = None,
    focal_length_mm: float | None = None,
    aperture_mm: float | None = None,
    equipment_profile: EquipmentProfile | dict[str, Any] | None = None,
    strict_corpus_guard: bool | None = None,
    conversation_history: list[ConversationTurn] | list[dict[str, Any]] | None = None,
    harness_config: HarnessConfig | dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Execute the LangGraph ReAct + Harness workflow while streaming real-time SSE stage, tool, token, and done events."""
    graph = get_astro_graph()
    state: AstroState = _build_initial_state(
        question=question,
        llm_mode=llm_mode,
        execution_mode=execution_mode,
        mode=mode,
        date_override=date_override,
        focal_length_mm=focal_length_mm,
        aperture_mm=aperture_mm,
        equipment_profile=equipment_profile,
        strict_corpus_guard=strict_corpus_guard,
        conversation_history=conversation_history,
        harness_config=harness_config,
    )

    with LatencyTimer() as timer:
        yield {"event": "stage", "stage": "guard", "label": "Routing query & checking guardrails"}
        was_refused = False

        for update_frame in graph.stream_react(state):
            for node_name, node_delta in update_frame.items():
                prev_trace_len = len(state.get("tool_trace", []))
                if isinstance(node_delta, dict):
                    state.update(node_delta)  # type: ignore[arg-type]

                if node_name == "refuse":
                    was_refused = True
                elif node_name == "agent":
                    messages = state.get("messages", [])
                    pending_calls = (
                        (getattr(messages[-1], "tool_calls", []) or []) if messages else []
                    )
                    thoughts = state.get("react_thoughts", [])
                    latest_thought = thoughts[-1] if thoughts else ""
                    if pending_calls:
                        pending_names = ", ".join(dict.fromkeys(tc["name"] for tc in pending_calls))
                        step_num = int(state.get("steps", 1))
                        stage_ev: dict[str, Any] = {
                            "event": "stage",
                            "stage": f"tools_step_{step_num}",
                            "label": f"ReAct Step {step_num} · Executing tools ({pending_names})",
                        }
                        if latest_thought:
                            stage_ev["detail"] = latest_thought
                        yield stage_ev
                elif node_name == "tools":
                    for tc in state.get("tool_trace", [])[prev_trace_len:]:
                        yield {
                            "event": "tool",
                            "name": tc.name,
                            "status": tc.status,
                            "duration_ms": round(tc.duration_s * 1000, 1),
                        }

        if not was_refused:
            yield {
                "event": "stage",
                "stage": "compose",
                "label": "Synthesizing grounded astronomy guide",
            }
            for delta, compose_update in stream_compose_node(state):
                if delta:
                    yield {"event": "token", "delta": delta}
                if compose_update is not None:
                    state.update(compose_update)  # type: ignore[arg-type]

            state.update(validate_node(state))  # type: ignore[arg-type]

            while _route_after_validate(state) == "compose":
                yield {
                    "event": "stage",
                    "stage": "compose",
                    "label": "Re-verifying citations & schedule",
                }
                state.update(compose_node(state))  # type: ignore[arg-type]
                state.update(validate_node(state))  # type: ignore[arg-type]

            if _route_after_validate(state) == "scrub":
                state.update(scrub_node(state))  # type: ignore[arg-type]

    final_answer, state, eval_row = _finalize_agent_result(state, question_id, timer.elapsed_s)
    yield {
        "event": "done",
        "answer": final_answer,
        "state": state,
        "eval_row": eval_row,
    }


__all__ = [
    "AstroWorkflow",
    "build_astro_graph",
    "get_astro_graph",
    "run_agent",
    "run_agent_stream",
    "scrub_node",
]
