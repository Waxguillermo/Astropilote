"""Agent workflow runner (`START -> guard -> agent <-> tools -> compose -> validate -> END`) + SSE streaming."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

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
)
from astropilote.agent.state import AstroState
from astropilote.agent.validator import scrub_unverified_citations
from astropilote.config import settings
from astropilote.tracking.tracker import LatencyTimer


def _route_after_guard(state: AstroState) -> Literal["refuse", "agent"]:
    """Route off-topic, impossible forecasts, and prompt injections directly to `refuse`."""
    intent = state.get("intent")
    if intent and intent.request_type in ("off_topic", "impossible_forecast", "prompt_injection"):
        return "refuse"
    return "agent"


def _route_after_agent(state: AstroState) -> Literal["tools", "compose"]:
    """Continue the `agent <-> tools` loop if tool calls are pending and `steps <= max_agent_steps`."""
    messages = state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        tool_calls = getattr(last_msg, "tool_calls", []) or []
        if tool_calls and state.get("steps", 0) <= settings.max_agent_steps:
            return "tools"
    return "compose"


def _route_after_validate(state: AstroState) -> Literal["compose", "__end__"]:
    """Trigger at most `max_compose_retries` retry back to `compose` if verification failed."""
    ans = state.get("answer")
    if ans and not ans.verified and state.get("retries", 0) <= settings.max_compose_retries:
        return "compose"
    return "__end__"


class AstroWorkflow:
    """Deterministic state-machine orchestrator supporting Anthropic API and Local Ollama modes."""

    def invoke(self, initial_state: AstroState) -> AstroState:
        """Execute `START -> guard -> (refuse | agent <-> tools -> compose -> validate) -> END`."""
        state: AstroState = dict(initial_state)  # type: ignore[assignment]

        state.update(guard_node(state))  # type: ignore[arg-type]
        if _route_after_guard(state) == "refuse":
            state.update(refuse_node(state))  # type: ignore[arg-type]
            return state

        while True:
            state.update(agent_node(state))  # type: ignore[arg-type]
            if _route_after_agent(state) == "tools":
                state.update(tools_node(state))  # type: ignore[arg-type]
            else:
                break

        while True:
            state.update(compose_node(state))  # type: ignore[arg-type]
            state.update(validate_node(state))  # type: ignore[arg-type]
            if _route_after_validate(state) == "__end__":
                break

        ans = state.get("answer")
        if ans is not None and not ans.verified:
            state["answer"] = scrub_unverified_citations(ans, state)

        return state


def build_astro_graph() -> AstroWorkflow:
    """Construct and return the AstroPilote workflow orchestrator."""
    return AstroWorkflow()


_compiled_graph: AstroWorkflow | None = None


def get_astro_graph() -> AstroWorkflow:
    """Return a cached `AstroWorkflow` instance."""
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
        "intent": None,
        "retrieved": [],
        "calculated_altitudes": {},
        "night_window": None,
        "tool_trace": [],
        "steps": 0,
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
    elif final_state.get("steps", 0) > settings.max_agent_steps:
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
) -> tuple[AstroAnswer, AstroState, EvalResultRow]:
    """Execute the full AstroPilote workflow on `question` and return `(answer, final_state, eval_row)`."""
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
) -> Iterator[dict[str, Any]]:
    """Execute the AstroPilote workflow while streaming real-time stage, tool, token, and done events."""
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
    )

    with LatencyTimer() as timer:
        yield {"event": "stage", "stage": "guard", "label": "Routing query & checking guardrails"}
        state.update(guard_node(state))  # type: ignore[arg-type]

        if _route_after_guard(state) == "refuse":
            state.update(refuse_node(state))  # type: ignore[arg-type]
        else:
            while True:
                state.update(agent_node(state))  # type: ignore[arg-type]
                if _route_after_agent(state) == "tools":
                    pending_calls = getattr(state["messages"][-1], "tool_calls", []) or []
                    pending_names = ", ".join(dict.fromkeys(tc["name"] for tc in pending_calls))
                    yield {
                        "event": "stage",
                        "stage": "tools",
                        "label": f"Executing astronomy tools ({pending_names})",
                    }
                    prev_trace_len = len(state.get("tool_trace", []))
                    state.update(tools_node(state))  # type: ignore[arg-type]
                    for tc in state.get("tool_trace", [])[prev_trace_len:]:
                        yield {
                            "event": "tool",
                            "name": tc.name,
                            "duration_ms": round(tc.duration_s * 1000, 1),
                        }
                else:
                    break

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

            ans = state.get("answer")
            if ans is not None and not ans.verified:
                state["answer"] = scrub_unverified_citations(ans, state)

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
]
