"""Agent workflow runner (`START -> guard -> agent <-> tools -> compose -> validate -> END`)."""

from __future__ import annotations

from typing import Literal

from astropilote.agent.nodes import (
    agent_node,
    compose_node,
    guard_node,
    refuse_node,
    tools_node,
    validate_node,
)
from astropilote.agent.schemas import AstroAnswer, EvalResultRow
from astropilote.agent.state import AstroState
from astropilote.config import settings
from astropilote.tracking.tracker import LatencyTimer, compute_cost_usd


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
    """Trigger at most `max_compose_retries` (1) retry back to `compose` if verification failed."""
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


def run_agent(
    question: str,
    question_id: str = "q000",
    llm_mode: Literal["api", "local"] | None = None,
) -> tuple[AstroAnswer, AstroState, EvalResultRow]:
    """Execute the full AstroPilote workflow on `question` and return `(answer, final_state, eval_row)`."""
    graph = get_astro_graph()
    initial_state: AstroState = {
        "messages": [],
        "question": question,
        "llm_mode_override": llm_mode,
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
    }

    with LatencyTimer() as timer:
        final_state: AstroState = graph.invoke(initial_state)

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
    effective_mode = llm_mode or settings.llm_mode
    if settings.eval_offline_mode or effective_mode == "local":
        cost_usd = 0.0
    else:
        from astropilote.llm import _resolve_openrouter_model

        priced_model = (
            _resolve_openrouter_model(settings.anthropic_model)
            if settings.anthropic_api_key.strip().startswith("sk-or-")
            else settings.anthropic_model
        )
        cost_usd = compute_cost_usd(priced_model, tokens_in, tokens_out)
    retrieved_ids = list(dict.fromkeys(c.doc_id for c in final_state.get("retrieved", [])))

    eval_row = EvalResultRow(
        id=question_id,
        answer=answer.answer,
        cited_sources=answer.sources,
        retrieved_ids=retrieved_ids,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_s=timer.elapsed_s,
        tool_calls=final_state.get("tool_trace", []),
        stop_reason=stop_reason,
        cost_usd=cost_usd,
    )
    return answer, final_state, eval_row


__all__ = [
    "AstroWorkflow",
    "build_astro_graph",
    "get_astro_graph",
    "run_agent",
]
