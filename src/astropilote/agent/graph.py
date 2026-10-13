"""Compiled LangGraph workflow (`START -> guard -> agent <-> tools -> compose -> validate -> END`)."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

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


def build_astro_graph():
    """Construct and compile the AstroPilote LangGraph `StateGraph`."""
    workflow = StateGraph(AstroState)

    workflow.add_node("guard", guard_node)
    workflow.add_node("refuse", refuse_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("compose", compose_node)
    workflow.add_node("validate", validate_node)

    workflow.add_edge(START, "guard")
    workflow.add_conditional_edges(
        "guard",
        _route_after_guard,
        {"refuse": "refuse", "agent": "agent"},
    )
    workflow.add_edge("refuse", END)
    workflow.add_conditional_edges(
        "agent",
        _route_after_agent,
        {"tools": "tools", "compose": "compose"},
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("compose", "validate")
    workflow.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"compose": "compose", "__end__": END},
    )

    return workflow.compile()


_compiled_graph = None


def get_astro_graph():
    """Return a cached compiled LangGraph instance."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_astro_graph()
    return _compiled_graph


def run_agent(question: str, question_id: str = "q000") -> tuple[AstroAnswer, AstroState, EvalResultRow]:
    """Execute the full AstroPilote LangGraph on `question` and return `(answer, final_state, eval_row)`."""
    graph = get_astro_graph()
    initial_state: AstroState = {
        "messages": [],
        "question": question,
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
    cost_usd = compute_cost_usd(settings.llm_model, tokens_in, tokens_out)
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
