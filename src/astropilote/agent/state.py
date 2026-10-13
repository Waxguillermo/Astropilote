"""Shared Google ADK agent state definition (`AstroState`) and lightweight message envelopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from astropilote.agent.schemas import AstroAnswer, Chunk, Intent, ToolCallRecord


@dataclass
class AgentMessage:
    """Lightweight message envelope replacing legacy LangChain messages inside `AstroState`."""

    role: str
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class AstroState(TypedDict):
    """State dictionary shared across Google ADK pipeline stages (`guard`, `agent`, `tools`, `compose`, `validate`, `refuse`)."""

    messages: list[AgentMessage]
    question: str
    intent: Intent | None
    retrieved: list[Chunk]
    calculated_altitudes: dict[str, list[float]]
    night_window: tuple[str, str] | None
    tool_trace: list[ToolCallRecord]
    steps: int
    answer: AstroAnswer | None
    retries: int
    tokens_in: int
    tokens_out: int
