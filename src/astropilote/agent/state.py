"""Shared agent state definition (`AstroState`) and lightweight message envelopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from astropilote.agent.schemas import AstroAnswer, Chunk, Intent, ToolCallRecord


@dataclass
class AgentMessage:
    """Lightweight message envelope inside `AstroState`."""

    role: str
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class AstroState(TypedDict, total=False):
    """State dictionary shared across pipeline stages (`guard`, `agent`, `tools`, `compose`, `validate`, `refuse`)."""

    messages: list[AgentMessage]
    question: str
    llm_mode_override: Literal["api", "local"] | None
    execution_mode: Literal["agent", "rag"] | None
    strict_corpus_guard: bool | None
    date_override: str | None
    focal_length_override: float | None
    aperture_override: float | None
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
