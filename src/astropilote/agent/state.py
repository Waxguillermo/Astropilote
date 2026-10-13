"""Shared LangGraph state definition (`AstroState`) passed between graph nodes."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from astropilote.agent.schemas import AstroAnswer, Chunk, Intent, ToolCallRecord


class AstroState(TypedDict):
    """State dictionary shared across `guard`, `agent`, `tools`, `compose`, `validate`, and `refuse`."""

    messages: Annotated[list[AnyMessage], add_messages]
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
