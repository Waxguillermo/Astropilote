"""Uniform, machine-readable MCP tool responses and domain failures."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from mcp.types import CallToolResult, TextContent

from astropilote.backend.store import BackendError


def _result(payload: dict[str, Any], *, error: bool = False) -> CallToolResult:
    """Always expose the same JSON in MCP text and structured channels."""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))],
        structured_content=payload,
        is_error=error,
    )


def invoke(action: Callable[[], Any]) -> CallToolResult:
    """Turn business failures into structured tool errors, never fake successes.

    Unexpected exceptions remain exceptions: the MCP runtime will set is_error
    without leaking arbitrary internal state or treating a failure as success.
    """
    try:
        return _result({"ok": True, "data": action()})
    except BackendError as exc:
        return _result(exc.as_dict(), error=True)


def approval_required(required_action: str = "cancel_booking") -> CallToolResult:
    """Fail closed: tool-visible state cannot authorize dangerous actions."""
    return _result(
        {
            "ok": False,
            "error": {
                "code": "APPROVAL_REQUIRED",
                "message": "Dangerous action is disabled until a trusted LangGraph human approval is verified.",
                "details": {"required_action": required_action},
            },
        },
        error=True,
    )
