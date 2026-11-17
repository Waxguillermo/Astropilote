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


MAX_ERROR_MESSAGE_CHARS = 300


def error_payload(code: str, message: str, **details: Any) -> dict[str, Any]:
    """Build the standard error envelope `{ok: false, error: {code, message, details}}`."""
    return {
        "ok": False,
        "error": {
            "code": code,
            "message": message[:MAX_ERROR_MESSAGE_CHARS],
            "details": details,
        },
    }


def invoke(action: Callable[[], Any]) -> CallToolResult:
    """Turn every failure into a structured tool error, never a fake success.

    - `BackendError`: domain failure with its own code (`SLOT_UNAVAILABLE`, ...).
    - `ValueError` (excluding internal decode errors): syntactically valid input
      refused by the tool (`INVALID_ARGUMENT`, bounded message).
    - Any other exception: `TOOL_EXECUTION_ERROR` with only the exception type name,
      so neither tracebacks nor internal state leak.
    """
    try:
        return _result({"ok": True, "data": action()})
    except BackendError as exc:
        return _result(exc.as_dict(), error=True)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _result(
            error_payload(
                "TOOL_EXECUTION_ERROR",
                "The tool failed while executing.",
                exception=type(exc).__name__,
            ),
            error=True,
        )
    except ValueError as exc:
        return _result(error_payload("INVALID_ARGUMENT", str(exc)), error=True)
    except Exception as exc:
        return _result(
            error_payload(
                "TOOL_EXECUTION_ERROR",
                "The tool failed while executing.",
                exception=type(exc).__name__,
            ),
            error=True,
        )


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
