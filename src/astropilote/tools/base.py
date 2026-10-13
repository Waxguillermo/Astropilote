"""Decorator utility making tool functions compatible with both Google ADK (`LlmAgent`) and `.invoke({...})`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def adk_tool[F: Callable[..., Any]](fn: F) -> F:
    """Attach `.invoke(dict_args)` to a plain Python function while keeping `inspect.isfunction(fn)` True for Google ADK."""

    def _invoke(args: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        merged = dict(args or {})
        merged.update(kwargs)
        return fn(**merged)

    setattr(fn, "invoke", _invoke)
    setattr(fn, "name", fn.__name__)
    return fn
