"""Opt-in Arize Phoenix tracing (OpenTelemetry spans with OpenInference semantics).

`setup_tracing()` is called by every entry point (FastAPI lifespan, CLIs). It does nothing unless
`PHOENIX_ENABLED=true`; then it exports to Phoenix and auto-instruments:

* LangGraph graphs (through the LangChain callback system): one CHAIN span per graph and node;
* the Anthropic SDK (direct `sk-ant-` keys): one LLM span per `messages.create` / `.stream`.

The OpenRouter and Ollama backends are called through raw `httpx`, which no OpenInference
instrumentor covers: `llm_span` records those calls by hand. `tool_span` records each MCP tool
call made by the stdio gateway; the MCP SDK's own client spans nest under it and carry its
trace context to the server process, whose handler spans join the same trace.

Both copy the OpenInference context attributes (`using_metadata`, `using_session`, ...) that
the auto-instrumentors apply to theirs, so e.g. an A/B run's prompt variant is filterable on
every span.

`llm_span` does NOT make its span current: it wraps leaf calls, several of them inside
generators, where attaching a context would leak it to the consumer between `yield`s and detach
it from another context when the generator is closed. `tool_span` is never used in a generator
and must be current so the MCP spans become its children.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from openinference.instrumentation import (
    Message,
    TokenCount,
    Tool,
    ToolCall,
    get_attributes_from_context,
    get_input_attributes,
    get_llm_attributes,
    get_output_attributes,
    get_span_kind_attributes,
    get_tool_attributes,
)
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, Status, StatusCode

from astropilote.config import settings

logger = logging.getLogger(__name__)

_TRACER_NAME = "astropilote"
# OpenAI-style chat body keys that are not invocation parameters.
_NON_PARAM_KEYS = frozenset({"model", "messages", "tools", "stream", "stream_options"})

_setup_lock = threading.Lock()
_configured = False


def setup_tracing() -> bool:
    """Export spans to Phoenix and instrument LangGraph + Anthropic, once per process.

    Returns whether tracing is active. A Phoenix server that is down only makes the exporter
    log failures in the background; it never breaks a request.
    """
    global _configured
    if not settings.phoenix_enabled:
        return False
    with _setup_lock:
        if _configured:
            return True
        # Deferred: the exporter stack is only loaded when tracing is switched on.
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        tracer_provider = register(
            project_name=settings.phoenix_project_name,
            endpoint=settings.phoenix_collector_endpoint,
            batch=True,
            set_global_tracer_provider=True,
            verbose=False,
        )
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
        _configured = True
    logger.info(
        "Phoenix tracing enabled: project=%s endpoint=%s",
        settings.phoenix_project_name,
        settings.phoenix_collector_endpoint,
    )
    return True


def _tracer() -> trace.Tracer:
    # Resolved per call so spans follow the provider set by `setup_tracing` (or a test).
    return trace.get_tracer(_TRACER_NAME)


def _parent_context() -> Context | None:
    """Parent for a new span: the ambient OTel context, else the running LangGraph node's span.

    The LangChain instrumentor keeps node spans in LangChain's run tree without attaching them
    to the OTel context, so a span opened inside a node would otherwise start a new trace.
    `None` means "use the ambient context".
    """
    # Guarded: when not instrumented, the OpenInference lookup prints to stdout, which would
    # corrupt the stdio MCP servers' protocol stream.
    if not _configured or trace.get_current_span().get_span_context().is_valid:
        return None
    # `get_current_span` reads `langchain_core.runnables` / `.callbacks` after a bare
    # `import langchain_core`: load them here, or it raises in a process where LangGraph was
    # never imported (e.g. a direct `generate_grounded_answer` call).
    import langchain_core.callbacks
    import langchain_core.runnables.config  # noqa: F401
    from openinference.instrumentation.langchain import get_current_span as langchain_span

    node_span = langchain_span()
    return trace.set_span_in_context(node_span) if node_span is not None else None


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class LLMSpan:
    """Records the outcome of one LLM HTTP request on its span."""

    def __init__(self, span: Span) -> None:
        self.span = span

    def record_text(self, text: str) -> None:
        if self.span.is_recording():
            self.span.set_attributes(
                get_llm_attributes(output_messages=[Message(role="assistant", content=text)])
            )
            self.span.set_attributes(get_output_attributes(text))

    def record_tool_call(self, name: str, arguments: str | dict[str, Any]) -> None:
        if self.span.is_recording():
            call = ToolCall(function={"name": name, "arguments": arguments})
            self.span.set_attributes(
                get_llm_attributes(output_messages=[Message(role="assistant", tool_calls=[call])])
            )
            self.span.set_attributes(get_output_attributes(arguments))

    def record_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        if self.span.is_recording():
            self.span.set_attributes(
                get_llm_attributes(
                    token_count=TokenCount(
                        prompt=prompt_tokens,
                        completion=completion_tokens,
                        total=prompt_tokens + completion_tokens,
                    )
                )
            )

    def fail(self, description: str) -> None:
        """Mark a request that failed without raising (e.g. an HTTP error status)."""
        self.span.set_status(Status(StatusCode.ERROR, description))


@contextmanager
def llm_span(provider: str, body: Mapping[str, Any]) -> Iterator[LLMSpan]:
    """Trace one OpenAI-style chat request (`model`, `messages`, optional `tools`) as an LLM span.

    Covers OpenRouter `/chat/completions` and Ollama `/api/chat` bodies alike.
    """
    model = str(body.get("model", ""))
    span = _tracer().start_span(f"{provider}.chat {model}", context=_parent_context())
    if span.is_recording():
        tools = [Tool(json_schema=tool) for tool in body.get("tools") or []]
        span.set_attributes(get_span_kind_attributes("llm"))
        span.set_attributes(
            get_llm_attributes(
                provider=provider,
                model_name=model,
                invocation_parameters={k: v for k, v in body.items() if k not in _NON_PARAM_KEYS},
                input_messages=[
                    Message(role=str(m.get("role", "")), content=str(m.get("content", "")))
                    for m in body.get("messages") or []
                ],
                tools=tools or None,
            )
        )
        span.set_attributes(get_input_attributes(_to_json(body.get("messages") or [])))
        span.set_attributes(dict(get_attributes_from_context()))
    try:
        yield LLMSpan(span)
    except Exception as exc:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
        raise
    finally:
        # Also reached on GeneratorExit when a streaming consumer stops early: not an error.
        span.end()


class ToolSpan:
    """Records the result of one tool call on its span."""

    def __init__(self, span: Span) -> None:
        self.span = span

    def record_output(self, output: Any) -> None:
        if self.span.is_recording():
            self.span.set_attributes(
                get_output_attributes(_to_json(output), mime_type="application/json")
            )


@contextmanager
def tool_span(name: str, arguments: Mapping[str, Any]) -> Iterator[ToolSpan]:
    """Trace one tool invocation (name, JSON arguments, JSON result or exception) as a TOOL span."""
    span = _tracer().start_span(name, context=_parent_context())
    if span.is_recording():
        args_json = _to_json(dict(arguments))
        span.set_attributes(get_span_kind_attributes("tool"))
        span.set_attributes(get_tool_attributes(name=name, parameters=args_json))
        span.set_attributes(get_input_attributes(args_json, mime_type="application/json"))
        span.set_attributes(dict(get_attributes_from_context()))
    with trace.use_span(
        span, end_on_exit=True, record_exception=False, set_status_on_exception=False
    ):
        try:
            yield ToolSpan(span)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            raise


__all__ = ["LLMSpan", "ToolSpan", "llm_span", "setup_tracing", "tool_span"]
