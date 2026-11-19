"""Phoenix tracing: OpenInference LLM/TOOL spans for the raw-httpx backends and MCP tool calls."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from openinference.semconv.trace import SpanAttributes
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import astropilote.llm as llm_mod
import astropilote.tracing as tracing_mod
from astropilote.config import settings
from astropilote.tracing import llm_span, setup_tracing, tool_span

_OUT_0 = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.message"


@pytest.fixture
def provider() -> TracerProvider:
    return TracerProvider()


@pytest.fixture
def exporter(provider: TracerProvider, monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Route this module's spans to memory without touching the global tracer provider."""
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    monkeypatch.setattr(tracing_mod, "_tracer", lambda: provider.get_tracer("test"))
    return memory


def _only_span(exporter: InMemorySpanExporter) -> ReadableSpan:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1, [s.name for s in spans]
    return spans[0]


def _mock_client(body: str, status: int = 200) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda _req: httpx.Response(status, text=body))
    )


def test_setup_tracing_is_a_noop_when_disabled() -> None:
    assert settings.phoenix_enabled is False
    assert setup_tracing() is False


def test_llm_span_records_request_output_and_usage(exporter: InMemorySpanExporter) -> None:
    body = {
        "model": "anthropic/claude-haiku-4.5",
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ],
    }
    with llm_span("openrouter", body) as call:
        call.record_text("hi")
        call.record_usage(12, 3)

    attrs = _only_span(exporter).attributes or {}
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "LLM"
    assert attrs[SpanAttributes.LLM_PROVIDER] == "openrouter"
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "anthropic/claude-haiku-4.5"
    assert json.loads(str(attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS])) == {"temperature": 0.0}
    assert attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.message.content"] == "hello"
    assert attrs[f"{_OUT_0}.content"] == "hi"
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 12
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 15


def test_llm_span_marks_exceptions_as_errors(exporter: InMemorySpanExporter) -> None:
    with pytest.raises(RuntimeError), llm_span("ollama", {"model": "gemma4", "messages": []}):
        raise RuntimeError("daemon down")

    span = _only_span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert [e.name for e in span.events] == ["exception"]


_OLLAMA_STREAM = "\n".join(
    json.dumps(line)
    for line in (
        {"message": {"content": "M42 "}},
        {"message": {"content": "rises at 21h."}},
        {"done": True, "prompt_eval_count": 40, "eval_count": 7},
    )
)
_OLLAMA_PAYLOAD: dict[str, Any] = {
    "model": "gemma4",
    "stream": True,
    "options": {"temperature": 0.2},
    "messages": [{"role": "user", "content": "When does M42 rise?"}],
}


def test_ollama_stream_produces_one_llm_span(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm_mod, "_get_ollama_client", lambda: _mock_client(_OLLAMA_STREAM))

    items = list(llm_mod._iter_ollama_ndjson(_OLLAMA_PAYLOAD))

    assert items[-1] == ("", 40, 7)
    span = _only_span(exporter)
    attrs = span.attributes or {}
    assert span.status.status_code is not StatusCode.ERROR
    assert attrs[SpanAttributes.LLM_PROVIDER] == "ollama"
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "gemma4"
    assert attrs[f"{_OUT_0}.content"] == "M42 rises at 21h."
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 7


def test_stream_closed_early_ends_span_without_error(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm_mod, "_get_ollama_client", lambda: _mock_client(_OLLAMA_STREAM))

    stream = llm_mod._iter_ollama_ndjson(_OLLAMA_PAYLOAD)
    assert next(stream) == ("M42 ", 0, 0)
    stream.close()

    assert _only_span(exporter).status.status_code is not StatusCode.ERROR


def test_ollama_http_error_is_an_error_span(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm_mod, "_get_ollama_client", lambda: _mock_client("boom", status=500))

    with pytest.raises(httpx.HTTPStatusError):
        list(llm_mod._iter_ollama_ndjson(_OLLAMA_PAYLOAD))

    assert _only_span(exporter).status.status_code is StatusCode.ERROR


def test_openrouter_structured_call_records_tool_call(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"function": {"name": "route", "arguments": '{"request_type": "factual"}'}}
                    ]
                }
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 9},
    }
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-or-test")
    monkeypatch.setattr(
        llm_mod, "_get_openrouter_client", lambda: _mock_client(json.dumps(response))
    )

    parsed, in_tok, out_tok, _ = llm_mod._call_api_structured_tool(
        user_msg="What is M42?",
        sys_msg="Route the request.",
        tool_name="route",
        tool_description="Route a request.",
        tool_schema={"type": "object", "properties": {"request_type": {"type": "string"}}},
    )

    assert (parsed, in_tok, out_tok) == ({"request_type": "factual"}, 100, 9)
    attrs = _only_span(exporter).attributes or {}
    assert attrs[SpanAttributes.LLM_PROVIDER] == "openrouter"
    assert attrs[f"{_OUT_0}.tool_calls.0.tool_call.function.name"] == "route"
    assert f"{SpanAttributes.LLM_TOOLS}.0.tool.json_schema" in attrs
    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT in attrs


def test_openrouter_http_error_is_an_error_span(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-or-test")
    monkeypatch.setattr(
        llm_mod, "_get_openrouter_client", lambda: _mock_client("quota", status=429)
    )

    with pytest.raises(RuntimeError, match="429"):
        llm_mod._call_api_llm("sys", "hello")

    span = _only_span(exporter)
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "OpenRouter HTTP 429"


def test_tool_span_records_arguments_and_result(exporter: InMemorySpanExporter) -> None:
    with tool_span("mcp.sky.get_object", {"name": "M42"}) as span:
        span.record_output({"id": "M42", "magnitude": 4.0})

    attrs = _only_span(exporter).attributes or {}
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "TOOL"
    assert attrs[SpanAttributes.TOOL_NAME] == "mcp.sky.get_object"
    assert json.loads(str(attrs[SpanAttributes.INPUT_VALUE])) == {"name": "M42"}
    assert json.loads(str(attrs[SpanAttributes.OUTPUT_VALUE])) == {"id": "M42", "magnitude": 4.0}


def test_gateway_call_is_traced_as_a_tool_span(
    exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from astropilote.agentic.workflow import StdioMCPGateway, ToolFailure

    gateway = StdioMCPGateway(tmp_path / "db.sqlite3")
    failure = {"ok": False, "error": {"code": "SLOT_UNAVAILABLE"}}

    def _fail(*_args: Any) -> dict[str, Any]:
        raise ToolFailure(failure)

    monkeypatch.setattr(gateway, "_call", _fail)
    try:
        with pytest.raises(ToolFailure):
            gateway.call("observatory", "book_slot", {"slot_id": "S1"})
    finally:
        gateway.close()

    span = _only_span(exporter)
    assert span.name == "mcp.observatory.book_slot"
    assert span.status.status_code is StatusCode.ERROR


def test_spans_opened_inside_a_langgraph_node_nest_under_the_node(
    provider: TracerProvider, exporter: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LangChain instrumentor does not attach node spans to the OTel context: without the
    fallback in `_parent_context`, a tool call made by a node would start a separate trace."""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph
    from openinference.instrumentation.langchain import LangChainInstrumentor

    class State(TypedDict):
        n: int

    def read(state: State) -> State:
        with tool_span("mcp.observatory.list_slots", {}) as span:
            span.record_output([])
        return state

    builder = StateGraph(State)
    builder.add_node("read", read)
    builder.add_edge(START, "read")
    builder.add_edge("read", END)
    graph = builder.compile()

    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    monkeypatch.setattr(tracing_mod, "_configured", True)
    try:
        graph.invoke({"n": 1})
    finally:
        instrumentor.uninstrument()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    tool, node = spans["mcp.observatory.list_slots"], spans["read"]
    assert tool.parent is not None
    assert tool.parent.span_id == node.context.span_id
    assert tool.context.trace_id == node.context.trace_id


def test_parent_lookup_works_in_a_process_that_never_imported_langgraph() -> None:
    """OpenInference's LangChain span lookup reads `langchain_core.runnables` after a bare
    `import langchain_core`; it must not crash an LLM call made outside any graph."""
    import subprocess
    import sys

    script = (
        "import sys\n"
        "import astropilote.tracing as t\n"
        "from openinference.instrumentation.langchain import LangChainInstrumentor\n"
        "assert 'langgraph' not in sys.modules\n"
        "LangChainInstrumentor().instrument()\n"
        "t._configured = True\n"
        "with t.llm_span('ollama', {'model': 'gemma4', 'messages': []}):\n"
        "    pass\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_manual_spans_carry_the_openinference_context_metadata(
    exporter: InMemorySpanExporter,
) -> None:
    """A/B runs tag every span with their prompt variant through `using_metadata`."""
    from openinference.instrumentation import using_metadata

    with using_metadata({"prompt_variant": "optimized", "prompt_sha256": "abc123"}):
        with llm_span("openrouter", {"model": "m", "messages": []}):
            pass
        with tool_span("mcp.sky.get_object", {"name": "M42"}):
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    for span in spans:
        metadata = json.loads(str((span.attributes or {})[SpanAttributes.METADATA]))
        assert metadata == {"prompt_variant": "optimized", "prompt_sha256": "abc123"}
