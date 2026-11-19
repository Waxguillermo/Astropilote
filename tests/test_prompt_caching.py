"""Anthropic prompt caching on the agent's structured calls, and cache-aware cost accounting."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import astropilote.llm as llm_mod
from astropilote.config import settings
from astropilote.llm import StructuredReply, structured_reply_cost
from astropilote.tracking.tracker import _cache_read_multiplier, compute_cost_usd


def test_cache_read_multiplier_matches_dated_snapshots() -> None:
    assert _cache_read_multiplier("claude-sonnet-5-5-20261001") == 0.05
    assert _cache_read_multiplier("anthropic/claude-opus-5-5-20261101") == 0.05
    assert _cache_read_multiplier("claude-haiku-5-5-20261001") == 0.1


@pytest.mark.parametrize(
    ("model", "kwargs", "expected"),
    [
        # 1M prompt tokens on Sonnet 5.5 ($2/M input): uncached, all read (0.05x), all written (1.25x).
        ("claude-sonnet-5-5", {}, 2.0),
        ("claude-sonnet-5-5", {"cache_read_tokens": 1_000_000}, 0.1),
        ("anthropic/claude-sonnet-5-5", {"cache_write_tokens": 1_000_000}, 2.5),
        # Haiku 5.5 ($0.10/M input) reads at the default 0.1x.
        ("claude-haiku-5-5", {"cache_read_tokens": 1_000_000}, 0.01),
        # Mixed: 200k uncached + 300k written + 500k read on Haiku 5.5.
        (
            "claude-haiku-5-5",
            {"cache_write_tokens": 300_000, "cache_read_tokens": 500_000},
            0.02 + 0.0375 + 0.005,
        ),
    ],
)
def test_cost_reprices_cached_prompt_tokens(
    model: str, kwargs: dict[str, int], expected: float
) -> None:
    assert compute_cost_usd(model, 1_000_000, 0, **kwargs) == pytest.approx(expected)


def test_reply_cost_falls_back_to_plain_pricing_for_plain_tuples() -> None:
    plain = ({"ok": True}, 10_000, 1_000, "anthropic/claude-haiku-5-5")
    cached = StructuredReply(
        {"ok": True}, 10_000, 1_000, "anthropic/claude-haiku-5-5", cache_read_tokens=8_000
    )
    parsed, tokens_in, tokens_out, backend = cached  # unpacks like the 4-tuple
    assert (parsed, tokens_in, tokens_out, backend) == plain
    assert structured_reply_cost(plain) == compute_cost_usd(backend, 10_000, 1_000)
    assert structured_reply_cost(cached) < structured_reply_cost(plain)


class _FakeMessages:
    def __init__(self, reject_forced_choice: bool) -> None:
        self.calls: list[dict[str, Any]] = []
        self.reject_forced_choice = reject_forced_choice

    def create(self, **kwargs: Any) -> Any:
        from anthropic import BadRequestError

        self.calls.append(kwargs)
        if self.reject_forced_choice and kwargs["tool_choice"]["type"] == "tool":
            raise BadRequestError(
                'tool_choice: type "tool" and "any" are not supported for this model.',
                response=SimpleNamespace(status_code=400, headers={}, request=None),  # type: ignore[arg-type]  # minimal stub
                body=None,
            )
        return SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=300,
                output_tokens=50,
                cache_read_input_tokens=4_000,
                cache_creation_input_tokens=0,
            ),
            content=[SimpleNamespace(type="tool_use", input={"decision": "finish"})],
        )


@pytest.fixture
def fake_anthropic(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> _FakeMessages:
    import anthropic

    messages = _FakeMessages(reject_forced_choice=getattr(request, "param", False))
    monkeypatch.setattr(anthropic, "Anthropic", lambda **_kw: SimpleNamespace(messages=messages))
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-ant-test")
    monkeypatch.setattr(settings, "router_model", "claude-haiku-5-5")
    return messages


def _structured_call() -> Any:
    return llm_mod._call_api_structured_tool(
        user_msg="step 1 observations",
        sys_msg="SYSTEM PROMPT",
        tool_name="step",
        tool_description="d",
        tool_schema={"type": "object", "properties": {}},
    )


@pytest.mark.parametrize("fake_anthropic", [False, True], indirect=True)
def test_structured_call_caches_tools_and_system(fake_anthropic: _FakeMessages) -> None:
    reply = _structured_call()

    assert len(fake_anthropic.calls) == (2 if fake_anthropic.reject_forced_choice else 1)
    for call in fake_anthropic.calls:  # forced tool choice, and the `auto` retry when refused
        system = call["system"]
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        assert system[0]["text"].startswith("SYSTEM PROMPT")
        assert call["messages"] == [{"role": "user", "content": "step 1 observations"}]
    assert isinstance(reply, StructuredReply)
    assert tuple(reply) == ({"decision": "finish"}, 4_300, 50, "anthropic/claude-haiku-5-5")
    assert (reply.cache_read_tokens, reply.cache_write_tokens) == (4_000, 0)


def test_call_structured_llm_keeps_the_cache_usage(
    fake_anthropic: _FakeMessages, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(llm_mod, "_refresh_env_settings", lambda: None)
    reply = llm_mod.call_structured_llm(
        user_msg="u",
        sys_msg="s",
        tool_name="step",
        tool_description="d",
        tool_schema={"type": "object", "properties": {}},
        ollama_fields_hint="",
        mode_override="api",
        allow_fallback=False,
    )
    assert getattr(reply, "cache_read_tokens", None) == 4_000
