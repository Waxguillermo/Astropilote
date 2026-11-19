"""Telemetry and cost tracking for token usage, latency, tool calls, and USD cost."""

from __future__ import annotations

import time

# Per-1M-token USD pricing table (local Ollama models are $0.00)
MODEL_PRICING_PER_1M_USD: dict[str, tuple[float, float]] = {
    "gemma4": (0.0, 0.0),
    "gemma4:e2b": (0.0, 0.0),
    "gemma4:e4b": (0.0, 0.0),
    "ollama/gemma4": (0.0, 0.0),
    "qwen3:8b": (0.0, 0.0),
    "llama3.1:8b": (0.0, 0.0),
    "anthropic/claude-haiku-4.5": (1.00, 5.00),
    "claude-haiku-4.5": (1.00, 5.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "anthropic/claude-haiku-4-5-20251001": (1.00, 5.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "anthropic/claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-sonnet-5.5": (2.00, 10.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "anthropic/claude-sonnet-5.5": (2.00, 10.00),
    "anthropic/claude-sonnet-5-5": (2.00, 10.00),
    # Prompts up to 100K tokens (agent steps stay far below; beyond: $0.50 / $2.50).
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-5.5": (0.10, 0.50),
    "anthropic/claude-haiku-5-5": (0.10, 0.50),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
}


# Anthropic prompt caching, as multipliers of the base input rate: writes with the default
# 5-minute TTL cost 1.25x; reads cost 0.1x, except 0.05x on Sonnet 5.5 and Opus 5.5.
CACHE_WRITE_MULTIPLIER = 1.25
DEFAULT_CACHE_READ_MULTIPLIER = 0.1
CACHE_READ_MULTIPLIERS: dict[str, float] = {"claude-sonnet-5-5": 0.05, "claude-opus-5-5": 0.05}


def _cache_read_multiplier(model_name: str) -> float:
    m = model_name.strip().lower().removeprefix("anthropic/").replace(".", "-")
    return CACHE_READ_MULTIPLIERS.get(m, DEFAULT_CACHE_READ_MULTIPLIER)


def compute_cost_usd(
    model_name: str,
    tokens_in: int,
    tokens_out: int,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    """Compute estimated USD cost from input/output token counts for `model_name`.

    `tokens_in` counts every prompt token, cached ones included; the `cache_read_tokens` and
    `cache_write_tokens` shares of it are re-priced at the prompt-cache rates.
    """
    if tokens_in <= 0 and tokens_out <= 0:
        return 0.0
    m_clean = (model_name or "").strip()
    m_low = m_clean.lower()
    if not m_low or m_low.startswith("ollama/"):
        return 0.0
    if m_clean in MODEL_PRICING_PER_1M_USD:
        rates = MODEL_PRICING_PER_1M_USD[m_clean]
    elif m_low in MODEL_PRICING_PER_1M_USD:
        rates = MODEL_PRICING_PER_1M_USD[m_low]
    elif "haiku" in m_low:
        rates = (0.80, 4.00)
    elif "sonnet" in m_low:
        rates = (3.00, 15.00)
    elif "opus" in m_low:
        rates = (15.00, 75.00)
    else:
        rates = (0.0, 0.0)
    uncached_in = max(0, tokens_in - cache_read_tokens - cache_write_tokens)
    input_units = (
        uncached_in
        + CACHE_WRITE_MULTIPLIER * cache_write_tokens
        + _cache_read_multiplier(m_clean) * cache_read_tokens
    )
    cost = (input_units / 1_000_000.0) * rates[0] + (tokens_out / 1_000_000.0) * rates[1]
    return round(cost, 6)


def estimate_text_tokens(text: str) -> int:
    """Fallback token estimator (~1.33 subword tokens per whitespace word)."""
    if not text:
        return 0
    return max(1, int(round(len(text.split()) * 1.33)))


class LatencyTimer:
    """Context manager measuring wall-clock execution time in seconds."""

    def __init__(self) -> None:
        self.start_time: float = 0.0
        self.elapsed_s: float = 0.0

    def __enter__(self) -> LatencyTimer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed_s = round(time.perf_counter() - self.start_time, 4)
