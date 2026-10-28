"""Telemetry and cost tracking for token usage, latency, tool calls, and USD cost."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from astropilote.agent.schemas import ToolCallRecord

# Per-1M-token USD pricing table (local Ollama / Transformers models are $0.00)
MODEL_PRICING_PER_1M_USD: dict[str, tuple[float, float]] = {
    "gemma4": (0.0, 0.0),
    "gemma4:e2b": (0.0, 0.0),
    "gemma4:e4b": (0.0, 0.0),
    "ollama/gemma4": (0.0, 0.0),
    "HuggingFaceTB/SmolLM2-360M-Instruct": (0.0, 0.0),
    "qwen3:8b": (0.0, 0.0),
    "llama3.1:8b": (0.0, 0.0),
    "anthropic/claude-haiku-4.5": (0.80, 4.00),
    "claude-haiku-4.5": (0.80, 4.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
}


def compute_cost_usd(model_name: str, tokens_in: int, tokens_out: int) -> float:
    """Compute estimated USD cost from input/output token counts for `model_name`."""
    m_low = (model_name or "").lower()
    if m_low in MODEL_PRICING_PER_1M_USD:
        rates = MODEL_PRICING_PER_1M_USD[m_low]
    elif model_name in MODEL_PRICING_PER_1M_USD:
        rates = MODEL_PRICING_PER_1M_USD[model_name]
    elif "haiku" in m_low:
        rates = (0.80, 4.00)
    elif "sonnet" in m_low:
        rates = (3.00, 15.00)
    elif "opus" in m_low:
        rates = (15.00, 75.00)
    else:
        rates = (0.0, 0.0)
    cost = (tokens_in / 1_000_000.0) * rates[0] + (tokens_out / 1_000_000.0) * rates[1]
    return round(cost, 6)


def estimate_text_tokens(text: str) -> int:
    """Fallback token estimator (~1.33 subword tokens per whitespace word)."""
    if not text:
        return 0
    return max(1, int(round(len(text.split()) * 1.33)))


@dataclass
class RunMetrics:
    """Aggregated telemetry for a single agent run."""

    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


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
