"""Agentic Harness: circuit-breakers, anti-duplicate signatures, strict budgets, and per-tool timeouts."""

from __future__ import annotations

import json
import threading
import time as time_mod
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Literal

from astropilote.agent.schemas import HarnessConfig
from astropilote.agent.state import AstroState
from astropilote.config import settings
from astropilote.tools.catalog import normalize_object_id


@dataclass(frozen=True)
class HarnessGateResult:
    """Result of gating a proposed ReAct batch of tool calls through the Agent Harness."""

    approved_calls: list[dict[str, Any]]
    seen_signatures: list[str]
    new_events: list[str]
    budget_tripped: str | None = None


@dataclass(frozen=True)
class ToolExecOutcome:
    """Bounded-timeout execution outcome for a single deterministic tool invocation."""

    output: Any
    duration_s: float
    status: Literal["ok", "timeout", "error"]


def resolve_harness_config(state: AstroState) -> HarnessConfig:
    """Build an active `HarnessConfig` merging `settings` defaults with optional per-run overrides."""
    base = {
        "max_steps": settings.max_agent_steps,
        "max_tool_calls": settings.max_tool_calls,
        "max_tokens_budget": settings.max_tokens_budget,
        "max_cost_usd": settings.max_cost_usd,
        "tool_timeout_s": settings.tool_timeout_s,
        "max_identical_calls": 1,
        "max_consecutive_tool_errors": settings.max_consecutive_tool_errors,
    }
    raw_override = state.get("harness_config")
    if isinstance(raw_override, HarnessConfig):
        return raw_override
    if isinstance(raw_override, dict):
        merged = {**base, **{k: v for k, v in raw_override.items() if v is not None}}
        return HarnessConfig(**merged)
    return HarnessConfig(**base)


def _canonicalize_arg_value(key: str, val: Any) -> Any:
    """Normalize tool argument values so semantically identical tool calls share one signature."""
    if isinstance(val, float):
        return round(val, 4)
    if isinstance(val, str):
        stripped = " ".join(val.strip().split())
        if key in ("place_name", "query"):
            return stripped.lower()
        if key == "object_id":
            return normalize_object_id(stripped)
        if key == "constellation":
            return stripped.upper()
        return stripped
    if isinstance(val, list):
        if key == "object_ids":
            return sorted(normalize_object_id(str(x)) for x in val if str(x).strip())
        if key == "filters":
            return sorted(str(x).strip().lower() for x in val if str(x).strip())
        return [_canonicalize_arg_value(key, x) for x in val]
    if isinstance(val, dict):
        return {
            str(k): _canonicalize_arg_value(str(k), v)
            for k, v in sorted(val.items())
            if v is not None
        }
    return val


def canonical_tool_signature(tool_name: str, args: dict[str, Any] | None) -> str:
    """Compute a deterministic canonical string signature `tool_name:{sorted_normalized_json}`."""
    raw_args = args or {}
    normalized = {
        str(k): _canonicalize_arg_value(str(k), v)
        for k, v in sorted(raw_args.items())
        if v is not None
    }
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return f"{tool_name.strip()}:{serialized}"


def check_harness_budgets(
    state: AstroState,
    config: HarnessConfig,
) -> tuple[bool, str | None, str | None]:
    """Verify whether `state` remains within all step, tool-call, token, cost, and error budgets."""
    steps = int(state.get("steps", 0))
    if steps >= config.max_steps:
        return (
            False,
            "max_steps",
            f"budget_tripped:max_steps ({steps}/{config.max_steps})",
        )

    total_calls = len(state.get("tool_trace", []))
    if total_calls >= config.max_tool_calls:
        return (
            False,
            "max_tool_calls",
            f"budget_tripped:max_tool_calls ({total_calls}/{config.max_tool_calls})",
        )

    total_tokens = int(state.get("tokens_in", 0)) + int(state.get("tokens_out", 0))
    if total_tokens >= config.max_tokens_budget:
        return (
            False,
            "max_tokens",
            f"budget_tripped:max_tokens ({total_tokens}/{config.max_tokens_budget})",
        )

    cost_usd = float(state.get("cost_usd", 0.0))
    enforce_cost = not settings.eval_offline_mode or state.get("harness_config") is not None
    if enforce_cost and config.max_cost_usd > 0.0 and cost_usd >= config.max_cost_usd:
        return (
            False,
            "max_cost_usd",
            f"budget_tripped:max_cost_usd (${cost_usd:.6f}>=${config.max_cost_usd:.6f})",
        )

    consec_errors = int(state.get("consecutive_tool_errors", 0))
    if consec_errors >= config.max_consecutive_tool_errors:
        return (
            False,
            "consecutive_tool_errors",
            f"circuit_breaker:consecutive_tool_errors ({consec_errors}/{config.max_consecutive_tool_errors})",
        )

    return True, None, None


def gate_tool_calls(
    proposed_calls: list[dict[str, Any]],
    state: AstroState,
    config: HarnessConfig,
    allowed_tools: set[str],
) -> HarnessGateResult:
    """Filter a batch of proposed ReAct tool calls against budgets, allowed tools, and anti-duplicate rules."""
    seen = list(state.get("seen_tool_signatures", []))
    events: list[str] = []

    if not proposed_calls:
        return HarnessGateResult(approved_calls=[], seen_signatures=seen, new_events=events)

    within_budget, tripped_kind, tripped_msg = check_harness_budgets(state, config)
    if not within_budget:
        if tripped_msg:
            events.append(tripped_msg)
        return HarnessGateResult(
            approved_calls=[],
            seen_signatures=seen,
            new_events=events,
            budget_tripped=tripped_kind,
        )

    remaining_slots = max(0, config.max_tool_calls - len(state.get("tool_trace", [])))
    approved: list[dict[str, Any]] = []

    for idx, call in enumerate(proposed_calls):
        t_name = str(call.get("name") or "").strip()
        t_args = call.get("args") if isinstance(call.get("args"), dict) else {}
        t_id = str(call.get("id") or f"call_{state.get('steps', 0)}_{idx}_{t_name}")

        if t_name not in allowed_tools:
            events.append(f"blocked_unknown_tool:{t_name or '<empty>'}")
            continue

        sig = canonical_tool_signature(t_name, t_args)
        prior_count = sum(1 for s in seen if s == sig)
        if prior_count >= config.max_identical_calls:
            events.append(f"blocked_duplicate_call:{sig}")
            continue

        if len(approved) >= remaining_slots:
            events.append(
                f"truncated_tool_call_budget:{t_name} ({config.max_tool_calls} max calls)"
            )
            continue

        seen.append(sig)
        approved.append({"name": t_name, "args": t_args, "id": t_id})

    return HarnessGateResult(
        approved_calls=approved,
        seen_signatures=seen,
        new_events=events,
        budget_tripped=None,
    )


def execute_tool_with_timeout(
    t_name: str,
    t_args: dict[str, Any],
    tool_fn: Any,
    timeout_s: float,
    search_lock: threading.Lock | None = None,
) -> ToolExecOutcome:
    """Execute one tool call under a non-blocking wall-clock timeout and error circuit-breaker."""

    def _runner() -> Any:
        if t_name == "search_catalog" and search_lock is not None:
            with search_lock:
                return tool_fn(**t_args)
        return tool_fn(**t_args)

    t0 = time_mod.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"astro-tool-{t_name}")
    future = executor.submit(_runner)
    try:
        res = future.result(timeout=timeout_s)
        dur = round(time_mod.perf_counter() - t0, 4)
        executor.shutdown(wait=False)
        return ToolExecOutcome(output=res, duration_s=dur, status="ok")
    except FuturesTimeoutError:
        dur = round(time_mod.perf_counter() - t0, 4)
        executor.shutdown(wait=False, cancel_futures=True)
        return ToolExecOutcome(
            output={
                "ok": False,
                "timed_out": True,
                "tool": t_name,
                "error": f"Tool '{t_name}' exceeded timeout of {timeout_s:.2f}s.",
            },
            duration_s=dur,
            status="timeout",
        )
    except Exception as exc:
        dur = round(time_mod.perf_counter() - t0, 4)
        executor.shutdown(wait=False)
        return ToolExecOutcome(
            output={
                "ok": False,
                "timed_out": False,
                "tool": t_name,
                "error": f"Tool '{t_name}' raised {type(exc).__name__}: {exc}",
            },
            duration_s=dur,
            status="error",
        )


__all__ = [
    "HarnessConfig",
    "HarnessGateResult",
    "ToolExecOutcome",
    "canonical_tool_signature",
    "check_harness_budgets",
    "execute_tool_with_timeout",
    "gate_tool_calls",
    "resolve_harness_config",
]
