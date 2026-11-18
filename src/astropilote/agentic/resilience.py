"""Bounded recovery from transient MCP failures, plus deterministic fault injection.

Both classes wrap any gateway exposing `call(server, tool, args)` / `close()`, so they compose
around `StdioMCPGateway` without touching the agents that consume it:

    TracedGateway(RetryingGateway(FaultInjectingGateway(StdioMCPGateway(...))))

Recovery rules:
* only idempotent reads are replayed. A timed-out `create_booking` may already have committed
  (replaying could double-book) and `cancel_booking` carries a single-use approval token;
* only transient failures are replayed: a transport timeout / broken pipe, or a `RATE_LIMITED`
  (HTTP 429) answer. Domain errors (`SLOT_UNAVAILABLE`, ...) are final;
* waits are deterministic exponential backoff (no jitter: a single local client gains nothing
  from de-synchronising retries, and reproducible traces matter more), bounded per call;
* once a tool exhausts its attempts, its circuit opens for the wrapper's lifetime, so a planner
  that calls it again fails fast instead of waiting through the backoff again.

Retries happen below the planner and therefore never consume `max_steps`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from astropilote.agentic.workflow import MCPTransportError, ToolFailure
from astropilote.mcp.common import error_payload

logger = logging.getLogger(__name__)

RETRYABLE_TOOLS = frozenset(
    {
        "get_observatory_state",
        "list_telescopes",
        "list_slots",
        "get_booking",
        "geocode",
        "weather",
        "night",
        "visibility",
        "get_object",
        "search_catalog",
    }
)
TRANSIENT_CODES = frozenset({"TIMED_OUT", "MCP_TRANSPORT_ERROR", "RATE_LIMITED"})
CIRCUIT_OPEN = "CIRCUIT_OPEN"

FaultKind = Literal["timeout", "rate_limit"]
CallOutcome = Literal["ok", "recovered", "exhausted", "not_retried", "failed", "circuit_open"]


class FaultSpec(BaseModel):
    """One injected failure: `tool` fails `times` times in a row, then behaves normally."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: str = Field(min_length=1)
    kind: FaultKind
    times: int = Field(default=1, ge=1, le=5)
    retry_after_s: float | None = Field(default=None, ge=0.0, le=5.0)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_backoff_s: float = 0.2
    multiplier: float = 2.0
    max_backoff_s: float = 2.0
    max_total_wait_s: float = 5.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if min(self.base_backoff_s, self.max_backoff_s, self.max_total_wait_s) < 0:
            raise ValueError("backoff durations must be non-negative")
        if self.multiplier < 1:
            raise ValueError("multiplier must be >= 1")

    def backoff(self, failures: int, retry_after_s: float | None = None) -> float:
        """Wait before the next attempt after `failures` consecutive failures (1-based)."""
        if retry_after_s is not None:
            return min(self.max_backoff_s, retry_after_s)
        return min(self.max_backoff_s, self.base_backoff_s * self.multiplier ** (failures - 1))


def _failure_code(exc: MCPTransportError | ToolFailure) -> str:
    if isinstance(exc, MCPTransportError):
        return exc.code
    error = exc.payload.get("error")
    return str(error.get("code", "TOOL_ERROR")) if isinstance(error, dict) else "TOOL_ERROR"


def _retry_after(exc: MCPTransportError | ToolFailure) -> float | None:
    if not isinstance(exc, ToolFailure):
        return None
    error = exc.payload.get("error")
    details = error.get("details") if isinstance(error, dict) else None
    value = details.get("retry_after_s") if isinstance(details, dict) else None
    return float(value) if isinstance(value, int | float) and value >= 0 else None


class RetryingGateway:
    """Replay idempotent reads after transient failures, recording every attempt."""

    def __init__(
        self,
        delegate: Any,
        policy: RetryPolicy | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.delegate = delegate
        self.policy = policy or RetryPolicy()
        self._sleep = sleep
        self._open_circuits: set[tuple[str, str]] = set()
        self.log: list[dict[str, Any]] = []
        self.last_attempts: list[dict[str, Any]] = []

    def call(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        attempts: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"server": server, "tool": tool, "attempts": attempts}
        self.log.append(entry)
        self.last_attempts = attempts

        if (server, tool) in self._open_circuits:
            attempts.append({"n": 1, "code": CIRCUIT_OPEN, "backoff_s": 0.0})
            entry["outcome"] = "circuit_open"
            raise MCPTransportError(CIRCUIT_OPEN, f"{server}.{tool} already exhausted its retries")

        waited = 0.0
        for n in range(1, self.policy.max_attempts + 1):
            try:
                result = self.delegate.call(server, tool, args)
            except (MCPTransportError, ToolFailure) as exc:
                code = _failure_code(exc)
                if code not in TRANSIENT_CODES:
                    attempts.append({"n": n, "code": code, "backoff_s": 0.0})
                    entry["outcome"] = "failed"
                    raise
                if tool not in RETRYABLE_TOOLS:
                    attempts.append({"n": n, "code": code, "backoff_s": 0.0})
                    entry["outcome"] = "not_retried"
                    logger.warning(
                        "%s.%s failed with %s; writes are never replayed", server, tool, code
                    )
                    raise
                delay = self.policy.backoff(n, _retry_after(exc))
                if n == self.policy.max_attempts or waited + delay > self.policy.max_total_wait_s:
                    attempts.append({"n": n, "code": code, "backoff_s": 0.0})
                    entry["outcome"] = "exhausted"
                    self._open_circuits.add((server, tool))
                    logger.warning(
                        "%s.%s still failing (%s) after %d attempts", server, tool, code, n
                    )
                    if isinstance(exc, MCPTransportError):
                        raise
                    raise MCPTransportError(code, f"{server}.{tool} after {n} attempts") from exc
                attempts.append({"n": n, "code": code, "backoff_s": delay})
                logger.warning(
                    "%s.%s failed with %s; retry %d in %.2fs", server, tool, code, n, delay
                )
                self._sleep(delay)
                waited += delay
                continue
            except Exception:
                entry["outcome"] = "failed"
                raise
            attempts.append({"n": n, "code": "ok", "backoff_s": 0.0})
            entry["outcome"] = "recovered" if n > 1 else "ok"
            return result
        raise AssertionError("unreachable: the last attempt always returns or raises")

    def close(self) -> None:
        if hasattr(self.delegate, "close"):
            self.delegate.close()


@dataclass
class _PendingFault:
    spec: FaultSpec
    remaining: int


class FaultInjectingGateway:
    """Raise scripted transient failures before the call reaches the server (no side effect).

    `timeout` mimics the transport (`MCPTransportError("TIMED_OUT")`); `rate_limit` mimics a
    server answering HTTP 429 with a structured `RATE_LIMITED` tool error.
    """

    def __init__(self, delegate: Any, faults: Iterable[FaultSpec] = ()) -> None:
        self.delegate = delegate
        self._pending = [_PendingFault(spec, spec.times) for spec in faults]
        self.triggered: list[dict[str, str]] = []

    @property
    def declared(self) -> int:
        return sum(fault.spec.times for fault in self._pending)

    def call(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        fault = next((f for f in self._pending if f.spec.tool == tool and f.remaining > 0), None)
        if fault is None:
            return self.delegate.call(server, tool, args)
        fault.remaining -= 1
        self.triggered.append({"server": server, "tool": tool, "kind": fault.spec.kind})
        if fault.spec.kind == "timeout":
            raise MCPTransportError("TIMED_OUT", f"{server}.{tool} (injected fault)")
        details = (
            {} if fault.spec.retry_after_s is None else {"retry_after_s": fault.spec.retry_after_s}
        )
        raise ToolFailure(
            error_payload("RATE_LIMITED", "429 Too Many Requests (injected fault)", **details)
        )

    def close(self) -> None:
        if hasattr(self.delegate, "close"):
            self.delegate.close()
