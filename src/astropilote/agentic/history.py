"""Host-side recording of verified booking writes into the actor's long-term history.

`HistoryRecordingGateway` wraps the MCP gateway like the retry and trace layers do, so the
LLM agent and the rule-based workflow both feed the history without any code change:

    TracedGateway(HistoryRecordingGateway(RetryingGateway(...), memory, store.get_state))

Only a write the backend confirmed is recorded. The event is rebuilt from the trusted
SQLite state (slot -> telescope -> site) rather than from model or tool text, and is reduced
to validated identifiers by `BookingEvent`. Recording is best effort: a failure is logged and
never undoes or hides the booking itself.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from typing import Any

from astropilote.agentic.memory import BookingEvent, UserMemory, site_key

logger = logging.getLogger(__name__)

_RECORDED_WRITES = {"create_booking": "booked", "cancel_booking": "cancelled"}


class HistoryRecordingGateway:
    def __init__(
        self, delegate: Any, memory: UserMemory, state_reader: Callable[[], dict[str, Any]]
    ) -> None:
        self.delegate = delegate
        self.memory = memory
        self._read_state = state_reader

    @property
    def last_attempts(self) -> Any:
        """Forward the retry layer's attempts so the trace layer above still sees them."""
        return getattr(self.delegate, "last_attempts", None)

    def call(self, server: str, tool: str, args: dict[str, Any]) -> Any:
        result = self.delegate.call(server, tool, args)
        if server == "observatory" and tool in _RECORDED_WRITES and isinstance(result, dict):
            self._record(_RECORDED_WRITES[tool], result)
        return result

    def _record(self, kind: str, booking: dict[str, Any]) -> None:
        try:
            state = self._read_state()
            slot = next((s for s in state["slots"] if s["id"] == booking["slot_id"]), {})
            telescope = next(
                (t for t in state["telescopes"] if t["id"] == slot.get("telescope_id")), {}
            )
            event = BookingEvent(
                kind=kind,
                booking_id=booking["id"],
                slot_id=booking["slot_id"],
                telescope_id=slot.get("telescope_id"),
                site=site_key(telescope.get("site", "")),
                start_utc=slot.get("start_utc"),
            )
            self.memory.record_event(booking["user_id"], event)
        except (KeyError, TypeError, ValueError, OSError, sqlite3.Error) as exc:
            logger.warning("Booking history not recorded (%s): %s", kind, type(exc).__name__)

    def close(self) -> None:
        if hasattr(self.delegate, "close"):
            self.delegate.close()
