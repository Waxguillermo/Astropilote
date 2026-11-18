"""Persistent per-user memory for the simulated AstroPilote observatory.

This is **long-term application memory**, not a LangGraph checkpoint. It holds two things,
both scoped to the trusted local actor and both surviving independent runs and backend
resets:

* preferences from a closed registry (`preferred_telescope_id`, `preferred_site`,
  `preferred_city`);
* a short booking history (`booked` / `cancelled` events) recorded host-side after a
  verified write.

It never stores approvals, tokens, passwords or prompts. Every value an agent reads back
into its trusted context is an identifier matched by a regex or a member of a closed set,
so it cannot carry an injected instruction. `preferred_city` is the one free-form value
(<= 100 printable characters, no `<` / `>`) and is not exposed to any agent prompt.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_USER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_TELESCOPE_ID = re.compile(r"T-\d{2,4}\Z")
_CITY_NAME = re.compile(r"[^\x00-\x1f<>]{1,100}\Z")

SITES = ("lyon", "fontainebleau")
# Plain alias, not PEP 695 `type`: the package still supports Python 3.11.
Site = Literal["lyon", "fontainebleau"]
HISTORY_LIMIT = 20


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def site_key(label: str) -> str | None:
    """Canonical site of a telescope `site` label ("Observatoire de Lyon (simulé)" -> "lyon")."""
    if not isinstance(label, str):
        return None
    words = set(re.findall(r"[a-z]+", _fold(label)))
    matches = [site for site in SITES if site in words]
    return matches[0] if len(matches) == 1 else None


def _canonical_telescope(value: str) -> str:
    if not isinstance(value, str) or not _TELESCOPE_ID.fullmatch(value):
        raise ValueError("telescope_id must look like T-02")
    return value


def _canonical_site(value: str) -> str:
    site = _fold(value).strip() if isinstance(value, str) else ""
    if site not in SITES:
        raise ValueError(f"site must be one of {', '.join(SITES)}")
    return site


def _canonical_city(value: str) -> str:
    clean = " ".join(value.strip().split()) if isinstance(value, str) else ""
    if not clean or not _CITY_NAME.fullmatch(clean):
        raise ValueError("city must be a non-empty plain location name (<= 100 chars)")
    return clean


_PREFERENCES: dict[str, Callable[[str], str]] = {
    "preferred_telescope_id": _canonical_telescope,
    "preferred_site": _canonical_site,
    "preferred_city": _canonical_city,
}
PREFERENCE_KEYS = frozenset(_PREFERENCES)


class BookingEvent(BaseModel):
    """One verified booking write, reduced to identifiers (no free text by construction)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["booked", "cancelled"]
    booking_id: str = Field(pattern=r"^BK-\d{1,6}$")
    slot_id: str = Field(pattern=r"^S-\d{1,4}$")
    telescope_id: str | None = Field(default=None, pattern=r"^T-\d{1,4}$")
    site: Site | None = None
    start_utc: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    recorded_at: str | None = Field(
        default=None, pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$"
    )


class UserMemory:
    """SQLite preference and history store, independent from observatory state/checkpoints."""

    def __init__(self, db_path: str | Path = "data/backend/user_memory.sqlite3") -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS user_preferences ("
                "user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, "
                "updated_at TEXT NOT NULL, PRIMARY KEY (user_id, key))"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS user_events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, "
                "kind TEXT NOT NULL, booking_id TEXT NOT NULL, slot_id TEXT NOT NULL, "
                "telescope_id TEXT, site TEXT, start_utc TEXT, recorded_at TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=10)
        db.execute("PRAGMA busy_timeout=10000")
        return db

    @staticmethod
    def _actor(user_id: str) -> str:
        if not isinstance(user_id, str) or not _USER_ID.fullmatch(user_id):
            raise ValueError("user_id must be a non-empty trusted local actor identifier")
        return user_id

    @staticmethod
    def _key(key: str) -> str:
        if key not in _PREFERENCES:
            raise ValueError(f"unknown preference {key!r}")
        return key

    # --- Preferences -------------------------------------------------------------------

    def get_preference(self, user_id: str, key: str) -> str | None:
        """Read one of the actor's own preferences without exposing any other actor."""
        actor, name = self._actor(user_id), self._key(key)
        with closing(self._connect()) as db:
            found = db.execute(
                "SELECT value FROM user_preferences WHERE user_id=? AND key=?", (actor, name)
            ).fetchone()
        return found[0] if found else None

    def set_preference(self, user_id: str, key: str, value: str) -> str:
        """Store a schema-limited preference and return its canonical value.

        The caller verifies that the value exists in the observatory (via MCP) first.
        """
        actor, name = self._actor(user_id), self._key(key)
        canonical = _PREFERENCES[name](value)
        now = datetime.now(UTC).isoformat()
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO user_preferences (user_id,key,value,updated_at) "
                "VALUES (?,?,?,?) ON CONFLICT(user_id,key) DO UPDATE SET "
                "value=excluded.value, updated_at=excluded.updated_at",
                (actor, name, canonical, now),
            )
        return canonical

    def clear_preference(self, user_id: str, key: str) -> bool:
        """Forget one of the actor's own preferences when they explicitly ask."""
        actor, name = self._actor(user_id), self._key(key)
        with closing(self._connect()) as db, db:
            result = db.execute(
                "DELETE FROM user_preferences WHERE user_id=? AND key=?", (actor, name)
            )
        return result.rowcount > 0

    def preferences(self, user_id: str) -> dict[str, str]:
        actor = self._actor(user_id)
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT key, value FROM user_preferences WHERE user_id=? ORDER BY key",
                (actor,),
            ).fetchall()
        return {key: value for key, value in rows if key in _PREFERENCES}

    def preferred_telescope(self, user_id: str) -> str | None:
        return self.get_preference(user_id, "preferred_telescope_id")

    def set_preferred_telescope(self, user_id: str, telescope_id: str) -> None:
        self.set_preference(user_id, "preferred_telescope_id", telescope_id)

    def clear_preferred_telescope(self, user_id: str) -> bool:
        return self.clear_preference(user_id, "preferred_telescope_id")

    def preferred_city(self, user_id: str) -> str | None:
        return self.get_preference(user_id, "preferred_city")

    def set_preferred_city(self, user_id: str, city: str) -> str:
        return self.set_preference(user_id, "preferred_city", city)

    def clear_preferred_city(self, user_id: str) -> bool:
        return self.clear_preference(user_id, "preferred_city")

    # --- Booking history ---------------------------------------------------------------

    def record_event(self, user_id: str, event: BookingEvent) -> None:
        """Append a verified booking event, keeping only the actor's latest `HISTORY_LIMIT`."""
        actor = self._actor(user_id)
        now = datetime.now(UTC).isoformat()
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO user_events (user_id,kind,booking_id,slot_id,telescope_id,site,"
                "start_utc,recorded_at) VALUES (?,?,?,?,?,?,?,?)",
                (
                    actor,
                    event.kind,
                    event.booking_id,
                    event.slot_id,
                    event.telescope_id,
                    event.site,
                    event.start_utc,
                    now,
                ),
            )
            db.execute(
                "DELETE FROM user_events WHERE user_id=? AND id NOT IN ("
                "SELECT id FROM user_events WHERE user_id=? ORDER BY id DESC LIMIT ?)",
                (actor, actor, HISTORY_LIMIT),
            )

    def recent_events(self, user_id: str, limit: int = 5) -> list[BookingEvent]:
        """The actor's latest events, newest first (at most `HISTORY_LIMIT`)."""
        actor = self._actor(user_id)
        bounded = max(1, min(int(limit), HISTORY_LIMIT))
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT kind,booking_id,slot_id,telescope_id,site,start_utc,recorded_at "
                "FROM user_events WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (actor, bounded),
            ).fetchall()
        fields = ("kind", "booking_id", "slot_id", "telescope_id", "site", "start_utc")
        return [
            BookingEvent(**dict(zip(fields, row[:6], strict=True)), recorded_at=row[6])
            for row in rows
        ]

    def clear_history(self, user_id: str) -> int:
        actor = self._actor(user_id)
        with closing(self._connect()) as db, db:
            result = db.execute("DELETE FROM user_events WHERE user_id=?", (actor,))
        return result.rowcount

    def forget_user(self, user_id: str) -> None:
        """Drop every preference and history event of the actor."""
        actor = self._actor(user_id)
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM user_preferences WHERE user_id=?", (actor,))
            db.execute("DELETE FROM user_events WHERE user_id=?", (actor,))
