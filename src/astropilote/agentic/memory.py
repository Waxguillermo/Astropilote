"""Persistent user preferences for the simulated AstroPilote operator CLI.

This is **long-term application memory**, not a LangGraph checkpoint. It never
stores approvals, passwords or prompts. It is scoped to the trusted local actor.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

_USER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_TELESCOPE_ID = re.compile(r"T-\d{2,4}\Z")


class UserMemory:
    """SQLite preference store, independent from observatory state/checkpoints."""

    def __init__(self, db_path: str | Path = "data/backend/user_memory.sqlite3") -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS user_preferences ("
                "user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL, "
                "updated_at TEXT NOT NULL, PRIMARY KEY (user_id, key))"
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

    def preferred_telescope(self, user_id: str) -> str | None:
        """Read a single actor's preference without exposing any other actor."""
        actor = self._actor(user_id)
        with closing(self._connect()) as db:
            found = db.execute(
                "SELECT value FROM user_preferences WHERE user_id=? AND key=?",
                (actor, "preferred_telescope_id"),
            ).fetchone()
        return found[0] if found else None

    def set_preferred_telescope(self, user_id: str, telescope_id: str) -> None:
        """Store a schema-limited preference; caller verifies ID exists via MCP."""
        actor = self._actor(user_id)
        if not isinstance(telescope_id, str) or not _TELESCOPE_ID.fullmatch(telescope_id):
            raise ValueError("telescope_id must look like T-02")
        now = datetime.now(UTC).isoformat()
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO user_preferences (user_id,key,value,updated_at) "
                "VALUES (?,?,?,?) ON CONFLICT(user_id,key) DO UPDATE SET "
                "value=excluded.value, updated_at=excluded.updated_at",
                (actor, "preferred_telescope_id", telescope_id, now),
            )

    def clear_preferred_telescope(self, user_id: str) -> bool:
        """Forget the actor's own preference when they explicitly request it."""
        actor = self._actor(user_id)
        with closing(self._connect()) as db, db:
            result = db.execute(
                "DELETE FROM user_preferences WHERE user_id=? AND key=?",
                (actor, "preferred_telescope_id"),
            )
        return result.rowcount > 0
