"""Human-approval persistence: one-time cancellation capabilities and pending requests.

This is a simulated project backend, not a remote identity/authentication service.
The model must never have access to ``issue`` or to approval tokens. Token issuance
happens only inside the checkpointed post-interrupt LangGraph write node.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Literal


def _connect(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(db_path, timeout=10)
    db.execute("PRAGMA busy_timeout=10000")
    return db


class CancellationApprovals:
    """SQLite-backed, atomic, expiring, booking-bound, one-time authorizations."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect(self.db_path)) as db, db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS _cancellation_approvals (
                    token_hash TEXT PRIMARY KEY,
                    booking_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )"""
            )

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def issue(self, *, booking_id: str, user_id: str, ttl_seconds: int = 120) -> str:
        """TRUSTED CONTROLLER ONLY. Never expose this method as an MCP tool."""
        if not booking_id or not user_id or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
            raise ValueError("Invalid approval scope or TTL")
        token = secrets.token_urlsafe(48)
        now = time.time()
        with closing(_connect(self.db_path)) as db, db:
            db.execute("DELETE FROM _cancellation_approvals WHERE expires_at <= ?", (now,))
            db.execute(
                "INSERT INTO _cancellation_approvals (token_hash, booking_id, user_id, expires_at) "
                "VALUES (?,?,?,?)",
                (self._digest(token), booking_id, user_id, now + ttl_seconds),
            )
        return token

    def consume(self, *, token: str | None, booking_id: str, user_id: str) -> bool:
        """Atomically consume a valid capability; scope mismatch never consumes it."""
        if not isinstance(token, str) or len(token) < 32:
            return False
        digest = self._digest(token)
        with closing(_connect(self.db_path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT booking_id,user_id,expires_at FROM _cancellation_approvals "
                "WHERE token_hash=?",
                (digest,),
            ).fetchone()
            if row is None or row[0] != booking_id or row[1] != user_id or row[2] <= time.time():
                db.rollback()
                return False
            db.execute("DELETE FROM _cancellation_approvals WHERE token_hash=?", (digest,))
            db.commit()
            return True


ClaimStatus = Literal["claimed", "missing", "forbidden", "expired"]


class PendingApprovals:
    """SQLite registry of threads paused on a human-approval interrupt.

    Kept on disk next to the LangGraph checkpoints so a server restart does not lose it.
    A pending request is single-use, expiring, and bound to the user who started the
    thread: an approval cannot be replayed, raced, or issued for someone else's request.
    A request leaves the registry when it is claimed (even expired) or drained, so its
    caller always learns which paused thread to discard.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect(self.db_path)) as db, db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS _pending_approvals (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    context TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )"""
            )

    def register(
        self, *, thread_id: str, user_id: str, context: dict[str, Any], ttl_seconds: int
    ) -> None:
        """Record a thread waiting for its operator decision, with JSON-serializable `context`."""
        with closing(_connect(self.db_path)) as db, db:
            db.execute(
                "INSERT OR REPLACE INTO _pending_approvals "
                "(thread_id, user_id, context, expires_at) VALUES (?,?,?,?)",
                (thread_id, user_id, json.dumps(context), time.time() + ttl_seconds),
            )

    def claim(self, *, thread_id: str, user_id: str) -> tuple[ClaimStatus, dict[str, Any] | None]:
        """Atomically take a pending request; a mismatched user never consumes it."""
        with closing(_connect(self.db_path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT user_id, context, expires_at FROM _pending_approvals WHERE thread_id=?",
                (thread_id,),
            ).fetchone()
            if row is None:
                return "missing", None
            if row[0] != user_id:
                return "forbidden", None
            db.execute("DELETE FROM _pending_approvals WHERE thread_id=?", (thread_id,))
            if row[2] <= time.time():
                return "expired", None
            return "claimed", json.loads(row[1])

    def drain(self) -> list[str]:
        """Drop every pending request and return the thread IDs that were waiting."""
        with closing(_connect(self.db_path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            thread_ids = [row[0] for row in db.execute("SELECT thread_id FROM _pending_approvals")]
            db.execute("DELETE FROM _pending_approvals")
            return thread_ids
