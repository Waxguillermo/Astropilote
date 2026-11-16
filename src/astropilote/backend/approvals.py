"""Local, single-use cancellation capabilities issued ONLY by a trusted host.

This is a simulated project backend, not a remote identity/authentication service.
The model must never have access to ``issue`` or to approval tokens. Token issuance
happens only inside the checkpointed post-interrupt LangGraph write node.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
import time
from contextlib import closing
from pathlib import Path


class CancellationApprovals:
    """SQLite-backed, atomic, expiring, booking-bound, one-time authorizations."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS _cancellation_approvals (
                    token_hash TEXT PRIMARY KEY,
                    booking_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    expires_at REAL NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=10)
        db.execute("PRAGMA busy_timeout=10000")
        return db

    @staticmethod
    def _digest(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def issue(self, *, booking_id: str, user_id: str, ttl_seconds: int = 120) -> str:
        """TRUSTED CONTROLLER ONLY. Never expose this method as an MCP tool."""
        if not booking_id or not user_id or not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
            raise ValueError("Invalid approval scope or TTL")
        token = secrets.token_urlsafe(48)
        with closing(self._connect()) as db, db:
            db.execute(
                "INSERT INTO _cancellation_approvals (token_hash, booking_id, user_id, expires_at) "
                "VALUES (?,?,?,?)",
                (self._digest(token), booking_id, user_id, time.time() + ttl_seconds),
            )
        return token

    def consume(self, *, token: str | None, booking_id: str, user_id: str) -> bool:
        """Atomically consume a valid capability; scope mismatch never consumes it."""
        if not isinstance(token, str) or len(token) < 32:
            return False
        digest = self._digest(token)
        with closing(self._connect()) as db, db:
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
