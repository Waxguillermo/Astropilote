"""SQLite-backed simulated observatory: deterministic state, no real payments.

The LangGraph + MCP layer must gate dangerous operations (notably cancellation)
with a trusted human approval interrupt BEFORE invoking this backend. A boolean
supplied by an LLM is not an authorization mechanism.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class BackendError(Exception):
    """Domain failure that MCP can serialize into a structured tool error."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": {"code": self.code, "message": str(self), "details": self.details}}


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BackendError("INVALID_ARGUMENT", f"{field} must be a non-empty string", field=field)
    return value.strip()


def _utc(value: Any) -> str:
    if not isinstance(value, str):
        raise BackendError("INVALID_SEED", "Slot times must be ISO 8601 strings")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BackendError("INVALID_SEED", f"Invalid datetime: {value}") from exc
    if parsed.tzinfo is None:
        raise BackendError("INVALID_SEED", "Slot times must include a timezone")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _check_seed(seed: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the entire seed before touching the database."""
    if not isinstance(seed, dict):
        raise BackendError("INVALID_SEED", "Seed root must be an object")
    for key in ("telescopes", "slots", "bookings"):
        if not isinstance(seed.get(key), list):
            raise BackendError("INVALID_SEED", f"Seed must contain an array: {key}")

    telescopes: list[dict[str, Any]] = []
    telescope_ids: set[str] = set()
    for item in seed["telescopes"]:
        if not isinstance(item, dict):
            raise BackendError("INVALID_SEED", "Each telescope must be an object")
        ident = _required_text(item.get("id"), "telescope.id")
        if ident in telescope_ids:
            raise BackendError("INVALID_SEED", f"Duplicate telescope: {ident}")
        aperture = item.get("aperture_mm")
        if isinstance(aperture, bool) or not isinstance(aperture, int) or aperture <= 0:
            raise BackendError("INVALID_SEED", "aperture_mm must be a positive integer")
        if not isinstance(item.get("active"), bool):
            raise BackendError("INVALID_SEED", "active must be a boolean")
        telescopes.append({
            "id": ident, "name": _required_text(item.get("name"), "telescope.name"),
            "site": _required_text(item.get("site"), "telescope.site"),
            "aperture_mm": aperture, "active": int(item["active"]),
        })
        telescope_ids.add(ident)

    slots: list[dict[str, Any]] = []
    slot_ids: set[str] = set()
    intervals: dict[str, list[tuple[str, str]]] = {}
    for item in seed["slots"]:
        if not isinstance(item, dict):
            raise BackendError("INVALID_SEED", "Each slot must be an object")
        ident = _required_text(item.get("id"), "slot.id")
        telescope_id = _required_text(item.get("telescope_id"), "slot.telescope_id")
        if ident in slot_ids or telescope_id not in telescope_ids:
            raise BackendError("INVALID_SEED", f"Duplicate slot or unknown telescope: {ident}")
        start, end = _utc(item.get("start_utc")), _utc(item.get("end_utc"))
        if start >= end:
            raise BackendError("INVALID_SEED", f"Slot must end after start: {ident}")
        for other_start, other_end in intervals.get(telescope_id, []):
            if start < other_end and other_start < end:
                raise BackendError("INVALID_SEED", f"Overlapping slots for {telescope_id}")
        intervals.setdefault(telescope_id, []).append((start, end))
        status = item.get("status", "open")
        price = item.get("price_cents")
        if status not in ("open", "closed"):
            raise BackendError("INVALID_SEED", f"Invalid slot status: {ident}")
        if isinstance(price, bool) or not isinstance(price, int) or price < 0:
            raise BackendError("INVALID_SEED", "price_cents must be a non-negative integer")
        slots.append({"id": ident, "telescope_id": telescope_id, "start_utc": start,
                      "end_utc": end, "status": status, "price_cents": price})
        slot_ids.add(ident)

    bookings: list[dict[str, Any]] = []
    booking_ids: set[str] = set()
    occupied: set[str] = set()
    for item in seed["bookings"]:
        if not isinstance(item, dict):
            raise BackendError("INVALID_SEED", "Each booking must be an object")
        ident = _required_text(item.get("id"), "booking.id")
        slot_id = _required_text(item.get("slot_id"), "booking.slot_id")
        status = item.get("status")
        paid = item.get("paid")
        if ident in booking_ids or slot_id not in slot_ids:
            raise BackendError("INVALID_SEED", f"Duplicate booking or unknown slot: {ident}")
        if status not in ("confirmed", "cancelled") or not isinstance(paid, bool):
            raise BackendError("INVALID_SEED", f"Invalid booking status/paid: {ident}")
        if status == "confirmed":
            slot = next(s for s in slots if s["id"] == slot_id)
            if slot["status"] != "open" or slot_id in occupied:
                raise BackendError("INVALID_SEED", f"Booking conflicts with slot: {slot_id}")
            occupied.add(slot_id)
        refunded = item.get("refunded", status == "cancelled" and paid)
        if not isinstance(refunded, bool) or (refunded and (not paid or status != "cancelled")):
            raise BackendError("INVALID_SEED", f"Invalid refund state: {ident}")
        bookings.append({"id": ident, "slot_id": slot_id,
                         "user_id": _required_text(item.get("user_id"), "booking.user_id"),
                         "status": status, "paid": int(paid), "refunded": int(refunded)})
        booking_ids.add(ident)
    return telescopes, slots, bookings


class ObservatoryStore:
    """Small stateful backend, designed for deterministic evaluation and MCP wrapping."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS telescopes (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, site TEXT NOT NULL,
                    aperture_mm INTEGER NOT NULL, active INTEGER NOT NULL CHECK(active IN (0,1))
                );
                CREATE TABLE IF NOT EXISTS slots (
                    id TEXT PRIMARY KEY, telescope_id TEXT NOT NULL REFERENCES telescopes(id),
                    start_utc TEXT NOT NULL, end_utc TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('open','closed')),
                    price_cents INTEGER NOT NULL CHECK(price_cents >= 0)
                );
                CREATE TABLE IF NOT EXISTS bookings (
                    id TEXT PRIMARY KEY, slot_id TEXT NOT NULL REFERENCES slots(id),
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('confirmed','cancelled')),
                    paid INTEGER NOT NULL CHECK(paid IN (0,1)),
                    refunded INTEGER NOT NULL DEFAULT 0 CHECK(refunded IN (0,1))
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_confirmed_booking_per_slot
                    ON bookings(slot_id) WHERE status='confirmed';
            """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.db_path), timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def reset_backend(self, seed_path: str | Path) -> dict[str, Any]:
        """Atomically replace state from a validated seed. Never changes real-world services."""
        try:
            with Path(seed_path).open(encoding="utf-8") as stream:
                seed = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendError("INVALID_SEED", f"Cannot read seed: {exc}") from exc
        telescopes, slots, bookings = _check_seed(seed)
        with closing(self._connect()) as db:
            with db:
                db.execute("BEGIN IMMEDIATE")
                # Invalidate all outstanding cancellation grants on every evaluation reset.
                if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                              "AND name='_cancellation_approvals'").fetchone():
                    db.execute("DELETE FROM _cancellation_approvals")
                db.execute("DELETE FROM bookings")
                db.execute("DELETE FROM slots")
                db.execute("DELETE FROM telescopes")
                db.executemany(
                    "INSERT INTO telescopes (id,name,site,aperture_mm,active) "
                    "VALUES (:id,:name,:site,:aperture_mm,:active)", telescopes,
                )
                db.executemany(
                    "INSERT INTO slots (id,telescope_id,start_utc,end_utc,status,price_cents) "
                    "VALUES (:id,:telescope_id,:start_utc,:end_utc,:status,:price_cents)", slots,
                )
                db.executemany(
                    "INSERT INTO bookings (id,slot_id,user_id,status,paid,refunded) "
                    "VALUES (:id,:slot_id,:user_id,:status,:paid,:refunded)", bookings,
                )
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        """Canonical, independently inspectable snapshot used for state-based grading."""
        with closing(self._connect()) as db:
            telescopes = [dict(row) for row in db.execute("SELECT * FROM telescopes ORDER BY id")]
            slots = [dict(row) for row in db.execute("SELECT * FROM slots ORDER BY id")]
            bookings = [dict(row) for row in db.execute("SELECT * FROM bookings ORDER BY id")]
        occupied = {b["slot_id"] for b in bookings if b["status"] == "confirmed"}
        active = {t["id"] for t in telescopes if t["active"]}
        for telescope in telescopes:
            telescope["active"] = bool(telescope["active"])
        for slot in slots:
            slot["available"] = (slot["status"] == "open"
                                 and slot["telescope_id"] in active
                                 and slot["id"] not in occupied)
        for booking in bookings:
            booking["paid"] = bool(booking["paid"])
            booking["refunded"] = bool(booking["refunded"])
        return {"telescopes": telescopes, "slots": slots, "bookings": bookings}

    def list_telescopes(self) -> list[dict[str, Any]]:
        return self.get_state()["telescopes"]

    def list_slots(self, *, available_only: bool = False) -> list[dict[str, Any]]:
        slots = self.get_state()["slots"]
        return [slot for slot in slots if slot["available"]] if available_only else slots

    def create_booking(self, slot_id: str, user_id: str) -> dict[str, Any]:
        """Reserve an available slot (simulated; no money is charged)."""
        slot_id = _required_text(slot_id, "slot_id")
        user_id = _required_text(user_id, "user_id")
        with closing(self._connect()) as db:
            with db:
                db.execute("BEGIN IMMEDIATE")
                slot = db.execute(
                    "SELECT s.status, t.active FROM slots s JOIN telescopes t "
                    "ON s.telescope_id=t.id WHERE s.id=?", (slot_id,),
                ).fetchone()
                if slot is None:
                    raise BackendError("SLOT_NOT_FOUND", "Slot does not exist", slot_id=slot_id)
                occupied = db.execute(
                    "SELECT 1 FROM bookings WHERE slot_id=? AND status='confirmed'", (slot_id,),
                ).fetchone()
                if slot["status"] != "open" or not slot["active"] or occupied:
                    raise BackendError("SLOT_UNAVAILABLE", "Slot is not available", slot_id=slot_id)
                used = [int(m.group(1)) for row in db.execute("SELECT id FROM bookings")
                        if (m := re.fullmatch(r"BK-(\d+)", row["id"]))]
                booking_id = f"BK-{max(used, default=0) + 1:03d}"
                db.execute("INSERT INTO bookings (id,slot_id,user_id,status,paid,refunded) "
                           "VALUES (?,?,?,'confirmed',0,0)", (booking_id, slot_id, user_id))
        return self.get_booking(booking_id)

    def get_booking(self, booking_id: str) -> dict[str, Any]:
        booking_id = _required_text(booking_id, "booking_id")
        for booking in self.get_state()["bookings"]:
            if booking["id"] == booking_id:
                return booking
        raise BackendError("BOOKING_NOT_FOUND", "Booking does not exist", booking_id=booking_id)

    def cancel_booking(self, booking_id: str, user_id: str) -> dict[str, Any]:
        """DANGEROUS: cancel irreversibly and issue a *simulated* refund if paid.

        MCP/LangGraph MUST obtain trusted human approval before invoking this method.
        This backend deliberately cannot accept an LLM-supplied `approved=True` flag.
        """
        booking_id = _required_text(booking_id, "booking_id")
        user_id = _required_text(user_id, "user_id")
        with closing(self._connect()) as db:
            with db:
                db.execute("BEGIN IMMEDIATE")
                booking = db.execute("SELECT * FROM bookings WHERE id=?", (booking_id,)).fetchone()
                if booking is None:
                    raise BackendError("BOOKING_NOT_FOUND", "Booking does not exist", booking_id=booking_id)
                if booking["user_id"] != user_id:
                    raise BackendError("BOOKING_FORBIDDEN", "Booking belongs to another user")
                if booking["status"] != "confirmed":
                    raise BackendError("BOOKING_NOT_ACTIVE", "Booking is already cancelled")
                db.execute("UPDATE bookings SET status='cancelled', refunded=? WHERE id=?",
                           (int(bool(booking["paid"])), booking_id))
        return self.get_booking(booking_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated, resettable AstroPilote observatory")
    parser.add_argument("--db", default="data/backend/observatory.sqlite3")
    parser.add_argument("--seed", help="Reset the backend from this JSON seed before printing state")
    args = parser.parse_args()
    store = ObservatoryStore(args.db)
    state = store.reset_backend(args.seed) if args.seed else store.get_state()
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
