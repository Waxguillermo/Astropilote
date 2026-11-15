"""Deterministic SQLite backend tests: persistence, reset and dangerous writes."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from astropilote.backend.store import BackendError, ObservatoryStore

SEED = Path(__file__).resolve().parents[1] / "seeds" / "observatory_demo.json"


@pytest.fixture
def store(tmp_path: Path) -> ObservatoryStore:
    result = ObservatoryStore(tmp_path / "observatory.sqlite3")
    result.reset_backend(SEED)
    return result


def test_reset_is_repeatable_and_state_is_stable(store: ObservatoryStore) -> None:
    initial = store.get_state()
    assert len(initial["telescopes"]) == 2
    assert len(initial["slots"]) == 4
    assert len(initial["bookings"]) == 1
    store.create_booking("S-01", "observer")
    assert store.get_state() != initial
    assert store.reset_backend(SEED) == initial
    assert store.reset_backend(SEED) == initial


def test_lists_and_availability(store: ObservatoryStore) -> None:
    assert len(store.list_telescopes()) == 2
    assert [s["id"] for s in store.list_slots(available_only=True)] == ["S-01", "S-03"]
    assert store.get_booking("BK-001")["paid"] is True


def test_create_booking_persists_across_instances(store: ObservatoryStore) -> None:
    booking = store.create_booking("S-01", "observer-1")
    assert booking == {"id": "BK-002", "slot_id": "S-01", "user_id": "observer-1",
                       "status": "confirmed", "paid": False, "refunded": False}
    reopened = ObservatoryStore(store.db_path)
    assert reopened.get_booking("BK-002") == booking
    assert "S-01" not in [s["id"] for s in reopened.list_slots(available_only=True)]


def test_double_booking_cannot_change_state(store: ObservatoryStore) -> None:
    store.create_booking("S-01", "first")
    before = store.get_state()
    with pytest.raises(BackendError, match="not available") as exc:
        store.create_booking("S-01", "second")
    assert exc.value.code == "SLOT_UNAVAILABLE"
    assert store.get_state() == before


def test_closed_missing_and_invalid_slots_are_rejected(store: ObservatoryStore) -> None:
    before = store.get_state()
    for slot_id, code in [("S-04", "SLOT_UNAVAILABLE"), ("S-99", "SLOT_NOT_FOUND")]:
        with pytest.raises(BackendError) as exc:
            store.create_booking(slot_id, "observer")
        assert exc.value.code == code
    with pytest.raises(BackendError) as exc:
        store.create_booking("", "observer")
    assert exc.value.code == "INVALID_ARGUMENT"
    assert store.get_state() == before


def test_paid_cancellation_simulates_refund_and_is_irreversible(store: ObservatoryStore) -> None:
    cancelled = store.cancel_booking("BK-001", "astro-club")
    assert cancelled["status"] == "cancelled"
    assert cancelled["paid"] is True
    assert cancelled["refunded"] is True  # simulated, not an external payment
    assert "S-02" in [s["id"] for s in store.list_slots(available_only=True)]
    before = store.get_state()
    with pytest.raises(BackendError) as exc:
        store.cancel_booking("BK-001", "astro-club")
    assert exc.value.code == "BOOKING_NOT_ACTIVE"
    assert store.get_state() == before


def test_unauthorized_cancellation_does_not_write(store: ObservatoryStore) -> None:
    before = store.get_state()
    with pytest.raises(BackendError) as exc:
        store.cancel_booking("BK-001", "other-user")
    assert exc.value.code == "BOOKING_FORBIDDEN"
    assert store.get_state() == before


def test_invalid_seed_does_not_erase_existing_state(store: ObservatoryStore, tmp_path: Path) -> None:
    raw = json.loads(SEED.read_text(encoding="utf-8"))
    raw["slots"].append(dict(raw["slots"][0]))
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")
    before = store.get_state()
    with pytest.raises(BackendError) as exc:
        store.reset_backend(invalid)
    assert exc.value.code == "INVALID_SEED"
    assert store.get_state() == before


def test_concurrent_booking_allows_only_one_winner(store: ObservatoryStore) -> None:
    def attempt(user: str) -> str:
        try:
            store.create_booking("S-01", user)
            return "ok"
        except BackendError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(attempt, ["a", "b"]))
    assert sorted(outcomes) == ["SLOT_UNAVAILABLE", "ok"]
    assert len([b for b in store.get_state()["bookings"] if b["slot_id"] == "S-01"]) == 1


def test_error_is_structured(store: ObservatoryStore) -> None:
    with pytest.raises(BackendError) as exc:
        store.get_booking("missing")
    assert exc.value.as_dict()["error"]["code"] == "BOOKING_NOT_FOUND"
    assert exc.value.as_dict()["ok"] is False
