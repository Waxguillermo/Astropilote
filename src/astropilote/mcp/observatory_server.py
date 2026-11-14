"""Stateful observatory MCP server: readings and safe writes on SQLite.

Run: uv run python -m astropilote.mcp.observatory_server
Set ASTROPILOTE_BACKEND_DB to choose the file. Initialise separately with
`python -m astropilote.backend.store --seed seeds/observatory_demo.json`.

IMPORTANT: cancel_booking remains FAIL-CLOSED unless an expiring, one-time,
booking-bound approval capability issued by the trusted graph controller is
presented. Do not expose the token or token issuer to the model.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from mcp.server import MCPServer
from mcp.types import CallToolResult
from pydantic import Field

from astropilote.backend.approvals import CancellationApprovals
from astropilote.backend.store import ObservatoryStore
from astropilote.mcp.common import approval_required, invoke

DbPath = str | Path
Id = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^\S.*\S$|^\S$")]


def create_server(db_path: DbPath | None = None) -> MCPServer:
    """Make an MCP server tied to exactly one backend database for test isolation."""
    resolved_path = Path(
        db_path if db_path is not None else os.environ.get(
            "ASTROPILOTE_BACKEND_DB", "data/backend/observatory.sqlite3"
        )
    ).expanduser()
    store = ObservatoryStore(resolved_path)
    approvals = CancellationApprovals(resolved_path)
    server = MCPServer("astropilote-observatory")

    @server.tool()
    def get_observatory_state() -> CallToolResult:
        """Read canonical backend state; same source used by state-based evaluation."""
        return invoke(store.get_state)

    @server.tool()
    def list_telescopes() -> CallToolResult:
        """Read all simulated telescopes; no side effects."""
        return invoke(store.list_telescopes)

    @server.tool()
    def list_slots(available_only: bool = False) -> CallToolResult:
        """Read simulated time slots, optionally only those available."""
        return invoke(lambda: store.list_slots(available_only=available_only))

    @server.tool()
    def get_booking(booking_id: Id) -> CallToolResult:
        """Read a simulated reservation by ID; no side effects."""
        return invoke(lambda: store.get_booking(booking_id))

    @server.tool()
    def create_booking(slot_id: Id, user_id: Id) -> CallToolResult:
        """Reserve an available telescope slot in SQLite (simulation; no real payment)."""
        return invoke(lambda: store.create_booking(slot_id=slot_id, user_id=user_id))

    @server.tool()
    def cancel_booking(
        booking_id: Id, user_id: Id, approval_token: str | None = None
    ) -> CallToolResult:
        """DANGEROUS: require a server-verified, expiring, one-time capability.

        The trusted operator/graph controller creates the capability after
        LangGraph's interrupt; MCP callers cannot create capabilities.
        """
        if not approvals.consume(
            token=approval_token, booking_id=booking_id, user_id=user_id
        ):
            return approval_required()
        return invoke(lambda: store.cancel_booking(booking_id, user_id))

    return server


mcp = create_server()


if __name__ == "__main__":
    mcp.run(transport="stdio")
