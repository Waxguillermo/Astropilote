"""Opt-in user-memory integration for the Phase-4A conversational MCP router.

A preference may *rank* available slots but it never books or cancels them.
Approval remains exclusively on the Phase-3 trusted HITL controller path.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from astropilote.agentic.conversation import ConversationalAgent, _normalized
from astropilote.agentic.memory import UserMemory

# Strict full-sentence matching: unexpected suffixes, multiple actions and
# prompt injection strings cannot accidentally become memory commands.
_SAVE = re.compile(
    r"(?:mon telescope prefere est|je prefere le telescope|"
    r"my preferred telescope is|i prefer telescope)\s+(T-\d{2,4})\s*[.!]?\Z",
    flags=re.IGNORECASE,
)
_FORGET = frozenset({
    "oublie mon telescope prefere",
    "supprime ma preference de telescope",
    "forget my preferred telescope",
    "forget my telescope preference",
})
_RECALL = frozenset({
    "quel est mon telescope prefere",
    "quel telescope ai-je prefere",
    "what is my preferred telescope",
    "which telescope do i prefer",
})
_EXPLICIT_TELESCOPE = re.compile(r"\bT-\d{2,4}\b", flags=re.IGNORECASE)
_MEMORY_CUE = re.compile(
    r"^(?:mon telescope prefere|je prefere le telescope|my preferred telescope|"
    r"i prefer telescope|oublie mon telescope|supprime ma preference de telescope|"
    r"forget my telescope|forget my preferred telescope|"
    r"quel est mon telescope prefere|what is my preferred telescope|"
    r"which telescope do i prefer)\b"
)


class MemoryConversationalAgent(ConversationalAgent):
    """Phase-4A router + independent durable preference store."""

    def __init__(
        self,
        *,
        db_path: str | Path = "data/backend/observatory.sqlite3",
        checkpoint_path: str | Path = "data/backend/agentic_checkpoints.sqlite3",
        memory_path: str | Path = "data/backend/user_memory.sqlite3",
        memory: UserMemory | None = None,
        gateway: Any | None = None,
        workflow_factory: Any | None = None,
        astronomy_runner: Any | None = None,
    ) -> None:
        super().__init__(
            db_path=db_path, checkpoint_path=checkpoint_path,
            gateway=gateway, workflow_factory=workflow_factory,
            astronomy_runner=astronomy_runner,
        )
        self.memory = memory if memory is not None else UserMemory(memory_path)

    def handle(
        self,
        text: str,
        *,
        user_id: str,
        thread_id: str | None = None,
        reference_date: date | None = None,
        max_steps: int = 5,
    ) -> dict[str, Any]:
        # Validate the trusted actor before any preference read/write.
        self.memory._actor(user_id)
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            return super().handle(
                text, user_id=user_id, thread_id=thread_id,
                reference_date=reference_date, max_steps=max_steps,
            )
        normalized = _normalized(text).strip().rstrip(".!? ")
        save = _SAVE.fullmatch(normalized)
        if save is not None:
            telescope = save.group(1).upper()
            try:
                available_telescopes = self.gateway.call("observatory", "list_telescopes", {})
            except Exception:
                return {
                    "route": "memory", "status": "tool_error",
                    "message": "Impossible de vérifier le télescope auprès du serveur MCP.",
                    "tool_trace": [{"tool": "list_telescopes", "status": "error"}],
                }
            valid = any(
                entry.get("id") == telescope and entry.get("active") is True
                for entry in available_telescopes
                if isinstance(entry, dict)
            )
            if not valid:
                return {
                    "route": "memory", "status": "invalid_telescope",
                    "message": "Le télescope indiqué n'est pas actif dans l'observatoire simulé.",
                    "tool_trace": [{"tool": "list_telescopes", "status": "ok"}],
                }
            self.memory.set_preferred_telescope(user_id, telescope)
            return {
                "route": "memory", "status": "saved",
                "message": f"Préférence enregistrée pour cet utilisateur : {telescope}.",
                "preferred_telescope_id": telescope,
                "tool_trace": [{"tool": "list_telescopes", "status": "ok"}],
            }

        if normalized in _FORGET:
            existed = self.memory.clear_preferred_telescope(user_id)
            return {
                "route": "memory", "status": "forgotten",
                "message": "Préférence de télescope effacée pour cet utilisateur.",
                "had_preference": existed, "tool_trace": [],
            }
        if normalized in _RECALL:
            selected = self.memory.preferred_telescope(user_id)
            return {
                "route": "memory", "status": "found" if selected else "not_set",
                "message": f"Votre télescope préféré est {selected}." if selected else "Aucun télescope préféré mémorisé.",
                "preferred_telescope_id": selected, "tool_trace": [],
            }

        # A malformed preference statement is not an action command: fail
        # closed instead of sending its injected suffix into the booking router.
        if _MEMORY_CUE.match(normalized):
            return {
                "route": "clarify", "status": "needs_clarification",
                "message": "Formulez une seule commande de mémoire avec un identifiant T-xx valide.",
                "tool_trace": [],
            }

        # Never modify the Phase-3 workflow, even if a memory exists. Explicit
        # booking IDs, cancellation IDs and HITL approval remain authoritative.
        result = super().handle(
            text, user_id=user_id, thread_id=thread_id,
            reference_date=reference_date, max_steps=max_steps,
        )
        if result.get("route") != "availability" or result.get("status") not in ("completed", "unavailable"):
            return result
        selected = self.memory.preferred_telescope(user_id)
        if selected is None or _EXPLICIT_TELESCOPE.search(text):
            return result
        slots = result.get("data")
        if not isinstance(slots, list):
            return result
        preferred = [item for item in slots if item.get("telescope_id") == selected]
        if not preferred:
            return {
                **result, "memory_applied": False,
                "preferred_telescope_id": selected,
                "memory_note": "Aucun créneau disponible pour votre télescope préféré ; les autres restent proposés.",
            }
        others = [item for item in slots if item.get("telescope_id") != selected]
        best = preferred[0]
        return {
            **result, "data": preferred + others,
            "memory_applied": True,
            "preferred_telescope_id": selected,
            "recommended_slot_id": best["id"],
            "memory_note": "Le créneau recommandé correspond au télescope mémorisé ; aucune réservation n'a été créée.",
        }
