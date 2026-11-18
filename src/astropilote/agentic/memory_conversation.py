"""Opt-in user-memory integration for the deterministic conversational MCP router.

A preference may *rank* available slots but it never books or cancels them.
Approval remains exclusively on the trusted HITL controller path.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from astropilote.agentic.conversation import ConversationalAgent, _normalized
from astropilote.agentic.memory import UserMemory, site_key

# Strict full-sentence matching: unexpected suffixes, multiple actions and
# prompt injection strings cannot accidentally become memory commands.
_SAVE = re.compile(
    r"(?:mon telescope prefere est|je prefere le telescope|"
    r"my preferred telescope is|i prefer telescope)\s+(T-\d{2,4})\s*[.!]?\Z",
    flags=re.IGNORECASE,
)
_FORGET = frozenset(
    {
        "oublie mon telescope prefere",
        "supprime ma preference de telescope",
        "forget my preferred telescope",
        "forget my telescope preference",
    }
)
_RECALL = frozenset(
    {
        "quel est mon telescope prefere",
        "quel telescope ai-je prefere",
        "what is my preferred telescope",
        "which telescope do i prefer",
    }
)
_SAVE_SITE = re.compile(
    r"(?:mon site prefere est|mon site d'observation prefere est|je prefere le site de|"
    r"je prefere observer a|my preferred site is|i prefer the site|i prefer)\s+"
    r"(lyon|fontainebleau)\Z"
)
_FORGET_SITE = frozenset(
    {
        "oublie mon site prefere",
        "supprime ma preference de site",
        "forget my preferred site",
        "forget my site preference",
    }
)
_RECALL_SITE = frozenset(
    {
        "quel est mon site prefere",
        "what is my preferred site",
        "which site do i prefer",
    }
)
_RECALL_HISTORY = frozenset(
    {
        "qu'ai-je reserve la derniere fois",
        "quelles sont mes dernieres reservations",
        "montre mon historique de reservations",
        "mon historique de reservations",
        "what did i book last time",
        "show my booking history",
        "my booking history",
    }
)
_FORGET_HISTORY = frozenset(
    {
        "oublie mon historique",
        "oublie mon historique de reservations",
        "efface mon historique de reservations",
        "forget my booking history",
        "clear my booking history",
    }
)
_EXPLICIT_TELESCOPE = re.compile(r"\bT-\d{2,4}\b", flags=re.IGNORECASE)
_MEMORY_CUE = re.compile(
    r"^(?:mon telescope prefere|je prefere le telescope|my preferred telescope|"
    r"i prefer telescope|oublie mon telescope|supprime ma preference de telescope|"
    r"forget my telescope|forget my preferred telescope|"
    r"quel est mon telescope prefere|what is my preferred telescope|"
    r"which telescope do i prefer|"
    r"mon site prefere|mon site d'observation prefere|je prefere le site|my preferred site|"
    r"oublie mon site|supprime ma preference de site|forget my preferred site|"
    r"forget my site|quel est mon site prefere|what is my preferred site|"
    r"mon historique|oublie mon historique|efface mon historique|my booking history|"
    r"show my booking history|forget my booking history|clear my booking history)\b"
)


class MemoryConversationalAgent(ConversationalAgent):
    """Deterministic conversational router paired with durable per-user preferences."""

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
            db_path=db_path,
            checkpoint_path=checkpoint_path,
            gateway=gateway,
            workflow_factory=workflow_factory,
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
                text,
                user_id=user_id,
                thread_id=thread_id,
                reference_date=reference_date,
                max_steps=max_steps,
            )
        normalized = _normalized(text).strip().rstrip(".!? ")
        save = _SAVE.fullmatch(normalized)
        if save is not None:
            telescope = save.group(1).upper()
            try:
                available_telescopes = self.gateway.call("observatory", "list_telescopes", {})
            except Exception:
                return {
                    "route": "memory",
                    "status": "tool_error",
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
                    "route": "memory",
                    "status": "invalid_telescope",
                    "message": "Le télescope indiqué n'est pas actif dans l'observatoire simulé.",
                    "tool_trace": [{"tool": "list_telescopes", "status": "ok"}],
                }
            self.memory.set_preferred_telescope(user_id, telescope)
            return {
                "route": "memory",
                "status": "saved",
                "message": f"Préférence enregistrée pour cet utilisateur : {telescope}.",
                "preferred_telescope_id": telescope,
                "tool_trace": [{"tool": "list_telescopes", "status": "ok"}],
            }

        if normalized in _FORGET:
            existed = self.memory.clear_preferred_telescope(user_id)
            return {
                "route": "memory",
                "status": "forgotten",
                "message": "Préférence de télescope effacée pour cet utilisateur.",
                "had_preference": existed,
                "tool_trace": [],
            }
        if normalized in _RECALL:
            selected = self.memory.preferred_telescope(user_id)
            return {
                "route": "memory",
                "status": "found" if selected else "not_set",
                "message": f"Votre télescope préféré est {selected}."
                if selected
                else "Aucun télescope préféré mémorisé.",
                "preferred_telescope_id": selected,
                "tool_trace": [],
            }

        site_or_history = self._handle_site_or_history(normalized, user_id)
        if site_or_history is not None:
            return site_or_history

        # A malformed preference statement is not an action command: fail
        # closed instead of sending its injected suffix into the booking router.
        if _MEMORY_CUE.match(normalized):
            return {
                "route": "clarify",
                "status": "needs_clarification",
                "message": "Formulez une seule commande de mémoire (télescope T-xx, site "
                "Lyon ou Fontainebleau, ou historique).",
                "tool_trace": [],
            }

        # Never bypass the checkpointed workflow, even if a memory exists. Explicit
        # booking IDs, cancellation IDs and HITL approval remain authoritative.
        result = super().handle(
            text,
            user_id=user_id,
            thread_id=thread_id,
            reference_date=reference_date,
            max_steps=max_steps,
        )
        if result.get("route") != "availability" or result.get("status") not in (
            "completed",
            "unavailable",
        ):
            return result
        telescope = self.memory.preferred_telescope(user_id)
        site = self.memory.get_preference(user_id, "preferred_site")
        if (telescope is None and site is None) or _EXPLICIT_TELESCOPE.search(text):
            return result
        slots = result.get("data")
        if not isinstance(slots, list):
            return result

        def matches(slot: dict[str, Any]) -> tuple[bool, bool]:
            return (
                telescope is not None and slot.get("telescope_id") == telescope,
                site is not None and site_key(slot.get("telescope_site") or "") == site,
            )

        # Preferred telescope first, then the preferred site; stable within each group.
        ranked = sorted(slots, key=lambda slot: tuple(not hit for hit in matches(slot)))
        remembered: dict[str, Any] = {"preferred_telescope_id": telescope}
        if site is not None:
            remembered["preferred_site"] = site
        best = matches(ranked[0]) if ranked else (False, False)
        if not any(best):
            return {
                **result,
                **remembered,
                "memory_applied": False,
                "memory_note": "Aucun créneau disponible pour votre télescope préféré ; les autres restent proposés."
                if telescope is not None
                else "Aucun créneau disponible sur votre site préféré ; les autres restent proposés.",
            }
        return {
            **result,
            **remembered,
            "data": ranked,
            "memory_applied": True,
            "recommended_slot_id": ranked[0]["id"],
            "memory_note": "Le créneau recommandé correspond au télescope mémorisé ; aucune réservation n'a été créée."
            if best[0]
            else "Le créneau recommandé est sur votre site préféré ; aucune réservation n'a été créée.",
        }

    def _handle_site_or_history(self, normalized: str, user_id: str) -> dict[str, Any] | None:
        """Exact-sentence site and history commands; None when the text is not one of them."""
        save = _SAVE_SITE.fullmatch(normalized)
        if save is not None:
            site = save.group(1)
            try:
                telescopes = self.gateway.call("observatory", "list_telescopes", {})
            except Exception:
                return {
                    "route": "memory",
                    "status": "tool_error",
                    "message": "Impossible de vérifier le site auprès du serveur MCP.",
                    "tool_trace": [{"tool": "list_telescopes", "status": "error"}],
                }
            hosted = any(
                isinstance(entry, dict)
                and entry.get("active") is True
                and site_key(str(entry.get("site", ""))) == site
                for entry in telescopes
            )
            trace = [{"tool": "list_telescopes", "status": "ok"}]
            if not hosted:
                return {
                    "route": "memory",
                    "status": "invalid_site",
                    "message": "Aucun télescope actif sur ce site dans l'observatoire simulé.",
                    "tool_trace": trace,
                }
            stored = self.memory.set_preference(user_id, "preferred_site", site)
            return {
                "route": "memory",
                "status": "saved",
                "message": f"Site préféré enregistré pour cet utilisateur : {stored}.",
                "preferred_site": stored,
                "tool_trace": trace,
            }
        if normalized in _FORGET_SITE:
            existed = self.memory.clear_preference(user_id, "preferred_site")
            return {
                "route": "memory",
                "status": "forgotten",
                "message": "Site préféré effacé pour cet utilisateur.",
                "had_preference": existed,
                "tool_trace": [],
            }
        if normalized in _RECALL_SITE:
            site = self.memory.get_preference(user_id, "preferred_site")
            return {
                "route": "memory",
                "status": "found" if site else "not_set",
                "message": f"Votre site préféré est {site}."
                if site
                else "Aucun site préféré mémorisé.",
                "preferred_site": site,
                "tool_trace": [],
            }
        if normalized in _RECALL_HISTORY:
            history = [event.model_dump() for event in self.memory.recent_events(user_id)]
            last = history[0] if history else None
            return {
                "route": "memory",
                "status": "found" if last else "not_set",
                "message": (
                    f"Dernière opération : {last['kind']} {last['booking_id']} "
                    f"(créneau {last['slot_id']}, télescope {last['telescope_id']}, "
                    f"site {last['site']})."
                )
                if last
                else "Aucune réservation mémorisée.",
                "history": history,
                "tool_trace": [],
            }
        if normalized in _FORGET_HISTORY:
            removed = self.memory.clear_history(user_id)
            return {
                "route": "memory",
                "status": "forgotten",
                "message": "Historique de réservations effacé pour cet utilisateur.",
                "events_removed": removed,
                "tool_trace": [],
            }
        return None
