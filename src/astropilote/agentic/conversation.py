"""Fail-closed French/English conversational front controller for AstroPilote.

The *language* classifier is deterministic and conservative, rather than an LLM
claiming to understand arbitrary requests. Explicit requests are delegated to
Phase-3's real LangGraph -> MCP workflow. Ambiguity never triggers a write.
Astro queries are handed to the existing Generative AI LangGraph assistant.

Not an authentication boundary. The operator supplies ``user_id`` outside of
user-controlled text. Never expose this local controller as an unauthenticated
HTTP endpoint, and never turn a conversational 'oui' into HITL approval.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

Route = Literal["availability", "book", "cancel", "booking_info", "clarify", "noop", "astronomy"]
_PARIS = ZoneInfo("Europe/Paris")
_SLOT_PATTERN = re.compile(r"\bS-(\d{1,4})\b", flags=re.IGNORECASE)
_BOOKING_PATTERN = re.compile(r"\bBK-(\d{1,6})\b", flags=re.IGNORECASE)
_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_WEEKDAYS = {
    "lundi": 0, "monday": 0,
    "mardi": 1, "tuesday": 1,
    "mercredi": 2, "wednesday": 2,
    "jeudi": 3, "thursday": 3,
    "vendredi": 4, "friday": 4,
    "samedi": 5, "saturday": 5,
    "dimanche": 6, "sunday": 6,
}
_BOOK_VERBS = re.compile(r"\b(?:reserve|reserver|reservez|reserves|book)\b")
_CANCEL_VERBS = re.compile(r"\b(?:annule|annuler|annulez|annulation|cancel|cancellation|supprime|supprimer|delete)\b")
_READ_BOOKING = re.compile(r"\b(?:statut|status|detail|details|voir|consulter|montre|montrer|show|check|retrouve|retrouver|quel|quelle|what|which)\b")
_AVAILABILITY = re.compile(r"\b(?:disponible|disponibles|disponibilite|disponibilites|available|availability|creneau|creneaux|slots?|telescopes?)\b")
_ASKING = re.compile(r"\b(?:quel|quels|quelle|quelles|liste|lister|affiche|afficher|montre|montrer|voir|show|find|are|is|what|quand|when|disponible|disponibles|available|disponibilite|disponibilites|combien)\b")
_CONFIRMATION_ONLY = {"oui", "yes", "ok", "confirme", "confirmer", "approve", "approved", "approuve", "approuver", "je confirme", "i approve", "i confirm"}
_NEGATED_ACTION = re.compile(r"\b(?:ne\s+|n[' ]|do not\s+|don't\s+|dont\s+|pas\s+|sans\s+)(?:\w+\s+){0,2}(?:annule|annuler|cancel|reserve|reserver|book)\b")
_NEGATED_ACTION_POST = re.compile(r"\b(?:annule|annuler|reserve|reserver|book|cancel|supprime|supprimer)\s+(?:\w+\s+){0,2}(?:pas|jamais)\b")
_NEGATION_NEAR_ACTION = re.compile(
    r"\b(?:ne|n'|pas|sans|never|don't|dont|do not|not)\b[\s\w'.,-]{0,70}"
    r"\b(?:annule|annuler|annulez|annulation|cancel|cancellation|reserve|reserver|book|supprime|supprimer)\b"
)
_QUESTION_ABOUT_ACTION = re.compile(
    r"\b(?:puis-je|peux-tu|peut-on|est-ce possible|can i|could i|should i|is it possible)\b"
)
_APPROVAL_AS_CHAT = re.compile(r"\b(?:approuve|approuver|approve|approved|valide|valider|confirm|confirme|confirmer)\b")


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    return "".join(ch for ch in value if not unicodedata.combining(ch)).replace("’", "'")


def _one_id(pattern: re.Pattern[str], text: str, prefix: str) -> tuple[str | None, bool]:
    ids = {f"{prefix}-{int(part):0{3 if prefix == 'BK' else 2}d}" for part in pattern.findall(text)}
    return (next(iter(ids)) if len(ids) == 1 else None, len(ids) > 1)


@dataclass(frozen=True)
class Decision:
    route: Route
    reason: str
    slot_id: str | None = None
    booking_id: str | None = None
    local_date: str | None = None
    period: Literal["evening", "morning", "afternoon"] | None = None
    place_name: str | None = None
    require_clear_sky: bool = False
    target_object_id: str | None = None


_CITY_PATTERN = re.compile(r"\b(?:lyon|fontainebleau|paris|chamonix|marseille|nice|bordeaux)\b")
_CLEAR_SKY_CUE = re.compile(r"\b(?:weather|meteo|clear|degage|beau|sans nuage|cloud)\b")
_TARGET_CUE = re.compile(r"\b(m0?\d{1,3}|c0?\d{1,3}|omega centauri)\b")


def interpret(text: str, *, reference_date: date | None = None) -> Decision:
    """Classify a bounded set of commands without considering MCP tool output.

    Unknown requests go to the historical astronomy agent; unclear *action*
    requests always require a more specific user instruction.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 4000:
        return Decision("clarify", "Veuillez formuler une demande de 1 à 4 000 caractères.")

    t = _normalized(text)
    slot_id, multi_slots = _one_id(_SLOT_PATTERN, t, "S")
    booking_id, multi_bookings = _one_id(_BOOKING_PATTERN, t, "BK")
    book = bool(_BOOK_VERBS.search(t))
    cancel = bool(_CANCEL_VERBS.search(t))

    # A denial, multiple targets, or contradictory write instructions must not
    # be interpreted as permission to perform a narrower single action.
    if (book or cancel) and (
        _NEGATED_ACTION.search(t) or _NEGATED_ACTION_POST.search(t) or _NEGATION_NEAR_ACTION.search(t)
    ):
        return Decision("noop", "Aucune action effectuée : la demande contient une négation.")
    if t.strip(" .!?,") in _CONFIRMATION_ONLY or (
        booking_id and _APPROVAL_AS_CHAT.search(t) and not _BOOK_VERBS.search(t)
    ):
        return Decision("noop", "Une confirmation dans le chat ne vaut pas une approbation humaine. Utilisez la reprise opérateur HITL.")
    if (book or cancel) and _QUESTION_ABOUT_ACTION.search(t):
        return Decision("clarify", "Cette question ne déclenche aucune action ; formulez explicitement l'opération souhaitée.")
    if multi_slots or multi_bookings or (book and cancel):
        return Decision("clarify", "Précisez une seule opération et un seul identifiant par demande.")
    if (book or cancel) and re.search(
        r"\b(?:pour|for)\s+(?!(?:moi|me|nous|us|le|la|les)\b)[a-z][\w-]*", t
    ):
        return Decision("clarify", "Les actions pour un tiers ne sont pas autorisées par cette interface locale.")
    if cancel:
        if not booking_id:
            parsed_date, issue = _requested_date(t, reference_date=reference_date)
            if not issue and parsed_date:
                return Decision(
                    "cancel",
                    "Annulation par date demandée : vérification du créneau puis validation humaine.",
                    local_date=parsed_date,
                )
            return Decision("clarify", "Indiquez l'identifiant de réservation à annuler (ex. BK-001).")
        return Decision("cancel", "Annulation demandée : validation humaine obligatoire.", booking_id=booking_id)
    if book:
        if not slot_id:
            return Decision("clarify", "Quel créneau souhaitez-vous réserver ? Indiquez un identifiant (ex. S-01).")
        if booking_id:
            return Decision("clarify", "Précisez uniquement un identifiant de créneau S-xx à réserver.")
        parsed_date, issue = _requested_date(t, reference_date=reference_date)
        if issue:
            return Decision("clarify", issue)
        city_match = _CITY_PATTERN.search(t)
        place_name = city_match.group(0).capitalize() if city_match else None
        require_clear = bool(_CLEAR_SKY_CUE.search(t))
        target_match = _TARGET_CUE.search(t)
        raw_target = target_match.group(1) if target_match else None
        norm_target: str | None = None
        if raw_target:
            if raw_target == "omega centauri" or raw_target in ("c80", "c080"):
                norm_target = "C080"
            elif raw_target.startswith("m"):
                norm_target = f"M{int(raw_target[1:]):03d}"
            elif raw_target.startswith("c"):
                norm_target = f"C{int(raw_target[1:]):03d}"
        return Decision(
            "book",
            "Réservation d'un créneau identifié.",
            slot_id=slot_id,
            local_date=parsed_date,
            place_name=place_name,
            require_clear_sky=require_clear,
            target_object_id=norm_target,
        )

    if _READ_BOOKING.search(t) and (booking_id or re.search(r"\b(?:reservation|booking)\b", t)):
        if not booking_id:
            return Decision("clarify", "Indiquez l'identifiant BK-xxx de la réservation à consulter.")
        return Decision("booking_info", "Consultation d'une réservation.", booking_id=booking_id)

    equipment_advice = bool(re.search(r"\b(?:meilleur|meilleure|best|ideal|focale|grossissement|photographier|photography|imaging)\b", t))
    explicit_availability = bool(re.search(r"\b(?:disponible|disponibles|disponibilite|disponibilites|available|availability|creneau|creneaux|slots?)\b", t))
    if _AVAILABILITY.search(t) and _ASKING.search(t) and (explicit_availability or not equipment_advice):
        parsed_date, issue = _requested_date(t, reference_date=reference_date)
        if issue:
            return Decision("clarify", issue)
        period = (
            "evening" if re.search(r"\b(?:soir|soiree|evening|tonight)\b", t)
            else "morning" if re.search(r"\b(?:matin|morning)\b", t)
            else "afternoon" if re.search(r"\b(?:apres-midi|afternoon)\b", t)
            else None
        )
        return Decision("availability", "Consultation des créneaux simulés.", local_date=parsed_date, period=period)

    return Decision("astronomy", "Question transmise au graphe astronomique existant.")


def _requested_date(t: str, *, reference_date: date | None) -> tuple[str | None, str | None]:
    explicit = set(_DATE_PATTERN.findall(t))
    if len(explicit) > 1:
        return None, "Précisez une seule date pour la recherche de créneaux."
    if explicit:
        raw = next(iter(explicit))
        try:
            return date.fromisoformat(raw).isoformat(), None
        except ValueError:
            return None, "Date invalide : utilisez le format AAAA-MM-JJ."

    anchor = reference_date or datetime.now(_PARIS).date()
    if re.search(r"\b(?:demain|tomorrow)\b", t):
        return (anchor + timedelta(days=1)).isoformat(), None
    if re.search(r"\b(?:aujourd'hui|today)\b", t):
        return anchor.isoformat(), None
    days = {day_num for label, day_num in _WEEKDAYS.items() if re.search(rf"\b{label}\b", t)}
    if len(days) > 1:
        return None, "Précisez un seul jour de la semaine."
    if len(days) == 1:
        weekday = next(iter(days))
        delta = (weekday - anchor.weekday()) % 7
        return (anchor + timedelta(days=delta)).isoformat(), None
    return None, None


def _match_period(local_time: time, period: str | None) -> bool:
    hour = local_time.hour
    if period == "morning":
        return hour < 12
    if period == "afternoon":
        return 12 <= hour < 17
    if period == "evening":
        return 17 <= hour < 24
    return True


class ConversationalAgent:
    """Locally trusted entrypoint: natural language -> deterministic route -> real MCP.

    No code here may issue or accept a cancellation capability. Only the trusted
    Phase-3 operator can call ``resume --approve`` after LangGraph's interrupt.
    """

    def __init__(
        self,
        *,
        db_path: str | Path = "data/backend/observatory.sqlite3",
        checkpoint_path: str | Path = "data/backend/agentic_checkpoints.sqlite3",
        gateway: Any | None = None,
        workflow_factory: Callable[[], Any] | None = None,
        astronomy_runner: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.checkpoint_path = Path(checkpoint_path)
        self._gateway = gateway
        self._workflow_factory = workflow_factory
        self._astronomy_runner = astronomy_runner

    @property
    def gateway(self) -> Any:
        if self._gateway is None:
            from astropilote.agentic.workflow import StdioMCPGateway

            self._gateway = StdioMCPGateway(self.db_path)
        return self._gateway

    def _make_workflow(self) -> Any:
        if self._workflow_factory is not None:
            return self._workflow_factory()
        from astropilote.agentic.workflow import ObservatoryWorkflow

        return ObservatoryWorkflow(
            db_path=self.db_path, checkpoint_path=self.checkpoint_path, gateway=self.gateway
        )

    def handle(
        self,
        text: str,
        *,
        user_id: str,
        thread_id: str | None = None,
        reference_date: date | None = None,
        max_steps: int = 5,
    ) -> dict[str, Any]:
        """Route a sentence; ``user_id`` must come from the trusted CLI host.

        Free-form conversation never resumes an approval. It can only *start*
        a phase-3 cancellation workflow and leave it suspended.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("A non-empty trusted user_id is required")
        decision = interpret(text, reference_date=reference_date)
        if decision.route in ("clarify", "noop"):
            return {"route": decision.route, "status": "needs_clarification" if decision.route == "clarify" else "no_action", "message": decision.reason, "tool_trace": []}
        if decision.route == "astronomy":
            return self._astronomy(text)
        if decision.route in ("book", "cancel"):
            return self._handle_action_route(
                decision=decision,
                user_id=user_id.strip(),
                thread_id=thread_id,
                max_steps=max_steps,
            )
        trace: list[dict[str, str]] = []
        current_tool = "get_booking" if decision.route == "booking_info" else "list_telescopes"
        try:
            if decision.route == "booking_info":
                booking = self.gateway.call(
                    "observatory", "get_booking", {"booking_id": decision.booking_id}
                )
                trace.append({"tool": "get_booking", "status": "ok"})
                if booking.get("user_id") != user_id.strip():
                    return {
                        "route": "booking_info", "status": "forbidden",
                        "message": "Cette réservation n'appartient pas à l'utilisateur courant.",
                        "tool_trace": trace,
                    }
                return {
                    "route": "booking_info", "status": "completed",
                    "message": "Réservation trouvée (simulation).",
                    "data": booking, "tool_trace": trace,
                }

            telescopes = self.gateway.call("observatory", "list_telescopes", {})
            trace.append({"tool": "list_telescopes", "status": "ok"})
            current_tool = "list_slots"
            slots = self.gateway.call("observatory", "list_slots", {"available_only": True})
            trace.append({"tool": "list_slots", "status": "ok"})
            names = {item["id"]: item["name"] for item in telescopes if item.get("active")}
            matches = []
            for slot in slots:
                if slot.get("telescope_id") not in names:
                    continue
                start = datetime.fromisoformat(
                    slot["start_utc"].replace("Z", "+00:00")
                ).astimezone(_PARIS)
                if decision.local_date and start.date().isoformat() != decision.local_date:
                    continue
                if not _match_period(start.timetz(), decision.period):
                    continue
                matches.append({
                    **slot,
                    "telescope_name": names[slot["telescope_id"]],
                    "start_local": start.isoformat(),
                })
            return {
                "route": "availability",
                "status": "completed" if matches else "unavailable",
                "message": f"{len(matches)} créneau(x) disponible(s) dans l'observatoire simulé.",
                "filter_date": decision.local_date,
                "filter_period": decision.period,
                "data": matches,
                "tool_trace": trace,
            }
        except Exception as exc:
            # Fail closed; do not reveal credentials or exception messages.
            trace.append({"tool": current_tool, "status": "error"})
            payload = getattr(exc, "payload", None)
            error = payload.get("error") if isinstance(payload, dict) else None
            code = error.get("code") if isinstance(error, dict) else None
            if code == "BOOKING_NOT_FOUND":
                return {
                    "route": decision.route, "status": "not_found",
                    "message": "Réservation introuvable.",
                    "error": {"code": code}, "tool_trace": trace,
                }
            return {
                "route": decision.route, "status": "tool_error",
                "message": "Le serveur MCP de l'observatoire n'est pas disponible.",
                "error": {"code": "MCP_UNAVAILABLE"}, "tool_trace": trace,
            }

    def _handle_action_route(
        self,
        *,
        decision: Decision,
        user_id: str,
        thread_id: str | None,
        max_steps: int,
    ) -> dict[str, Any]:
        pre_trace: list[dict[str, Any]] = []
        resolved_booking_id = decision.booking_id

        if decision.route == "cancel" and resolved_booking_id is None and decision.local_date:
            state_snap = self.gateway.call("observatory", "get_observatory_state", {})
            pre_trace.append({"tool": "get_observatory_state", "status": "ok", "args": {}})
            slots_by_id = {s["id"]: s for s in state_snap.get("slots", [])}
            matching = []
            for bk in state_snap.get("bookings", []):
                if bk.get("user_id") != user_id or bk.get("status") != "confirmed":
                    continue
                slot = slots_by_id.get(bk.get("slot_id"))
                if not slot:
                    continue
                local_d = (
                    datetime.fromisoformat(slot["start_utc"].replace("Z", "+00:00"))
                    .astimezone(_PARIS)
                    .date()
                    .isoformat()
                )
                if local_d == decision.local_date:
                    matching.append(bk["id"])
            if len(matching) != 1:
                return {
                    "route": "clarify",
                    "status": "needs_clarification",
                    "message": "Plusieurs ou aucune réservation active à cette date ; précisez BK-xxx.",
                    "steps": 1,
                    "tool_trace": pre_trace,
                }
            resolved_booking_id = matching[0]

        if decision.route == "book" and decision.place_name and (
            decision.require_clear_sky or decision.target_object_id
        ):
            obs_date = decision.local_date or "2026-11-17"
            geo = self.gateway.call("sky", "geocode", {"place_name": decision.place_name})
            pre_trace.append({"tool": "geocode", "status": "ok", "args": {"place_name": decision.place_name}})
            if not geo.get("found"):
                return {
                    "route": "action",
                    "intent": "book",
                    "status": "impossible",
                    "message": "Site d'observation introuvable.",
                    "steps": 1,
                    "tool_trace": pre_trace,
                }
            lat, lon = float(geo["lat"]), float(geo["lon"])
            if decision.require_clear_sky:
                wx = self.gateway.call(
                    "sky",
                    "weather",
                    {"lat": lat, "lon": lon, "obs_date": obs_date, "place_name": decision.place_name},
                )
                pre_trace.append(
                    {"tool": "weather", "status": "ok", "args": {"lat": lat, "lon": lon, "obs_date": obs_date}}
                )
                if wx.get("available") is False or float(wx.get("cloud_cover_pct", 100)) > 40.0:
                    return {
                        "route": "action",
                        "intent": "book",
                        "status": "impossible",
                        "message": "Ciel couvert ou hors horizon météo : réservation refusée.",
                        "steps": 2,
                        "tool_trace": pre_trace,
                    }
            if decision.target_object_id:
                vis = self.gateway.call(
                    "sky",
                    "visibility",
                    {"object_ids": [decision.target_object_id], "lat": lat, "lon": lon, "obs_date": obs_date},
                )
                pre_trace.append(
                    {
                        "tool": "visibility",
                        "status": "ok",
                        "args": {"object_ids": [decision.target_object_id], "lat": lat, "lon": lon, "obs_date": obs_date},
                    }
                )
                objs = vis.get("objects") if isinstance(vis, dict) else None
                first_obj = objs[0] if isinstance(objs, list) and objs else {}
                if not first_obj.get("observable") and float(first_obj.get("max_altitude_deg", -90.0)) < 20.0:
                    return {
                        "route": "action",
                        "intent": "book",
                        "status": "impossible",
                        "message": "Cible inobservable sous 20° d'altitude depuis ce site.",
                        "steps": 2,
                        "tool_trace": pre_trace,
                    }

        resolved_thread = thread_id or f"conv-{uuid4().hex}"
        remaining_budget = max(1, max_steps - (1 if pre_trace else 0))
        with self._make_workflow() as workflow:
            result = workflow.start(
                thread_id=resolved_thread,
                action=decision.route,
                user_id=user_id,
                slot_id=decision.slot_id,
                booking_id=resolved_booking_id,
                max_steps=remaining_budget,
            )
        merged_trace = pre_trace + list(result.get("tool_trace", []))
        merged_steps = int(result.get("steps", 0)) + (1 if pre_trace else 0)
        return {
            "route": "action",
            "intent": decision.route,
            "message": decision.reason,
            **result,
            "steps": merged_steps,
            "tool_trace": merged_trace,
        }

    def _astronomy(self, text: str) -> dict[str, Any]:
        if self._astronomy_runner is not None:
            response = self._astronomy_runner(text)
            return {**response, "route": "astronomy"}
        from astropilote.agent.graph import run_agent

        answer, _, record = run_agent(text)
        return {
            "route": "astronomy", "status": "abstained" if answer.abstained else "answered",
            "message": answer.answer, "sources": answer.sources,
            "tool_trace": record.tool_calls, "tokens_in": record.tokens_in,
            "tokens_out": record.tokens_out, "cost_usd": record.cost_usd,
        }
