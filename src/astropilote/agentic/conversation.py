"""Fail-closed French/English conversational front controller for AstroPilote.

The *language* classifier is deterministic and conservative, rather than an LLM
claiming to understand arbitrary requests. Explicit requests are delegated to
the checkpointed LangGraph -> MCP workflow. Ambiguity never triggers a write.
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
from typing import Any, Literal, NamedTuple
from uuid import uuid4
from zoneinfo import ZoneInfo

Route = Literal["availability", "book", "cancel", "booking_info", "clarify", "noop", "astronomy"]
_PARIS = ZoneInfo("Europe/Paris")
_SLOT_PATTERN = re.compile(r"\bS-(\d{1,4})\b", flags=re.IGNORECASE)
_BOOKING_PATTERN = re.compile(r"\bBK-(\d{1,6})\b", flags=re.IGNORECASE)
_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_WEEKDAYS = {
    "lundi": 0,
    "monday": 0,
    "mardi": 1,
    "tuesday": 1,
    "mercredi": 2,
    "wednesday": 2,
    "jeudi": 3,
    "thursday": 3,
    "vendredi": 4,
    "friday": 4,
    "samedi": 5,
    "saturday": 5,
    "dimanche": 6,
    "sunday": 6,
}
_BOOK_VERBS = re.compile(r"\b(?:reserve|reserver|reservez|reserves|book)\b")
_CANCEL_VERBS = re.compile(
    r"\b(?:annule|annuler|annulez|annulation|cancel|cancellation|supprime|supprimer|delete)\b"
)
_READ_BOOKING = re.compile(
    r"\b(?:statut|status|detail|details|voir|consulter|montre|montrer|show|check|retrouve|retrouver|quel|quelle|what|which)\b"
)
_AVAILABILITY = re.compile(
    r"\b(?:disponible|disponibles|disponibilite|disponibilites|available|availability|creneau|creneaux|slots?|telescopes?|coupole|coupoles|domes?)\b"
)
_ASKING = re.compile(
    r"\b(?:quel|quels|quelle|quelles|liste|lister|affiche|afficher|montre|montrer|voir|show|find|are|is|what|quand|when|disponible|disponibles|available|disponibilite|disponibilites|combien)\b"
)
_CONFIRMATION_ONLY = {
    "oui",
    "yes",
    "ok",
    "confirme",
    "confirmer",
    "approve",
    "approved",
    "approuve",
    "approuver",
    "je confirme",
    "i approve",
    "i confirm",
}
_NEGATED_ACTION = re.compile(
    r"\b(?:ne\s+|n[' ]|do not\s+|don't\s+|dont\s+|pas\s+|sans\s+)(?:\w+\s+){0,2}(?:annule|annuler|cancel|reserve|reserver|book)\b"
)
_NEGATED_ACTION_POST = re.compile(
    r"\b(?:annule|annuler|reserve|reserver|book|cancel|supprime|supprimer)\s+(?:\w+\s+){0,2}(?:pas|jamais)\b"
)
_NEGATION_NEAR_ACTION = re.compile(
    r"\b(?:ne|n'|pas|sans|never|don't|dont|do not|not)\b[\s\w'.,-]{0,70}"
    r"\b(?:annule|annuler|annulez|annulation|cancel|cancellation|reserve|reserver|book|supprime|supprimer)\b"
)
_QUESTION_ABOUT_ACTION = re.compile(
    r"\b(?:puis-je|peux-tu|peut-on|est-ce possible|can i|could i|should i|is it possible)\b"
)
_APPROVAL_AS_CHAT = re.compile(
    r"\b(?:approuve|approuver|approve|approved|valide|valider|confirm|confirme|confirmer)\b"
)
_TELESCOPE_PATTERN = re.compile(r"\bT-(\d{1,4})\b", flags=re.IGNORECASE)
_ID_KINDS = {
    "slot": (_SLOT_PATTERN, "S", 2),
    "booking": (_BOOKING_PATTERN, "BK", 3),
    "telescope": (_TELESCOPE_PATTERN, "T", 2),
}
_PREFERENCE_CUE = re.compile(
    r"\b(?:prefere|preferes|prefer|preferred|preference|favori|favorite)\b"
)
_FORGET_CUE = re.compile(
    r"\b(?:oublie|oublier|efface|effacer|supprime|supprimer|forget|remove|delete|reset|clear)\b"
)
_CANCELLATION_WORDS = re.compile(r"\b(?:annule|annuler|annulez|annulation|cancel|cancellation)\b")
_NEGATION = re.compile(
    r"\b(?:ne|n'|pas|jamais|sans(?! nuage)|aucune?|not|no|never|without|dont)\b|n't\b"
)
# A question mark, an interrogative opener, or a French subject inversion ("est-il", "ai-je").
_QUESTION = re.compile(
    r"\?|^\W*(?:is|are|was|were|do|does|did|can|could|should|would|will|what|which|who|when"
    r"|where|why|how|quel|quelle|quels|quelles|qui|que|quoi|quand|ou|pourquoi|comment|combien)\b"
    r"|\w-(?:t-)?(?:je|tu|il|elle|on|ce|ils|elles)\b"
)
_INSTRUMENT_SYNONYMS = {"lunette": "refracteur", "refractor": "refracteur", "newtonian": "newton"}


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    return "".join(ch for ch in value if not unicodedata.combining(ch)).replace("’", "'")


def named_ids(text: str, kind: Literal["slot", "booking", "telescope"]) -> set[str]:
    """Canonical IDs of one kind (`S-01`, `BK-001`, `T-02`) explicitly written in the text."""
    pattern, prefix, width = _ID_KINDS[kind]
    return {f"{prefix}-{int(digits):0{width}d}" for digits in pattern.findall(_normalized(text))}


def _one_id(text: str, kind: Literal["slot", "booking", "telescope"]) -> tuple[str | None, bool]:
    ids = named_ids(text, kind)
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
    min_altitude_deg: float | None = None


_CITIES = "lyon|fontainebleau|paris|chamonix|marseille|nice|bordeaux"
_CITY_PATTERN = re.compile(rf"\b(?:{_CITIES})\b")
_CLEAR_SKY_CUE = re.compile(r"\b(?:weather|meteo|clear|degage|beau|sans nuages?|cloud)\b")
_TARGET = r"(?:m0?\d{1,3}|c0?\d{1,3}|omega centauri)"
_TARGET_CUE = re.compile(rf"\b({_TARGET})\b")
_CONDITIONAL = re.compile(
    r"\b(?:if|when|whenever|(?<!at )once(?! more| again)|provided|assuming|supposing|in case"
    r"|as long as|as soon as|si|s'ils?|quand|lorsque|des que|tant que|pourvu|au cas ou"
    r"|sous reserve|a condition)\b"
)
# Polite formulas built on a conditional word. Fixed idioms count anywhere; those that could
# also open a real condition ("if you can confirm ...") only when they close a clause.
_COURTESY = re.compile(
    r"\b(?:s'il[ -](?:vous|te)[ -]plait|as soon as possible|des que possible|just in case)\b"
    r"|\b(?:si (?:c'est )?possible|si tu peux|si vous pouvez|if possible|if you can"
    r"|as soon as you can|des que (?:tu peux|vous pouvez)|au cas ou)"
    r"(?=\s*(?:(?:please|merci|thanks|thank you|s'il[ -](?:vous|te)[ -]plait)\b\s*)*"
    r"(?:[,.;:!?]|$))"
)
_NEGATIVE_CONDITIONAL = re.compile(r"\b(?:unless|except if|sauf si|a moins)\b")
# The only conditions the controller can evaluate, as complete statements.
_CONDITION_STATEMENT = re.compile(
    r"(?:that |que |qu')?(?:"
    r"(?P<sky>(?:the )?(?:night )?(?:sky|weather) (?:is|stays|remains) (?:clear|good|fine)"
    r"|(?:the )?skies (?:are|stay|remain) clear|it(?: is|'s) clear"
    r"|(?:le )?(?:ciel|temps) (?:est|reste|soit) (?:degage|clair|beau)"
    r"|(?:la )?meteo (?:est|reste|soit) (?:degagee|claire|bonne)|(?:il )?fait beau)"
    rf"|{_TARGET} (?:is|est|soit) (?:visible|observable)"
    rf"|{_TARGET} (?:is|rises|climbs|gets|stays|est|soit|monte|passe|reste|culmine) "
    r"(?:above|over|au-dessus de|a plus de) (?P<altitude>\d{1,2}) ?(?:°|deg(?:rees?|res?)?))"
)
# Day expressions `_requested_date` resolves.
_DAY = re.compile(
    rf"\b(?:{'|'.join(_WEEKDAYS)}|demain|tomorrow|aujourd'hui|today|\d{{4}}-\d{{2}}-\d{{2}})\b"
)
# Where and when a condition holds, the slot it guards and closing formulas may surround it.
_CONDITION_ADJUNCT = re.compile(
    rf"\b(?:(?:in|at|over|above|from|a|sur|depuis)\s+)?(?:{_CITIES})\b"
    rf"|\b(?:(?:on|le|du)\s+)?{_DAY.pattern}(?:\s+(?:soir|night|evening))?"
    r"|\bs-\d{1,4}\b|\b(?:then|alors|please|merci|thanks|thank you)\b|[,.;:!]"
)
_ALTITUDE_CUE = re.compile(r"\b(\d{1,2})\s*(?:°|deg(?:rees?|res?)?\b)")
_CLEAR_SKY_MAX_CLOUD_PCT = 40.0


def _write_refusal(t: str) -> Decision | None:
    """Why a normalized request must not trigger the write it mentions, if that is the case.

    A denial, several targets, contradictory or third-party instructions are never read as
    permission for a narrower single action. Shared by the deterministic controller
    (`interpret`) and by the host policy applied to an LLM planner (`requested_write`).
    """
    _, multi_slots = _one_id(t, "slot")
    booking_id, multi_bookings = _one_id(t, "booking")
    book = bool(_BOOK_VERBS.search(t))
    cancel = bool(_CANCEL_VERBS.search(t))

    if (book or cancel) and (
        _NEGATED_ACTION.search(t)
        or _NEGATED_ACTION_POST.search(t)
        or _NEGATION_NEAR_ACTION.search(t)
    ):
        return Decision("noop", "Aucune action effectuée : la demande contient une négation.")
    if t.strip(" .!?,") in _CONFIRMATION_ONLY or (
        booking_id and _APPROVAL_AS_CHAT.search(t) and not book
    ):
        return Decision(
            "noop",
            "Une confirmation dans le chat ne vaut pas une approbation humaine. Utilisez la reprise opérateur HITL.",
        )
    if (book or cancel) and _QUESTION_ABOUT_ACTION.search(t):
        return Decision(
            "clarify",
            "Cette question ne déclenche aucune action ; formulez explicitement l'opération souhaitée.",
        )
    if multi_slots or multi_bookings or (book and cancel):
        return Decision(
            "clarify", "Précisez une seule opération et un seul identifiant par demande."
        )
    if (book or cancel) and re.search(
        r"\b(?:pour|for)\s+(?!(?:moi|me|nous|us|le|la|les|un|une|des|mon|ma|mes|notre|nos|"
        r"ce|cet|cette|ces|my|our|the|a|an|this|that|tonight|tomorrow|today|demain|aujourd'hui|"
        r"observer|observing|imaging|astrophotography|"
        r"lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b)[a-z][\w-]*",
        t,
    ):
        return Decision(
            "clarify",
            "Les actions pour un tiers ne sont pas autorisées par cette interface locale.",
        )
    return None


def _stated_condition(plain: str) -> tuple[str, float | None] | None:
    """Classify the condition a booking states with a conditional word, if it states one.

    Only one positive statement about a clear sky (`sky`) or a Messier/Caldwell altitude
    (`target`, with the minimum altitude it names) can be evaluated. Anything else is
    `unsupported`, so that "book only if X" never becomes a write that ignores or inverts
    part of X.
    """
    conditionals = list(_CONDITIONAL.finditer(plain))
    negative = _NEGATIVE_CONDITIONAL.search(plain)
    if not conditionals and not negative:
        return None
    if negative or len(conditionals) > 1:
        return "unsupported", None
    # "Book S-01 if X" runs to the end; "If X, book S-01" states X before the verb.
    tail = plain[conditionals[0].end() :]
    booked_first = _BOOK_VERBS.search(plain[: conditionals[0].start()])
    clause = tail if booked_first else _BOOK_VERBS.split(tail, maxsplit=1)[0]
    statement = _CONDITION_STATEMENT.fullmatch(
        " ".join(_CONDITION_ADJUNCT.sub(" ", clause).split())
    )
    if statement is None:
        return "unsupported", None
    if statement["sky"]:
        return "sky", None
    return "target", float(statement["altitude"]) if statement["altitude"] else None


def interpret(text: str, *, reference_date: date | None = None) -> Decision:
    """Classify a bounded set of commands without considering MCP tool output.

    Unknown requests go to the historical astronomy agent; unclear *action*
    requests always require a more specific user instruction.
    """
    if not isinstance(text, str) or not text.strip() or len(text) > 4000:
        return Decision("clarify", "Veuillez formuler une demande de 1 à 4 000 caractères.")

    t = _normalized(text)
    refusal = _write_refusal(t)
    if refusal is not None:
        return refusal
    slot_id, _ = _one_id(t, "slot")
    booking_id, _ = _one_id(t, "booking")
    book = bool(_BOOK_VERBS.search(t))
    cancel = bool(_CANCEL_VERBS.search(t))
    if cancel:
        if not booking_id:
            parsed_date, issue = _requested_date(t, reference_date=reference_date)
            if not issue and parsed_date:
                return Decision(
                    "cancel",
                    "Annulation par date demandée : vérification du créneau puis validation humaine.",
                    local_date=parsed_date,
                )
            return Decision(
                "clarify", "Indiquez l'identifiant de réservation à annuler (ex. BK-001)."
            )
        return Decision(
            "cancel", "Annulation demandée : validation humaine obligatoire.", booking_id=booking_id
        )
    if book:
        if not slot_id:
            return Decision(
                "clarify",
                "Quel créneau souhaitez-vous réserver ? Indiquez un identifiant (ex. S-01).",
            )
        if booking_id:
            return Decision(
                "clarify", "Précisez uniquement un identifiant de créneau S-xx à réserver."
            )
        parsed_date, issue = _requested_date(t, reference_date=reference_date)
        if issue:
            return Decision("clarify", issue)
        cities = set(_CITY_PATTERN.findall(t))
        place_name = next(iter(cities)).capitalize() if len(cities) == 1 else None
        targets = {
            "C080" if raw == "omega centauri" else f"{raw[0].upper()}{int(raw[1:]):03d}"
            for raw in _TARGET_CUE.findall(t)
        }
        norm_target = next(iter(targets), None)
        kind, min_altitude = _stated_condition(_COURTESY.sub(" ", t)) or (None, None)
        require_clear = bool(_CLEAR_SKY_CUE.search(t)) or kind == "sky"
        # A booking tied to the sky or to a target is only executed when the whole condition
        # can be evaluated, for one site and one night.
        if kind or require_clear or targets:
            if kind == "unsupported" or len(targets) > 1 or _NEGATION.search(t):
                return Decision(
                    "clarify",
                    "Condition non vérifiable : seuls un ciel dégagé ou la visibilité d'un objet "
                    "Messier/Caldwell peuvent conditionner une réservation.",
                )
            if place_name is None:
                return Decision(
                    "clarify",
                    "Précisez un seul site d'observation connu pour vérifier cette condition.",
                )
            nights = {
                _requested_date(day, reference_date=reference_date)[0] for day in _DAY.findall(t)
            }
            if len(nights) != 1:
                return Decision(
                    "clarify", "Précisez une seule date (AAAA-MM-JJ) pour la nuit à vérifier."
                )
            if kind != "target" and norm_target and (angle := _ALTITUDE_CUE.search(t)):
                min_altitude = float(angle.group(1))
        return Decision(
            "book",
            "Réservation d'un créneau identifié.",
            slot_id=slot_id,
            local_date=parsed_date,
            place_name=place_name,
            require_clear_sky=require_clear,
            target_object_id=norm_target,
            min_altitude_deg=min_altitude,
        )

    if _READ_BOOKING.search(t) and (booking_id or re.search(r"\b(?:reservation|booking)\b", t)):
        if not booking_id:
            return Decision(
                "clarify", "Indiquez l'identifiant BK-xxx de la réservation à consulter."
            )
        return Decision("booking_info", "Consultation d'une réservation.", booking_id=booking_id)

    equipment_advice = bool(
        re.search(
            r"\b(?:meilleur|meilleure|best|ideal|focale|grossissement|photographier|photography|imaging)\b",
            t,
        )
    )
    explicit_availability = bool(
        re.search(
            r"\b(?:disponible|disponibles|disponibilite|disponibilites|available|availability|creneau|creneaux|slots?)\b",
            t,
        )
    )
    if (
        _AVAILABILITY.search(t)
        and _ASKING.search(t)
        and (explicit_availability or not equipment_advice)
    ):
        parsed_date, issue = _requested_date(t, reference_date=reference_date)
        if issue:
            return Decision("clarify", issue)
        period = (
            "evening"
            if re.search(r"\b(?:soir|soiree|evening|tonight)\b", t)
            else "morning"
            if re.search(r"\b(?:matin|morning)\b", t)
            else "afternoon"
            if re.search(r"\b(?:apres-midi|afternoon)\b", t)
            else None
        )
        return Decision(
            "availability",
            "Consultation des créneaux simulés.",
            local_date=parsed_date,
            period=period,
        )

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


def _paris_start(slot: dict[str, Any]) -> datetime:
    """Return a slot's `start_utc` as an aware Europe/Paris datetime."""
    return datetime.fromisoformat(str(slot["start_utc"]).replace("Z", "+00:00")).astimezone(_PARIS)


def requested_write(text: str) -> Literal["book", "cancel"] | None:
    """Return the single booking or cancellation the text explicitly asks for, if any.

    Applies the same refusals as `interpret` (negation, question, chat "approval", several
    targets, third party): this is the host-side check for writes proposed by an LLM planner.
    """
    t = _normalized(text)
    book, cancel = bool(_BOOK_VERBS.search(t)), bool(_CANCEL_VERBS.search(t))
    if book == cancel or _write_refusal(t) is not None:
        return None
    return "book" if book else "cancel"


def _preference_statement(t: str) -> Literal["set", "clear"] | None:
    """Classify a normalized sentence as stating a telescope preference, dropping one, or neither.

    Only a plain statement counts: a question, a negation, a condition or a sentence that
    also books or cancels never does. A forget verb drops the preference it precedes.
    """
    cue = _PREFERENCE_CUE.search(t)
    if (
        cue is None
        or _QUESTION.search(t)
        or _NEGATION.search(t)
        or _CONDITIONAL.search(_COURTESY.sub(" ", t))
        or _BOOK_VERBS.search(t)
        or _CANCELLATION_WORDS.search(t)
    ):
        return None
    if _FORGET_CUE.search(t[: cue.start()]):
        return "clear"
    # After the preference, "clear" reads as an adjective ("on clear nights"); any other
    # forget verb contradicts the statement.
    contradicted = {verb.group() for verb in _FORGET_CUE.finditer(t, cue.end())} - {"clear"}
    return None if contradicted else "set"


class MemoryWrite(NamedTuple):
    """A memory change the user's own text explicitly asks for, bound to the key it names."""

    op: Literal["set", "clear"]
    key: Literal["preferred_telescope_id", "preferred_site", "history"]
    value: str | None = None


_SITE_WORD = re.compile(r"\b(?:site|lieu|location|observatoire|observatory)\b")
_SITE_NAME = re.compile(r"\b(lyon|fontainebleau)\b")
_TELESCOPE_WORD = re.compile(r"\b(?:telescope|lunette|instrument)\b")
_HISTORY_WORD = re.compile(r"\b(?:historique|history)\b")


def _history_erasure(t: str) -> bool:
    """A plain request to erase the booking history (not a question, negation or action)."""
    return bool(
        _FORGET_CUE.search(t)
        and _HISTORY_WORD.search(t)
        and not (
            _QUESTION.search(t)
            or _NEGATION.search(t)
            or _BOOK_VERBS.search(t)
            or _CANCELLATION_WORDS.search(t)
        )
    )


def requested_memory_write(text: str) -> MemoryWrite | None:
    """Return the memory change the text explicitly asks for, or None.

    Each change names its key, so "forget my preferred site" can never authorise clearing
    the telescope preference (and vice versa).
    """
    t = _normalized(text)
    if _history_erasure(t):
        return MemoryWrite("clear", "history")
    statement = _preference_statement(t)
    if statement is None:
        return None
    sites = set(_SITE_NAME.findall(t))
    telescope_id, _ = _one_id(t, "telescope")
    if statement == "clear":
        about_site = (_SITE_WORD.search(t) or sites) and not _TELESCOPE_WORD.search(t)
        return MemoryWrite("clear", "preferred_site" if about_site else "preferred_telescope_id")
    if telescope_id:
        return MemoryWrite("set", "preferred_telescope_id", telescope_id)
    if len(sites) == 1 and not _TELESCOPE_WORD.search(t):
        return MemoryWrite("set", "preferred_site", sites.pop())
    return None


def _named_telescope(normalized_text: str, telescopes: list[dict[str, Any]]) -> str | None:
    """Return the only active telescope named in the text, by a word of its name or its `NNN mm` aperture."""
    words = {
        _INSTRUMENT_SYNONYMS.get(word, word) for word in re.findall(r"[a-z]+", normalized_text)
    }
    apertures = set(re.findall(r"\b(\d{2,4})\s*mm\b", normalized_text))
    matches = [
        telescope["id"]
        for telescope in telescopes
        if telescope.get("active")
        and (
            str(telescope.get("aperture_mm")) in apertures
            or words & set(re.findall(r"[a-z]{3,}", _normalized(str(telescope.get("name", "")))))
        )
    ]
    return matches[0] if len(matches) == 1 else None


def resolve_instrument_reference(text: str, state: dict[str, Any], *, reference_date: date) -> str:
    """Rewrite a request that names an instrument ("la lunette 120") into its explicit-ID form.

    Bookings and preferences only act on explicit IDs. The request is rewritten only when it
    is a plain preference statement or a booking, names exactly one active instrument of the
    live observatory `state`, and (for a booking) leaves a single open slot. Negations,
    questions, forget requests and every other case are returned unchanged.
    """
    norm = _normalized(text)
    if any(named_ids(norm, kind) for kind in _ID_KINDS):
        return text
    telescope_id = _named_telescope(norm, state.get("telescopes", []))
    if telescope_id is None:
        return text
    if _preference_statement(norm) == "set":
        return f"My preferred telescope is {telescope_id}"
    if requested_write(text) == "book":
        requested, issue = _requested_date(norm, reference_date=reference_date)
        if issue is None:
            open_slots = [
                slot["id"]
                for slot in state.get("slots", [])
                if slot.get("telescope_id") == telescope_id
                and slot.get("available")
                and (requested is None or _paris_start(slot).date().isoformat() == requested)
            ]
            if len(open_slots) == 1:
                return f"{text.strip()} {open_slots[0]}"
    return text


class ConversationalAgent:
    """Locally trusted entrypoint: natural language -> deterministic route -> real MCP.

    No code here may issue or accept a cancellation capability. Only the trusted
    local operator can call ``resume --approve`` after LangGraph's interrupt.
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
        a checkpointed cancellation workflow and leave it suspended.
        """
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("A non-empty trusted user_id is required")
        decision = interpret(text, reference_date=reference_date)
        if decision.route in ("clarify", "noop"):
            return {
                "route": decision.route,
                "status": "needs_clarification" if decision.route == "clarify" else "no_action",
                "message": decision.reason,
                "tool_trace": [],
            }
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
                        "route": "booking_info",
                        "status": "forbidden",
                        "message": "Cette réservation n'appartient pas à l'utilisateur courant.",
                        "tool_trace": trace,
                    }
                return {
                    "route": "booking_info",
                    "status": "completed",
                    "message": "Réservation trouvée (simulation).",
                    "data": booking,
                    "tool_trace": trace,
                }

            telescopes = self.gateway.call("observatory", "list_telescopes", {})
            trace.append({"tool": "list_telescopes", "status": "ok"})
            current_tool = "list_slots"
            slots = self.gateway.call("observatory", "list_slots", {"available_only": True})
            trace.append({"tool": "list_slots", "status": "ok"})
            names = {item["id"]: item["name"] for item in telescopes if item.get("active")}
            sites = {item["id"]: item.get("site") for item in telescopes if item.get("active")}
            matches = []
            for slot in slots:
                if slot.get("telescope_id") not in names:
                    continue
                start = _paris_start(slot)
                if decision.local_date and start.date().isoformat() != decision.local_date:
                    continue
                if not _match_period(start.timetz(), decision.period):
                    continue
                matches.append(
                    {
                        **slot,
                        "telescope_name": names[slot["telescope_id"]],
                        "telescope_site": sites[slot["telescope_id"]],
                        "start_local": start.isoformat(),
                    }
                )
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
                    "route": decision.route,
                    "status": "not_found",
                    "message": "Réservation introuvable.",
                    "error": {"code": code},
                    "tool_trace": trace,
                }
            return {
                "route": decision.route,
                "status": "tool_error",
                "message": "Le serveur MCP de l'observatoire n'est pas disponible.",
                "error": {"code": "MCP_UNAVAILABLE"},
                "tool_trace": trace,
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
        try:
            if decision.route == "cancel" and resolved_booking_id is None and decision.local_date:
                resolved_booking_id = self._booking_on_date(decision.local_date, user_id, pre_trace)
                if resolved_booking_id is None:
                    return {
                        "route": "clarify",
                        "status": "needs_clarification",
                        "message": "Plusieurs ou aucune réservation active à cette date ; précisez BK-xxx.",
                        "steps": 1,
                        "tool_trace": pre_trace,
                    }
            if decision.route == "book" and (
                decision.require_clear_sky or decision.target_object_id
            ):
                refusal = self._unmet_booking_condition(decision, pre_trace)
                if refusal is not None:
                    return {
                        "route": "action",
                        "intent": "book",
                        "status": "impossible",
                        **refusal,
                        "tool_trace": pre_trace,
                    }
        except Exception:
            # Fail closed before any write; do not reveal credentials or exception messages.
            return {
                "route": "action",
                "intent": decision.route,
                "status": "tool_error",
                "message": "Le serveur MCP n'est pas disponible ; aucune action effectuée.",
                "error": {"code": "MCP_UNAVAILABLE"},
                "steps": 1,
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

    def _traced_call(
        self, server: str, tool: str, args: dict[str, Any], trace: list[dict[str, Any]]
    ) -> Any:
        """Call an MCP tool; the trace entry stays in `error` status if the call raises."""
        entry = {"tool": tool, "status": "error", "args": args}
        trace.append(entry)
        result = self.gateway.call(server, tool, args)
        entry["status"] = "ok"
        return result

    def _booking_on_date(
        self, local_date: str, user_id: str, trace: list[dict[str, Any]]
    ) -> str | None:
        """Return the user's confirmed booking starting on `local_date` (Paris) when it is unique."""
        state = self._traced_call("observatory", "get_observatory_state", {}, trace)
        slots_by_id = {slot["id"]: slot for slot in state.get("slots", [])}
        matching = [
            booking["id"]
            for booking in state.get("bookings", [])
            if booking.get("user_id") == user_id
            and booking.get("status") == "confirmed"
            and booking.get("slot_id") in slots_by_id
            and _paris_start(slots_by_id[booking["slot_id"]]).date().isoformat() == local_date
        ]
        return matching[0] if len(matching) == 1 else None

    def _unmet_booking_condition(
        self, decision: Decision, trace: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Check the weather / visibility condition attached to a booking; `None` means it holds."""
        geo = self._traced_call("sky", "geocode", {"place_name": decision.place_name}, trace)
        if not geo.get("found"):
            return {"message": "Site d'observation introuvable.", "steps": 1}
        site = {"lat": float(geo["lat"]), "lon": float(geo["lon"]), "obs_date": decision.local_date}

        if decision.require_clear_sky:
            wx = self._traced_call(
                "sky", "weather", {**site, "place_name": decision.place_name}, trace
            )
            if (
                wx.get("available") is False
                or float(wx.get("cloud_cover_pct", 100)) > _CLEAR_SKY_MAX_CLOUD_PCT
            ):
                return {
                    "message": "Ciel couvert ou hors horizon météo : réservation refusée.",
                    "steps": 2,
                }

        if decision.target_object_id:
            vis = self._traced_call(
                "sky", "visibility", {"object_ids": [decision.target_object_id], **site}, trace
            )
            # `sky.visibility` maps catalog IDs to per-object reports; exactly one was requested.
            reports = vis.get("objects") if isinstance(vis, dict) else None
            report = next(iter(reports.values()), None) if isinstance(reports, dict) else None
            max_alt = report.get("max_altitude_deg") if isinstance(report, dict) else None
            reachable = (
                isinstance(report, dict)
                and report.get("observable") is True
                and isinstance(max_alt, (int, float))
                and max_alt >= (decision.min_altitude_deg or 0.0)
            )
            if not reachable:
                return {
                    "message": "Cible non observable au-dessus de l'altitude requise depuis ce site.",
                    "steps": 2,
                }
        return None

    def _astronomy(self, text: str) -> dict[str, Any]:
        if self._astronomy_runner is not None:
            response = self._astronomy_runner(text)
            return {**response, "route": "astronomy"}
        from astropilote.agent.graph import run_agent

        answer, _, record = run_agent(text)
        return {
            "route": "astronomy",
            "status": "abstained" if answer.abstained else "answered",
            "message": answer.answer,
            "sources": answer.sources,
            "tool_trace": record.tool_calls,
            "tokens_in": record.tokens_in,
            "tokens_out": record.tokens_out,
            "cost_usd": record.cost_usd,
        }
