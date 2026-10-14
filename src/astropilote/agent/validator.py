"""Deterministic Python verifier (`validate` node) checking citations, altitudes, and darkness windows."""

from __future__ import annotations

import re
from datetime import time

from astropilote.agent.schemas import AstroAnswer
from astropilote.agent.state import AstroState
from astropilote.tools.catalog import normalize_object_id

_INLINE_CITE_RE = re.compile(
    r"\[((?:M|C)\d{1,3}|NGC\s*\d{1,4}|IC\s*\d{1,4}|CON_[A-Z]{3})\]",
    flags=re.IGNORECASE,
)


def _time_to_night_minutes(t: time) -> int:
    """Map evening/morning times onto a continuous axis anchored at 12:00 noon (0..1440)."""
    mins = t.hour * 60 + t.minute
    return mins - 720 if mins >= 720 else mins + 720


def _parse_hhmm(hhmm: str) -> time | None:
    try:
        parts = hhmm.strip().split(":")
        return time(int(parts[0]), int(parts[1]))
    except Exception:
        return None


def verify_astro_answer(answer: AstroAnswer, state: AstroState) -> AstroAnswer:
    """Verify the 4 invariants required before sending `AstroAnswer` to the user:

    1. Every cited source (`answer.sources` and `item.sources`) was actually retrieved.
    2. Every `PlanItem.altitude_deg` matches the Skyfield/ephemeris calculation within ±2.0°.
    3. Every `PlanItem.start` falls inside the astronomical darkness window (`night_window`).
    4. When `answer.abstained` is True, `sources` and `plan` must be empty.
    """
    errors: list[str] = []
    retrieved_ids = {c.doc_id for c in state.get("retrieved", [])}
    calculated_alts = state.get("calculated_altitudes", {})
    night_window = state.get("night_window")

    # Check 4: Abstention consistency
    if answer.abstained:
        if answer.sources:
            errors.append(
                f"Abstained answer must cite zero sources, but cited: {answer.sources}"
            )
        if answer.plan:
            errors.append("Abstained answer must have an empty observation plan.")
    else:
        if not answer.sources:
            errors.append("Non-abstained answer must cite at least one retrieved source.")

    # Check 1: Every cited source in `answer.sources` AND inline in `answer.answer` must be in `retrieved_ids`
    for src in answer.sources:
        if src not in retrieved_ids:
            errors.append(f"Cited source '{src}' was not found in retrieved chunks {sorted(retrieved_ids)}.")

    for match in _INLINE_CITE_RE.finditer(answer.answer or ""):
        raw_cite = match.group(1)
        norm_cite = normalize_object_id(raw_cite)
        if answer.abstained:
            errors.append(
                f"Abstained answer text must not contain inline citations, found '[{raw_cite}]'."
            )
        elif norm_cite not in retrieved_ids:
            errors.append(
                f"Inline citation '[{raw_cite}]' (normalized '{norm_cite}') in answer text was not retrieved."
            )

    for item in answer.plan:
        for src in item.sources:
            if src not in retrieved_ids:
                errors.append(
                    f"Plan item '{item.object_id}' cited unretrieved source '{src}'."
                )

        # Check 2: Altitude within ±2.0° of calculated ephemerides
        obj_alts = calculated_alts.get(item.object_id, [])
        if obj_alts:
            closest_diff = min(abs(item.altitude_deg - a) for a in obj_alts)
            if closest_diff > 2.0:
                errors.append(
                    f"Plan item '{item.object_id}' altitude {item.altitude_deg}° differs by "
                    f"{closest_diff:.1f}° (> 2.0°) from calculated ephemeris values {obj_alts}."
                )

        # Check 3: Time slot inside astronomical darkness window
        if night_window:
            t_start = _parse_hhmm(night_window[0])
            t_end = _parse_hhmm(night_window[1])
            if t_start and t_end:
                slot_m = _time_to_night_minutes(item.start)
                win_start_m = _time_to_night_minutes(t_start)
                win_end_m = _time_to_night_minutes(t_end)
                if not (win_start_m <= slot_m <= win_end_m):
                    errors.append(
                        f"Plan item '{item.object_id}' start time {item.start.strftime('%H:%M')} "
                        f"falls outside astronomical darkness ({night_window[0]} -> {night_window[1]})."
                    )

    checked = answer.model_copy(deep=True)
    checked.verification_errors = errors
    checked.verified = len(errors) == 0
    return checked
