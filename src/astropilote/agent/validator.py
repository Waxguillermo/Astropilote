"""Deterministic Python verifier (`validate` node) checking citations, altitudes, and darkness windows."""

from __future__ import annotations

import re
from datetime import time

from astropilote.agent.citations import (
    _BRACKET_GROUP_RE,
    _SINGLE_TOKEN_RE,
    extract_inline_citations,
)
from astropilote.agent.schemas import AstroAnswer
from astropilote.agent.state import AstroState
from astropilote.config import settings
from astropilote.tools.catalog import normalize_object_id, resolve_object_record


def _time_to_night_minutes(t: time) -> int:
    """Map evening/morning times onto a continuous axis anchored at 12:00 noon (0..1440)."""
    mins = t.hour * 60 + t.minute
    return mins - 720 if mins >= 720 else mins + 720


def _parse_hhmm(hhmm: str) -> time | None:
    try:
        parts = hhmm.strip().split(":")
        return time(int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return None


def _is_retrieved_or_alias(identifier: str, retrieved_ids: set[str]) -> bool:
    """Return True if `identifier` (or its canonical corpus doc_id) is in `retrieved_ids`."""
    norm = normalize_object_id(identifier)
    if identifier in retrieved_ids or norm in retrieved_ids:
        return True
    rec = resolve_object_record(identifier)
    if rec and rec.get("doc_id") in retrieved_ids:
        return True
    return False


def verify_astro_answer(answer: AstroAnswer, state: AstroState) -> AstroAnswer:
    """Verify invariants required before sending `AstroAnswer` to the user:

    1. Every cited source (`answer.sources`, inline `[...]` citations, and `item.sources`) was actually retrieved.
    2. Every `PlanItem.altitude_deg` matches the ephemeris calculation within `settings.altitude_tolerance_deg`.
    3. Both `PlanItem.start` and `PlanItem.end` fall inside the astronomical darkness window (+/- twilight margin).
    4. When `answer.abstained` is True, `sources`, inline citations, and `plan` must be empty.
    """
    errors: list[str] = []
    retrieved_ids = {c.doc_id for c in state.get("retrieved", [])}
    calculated_alts = state.get("calculated_altitudes", {})
    night_window = state.get("night_window")
    tol_deg = settings.altitude_tolerance_deg
    margin_min = settings.twilight_margin_minutes

    if answer.abstained:
        if answer.sources:
            errors.append(f"Abstained answer must cite zero sources, but cited: {answer.sources}")
        if answer.plan:
            errors.append("Abstained answer must have an empty observation plan.")
    elif retrieved_ids and not answer.sources:
        errors.append("Non-abstained answer must cite at least one retrieved source.")

    for src in answer.sources:
        if not _is_retrieved_or_alias(src, retrieved_ids):
            errors.append(
                f"Cited source '{src}' was not found in retrieved chunks {sorted(retrieved_ids)}."
            )

    for raw_cite in extract_inline_citations(answer.answer or ""):
        norm_cite = normalize_object_id(raw_cite)
        if answer.abstained:
            errors.append(
                f"Abstained answer text must not contain inline citations, found '[{raw_cite}]'."
            )
        elif not _is_retrieved_or_alias(raw_cite, retrieved_ids):
            errors.append(
                f"Inline citation '[{raw_cite}]' (normalized '{norm_cite}') in answer text was not retrieved."
            )

    for item in answer.plan:
        for src in item.sources:
            if not _is_retrieved_or_alias(src, retrieved_ids):
                errors.append(f"Plan item '{item.object_id}' cited unretrieved source '{src}'.")

        obj_alts = calculated_alts.get(item.object_id, [])
        if obj_alts:
            closest_diff = min(abs(item.altitude_deg - a) for a in obj_alts)
            if closest_diff > tol_deg:
                errors.append(
                    f"Plan item '{item.object_id}' altitude {item.altitude_deg}° differs by "
                    f"{closest_diff:.1f}° (> {tol_deg:.1f}°) from calculated ephemeris values {obj_alts}."
                )

        if night_window:
            t_start = _parse_hhmm(night_window[0])
            t_end = _parse_hhmm(night_window[1])
            if t_start and t_end:
                start_m = _time_to_night_minutes(item.start)
                end_m = _time_to_night_minutes(item.end)
                win_start_m = _time_to_night_minutes(t_start) - margin_min
                win_end_m = _time_to_night_minutes(t_end) + margin_min
                if not (win_start_m <= start_m <= win_end_m):
                    errors.append(
                        f"Plan item '{item.object_id}' start time {item.start.strftime('%H:%M')} "
                        f"falls outside astronomical darkness ({night_window[0]} -> {night_window[1]})."
                    )
                if not (win_start_m <= end_m <= win_end_m):
                    errors.append(
                        f"Plan item '{item.object_id}' end time {item.end.strftime('%H:%M')} "
                        f"falls outside astronomical darkness ({night_window[0]} -> {night_window[1]})."
                    )
                if end_m < start_m:
                    errors.append(
                        f"Plan item '{item.object_id}' end time {item.end.strftime('%H:%M')} "
                        f"precedes start time {item.start.strftime('%H:%M')}."
                    )

    checked = answer.model_copy(deep=True)
    checked.verification_errors = errors
    checked.verified = len(errors) == 0
    return checked


def scrub_unverified_citations(answer: AstroAnswer, state: AstroState) -> AstroAnswer:
    """Clean hallucinated citation tokens from `answer` if verification still failed after retries, preserving `verified=False`."""
    if answer.verified:
        return answer
    retrieved_ids = {c.doc_id for c in state.get("retrieved", [])}
    cleaned = answer.model_copy(deep=True)
    cleaned.sources = [s for s in cleaned.sources if _is_retrieved_or_alias(s, retrieved_ids)]

    def _clean_bracket_group(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        parts = [p.strip() for p in re.split(r"[,;/]+", inner) if p.strip()]
        if not parts or not all(_SINGLE_TOKEN_RE.match(p) for p in parts):
            return match.group(0)
        valid_parts = [p for p in parts if _is_retrieved_or_alias(p, retrieved_ids)]
        if not valid_parts:
            return ""
        return f"[{', '.join(valid_parts)}]"

    cleaned_text = _BRACKET_GROUP_RE.sub(_clean_bracket_group, cleaned.answer or "")
    cleaned.answer = re.sub(r"[ \t]{2,}", " ", cleaned_text).strip()
    cleaned.verified = False
    return cleaned
