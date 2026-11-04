"""Prompt construction, full-body chunk formatting, and multi-turn conversation compaction."""

from __future__ import annotations

import json
from typing import Any

from astropilote.agent.i18n import language_name
from astropilote.agent.schemas import Chunk, ConversationTurn, PlanItem


def clean_chunk_body(txt: str, max_chars: int = 1400) -> str:
    """Strip indexing header prefixes while preserving the full Wikipedia/guide passage (~1,400 chars)."""
    cleaned = txt
    if cleaned.startswith("[") and "]\n" in cleaned:
        cleaned = cleaned.split("]\n", 1)[1]
    if cleaned.startswith("Section:"):
        parts = cleaned.split("\n", 1)
        cleaned = parts[1] if len(parts) > 1 else cleaned
    cleaned = cleaned.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    clipped = cleaned[:max_chars].rsplit(" ", 1)[0]
    return f"{clipped}…"


def format_retrieved_passages(
    chunks: list[Chunk], max_chunks: int = 6, max_chars: int = 1400
) -> str:
    """Format up to `max_chunks` retrieved corpus passages without destructive truncation."""
    return "\n\n".join(
        f"[{c.doc_id}] ({c.section}): {clean_chunk_body(c.text, max_chars=max_chars)}"
        for c in chunks[:max_chunks]
    )


def format_conversation_history_for_composer(history: list[ConversationTurn] | None) -> str:
    """Format prior conversation turns concisely so `compose_node` maintains multi-turn continuity."""
    if not history:
        return ""
    lines: list[str] = []
    for idx, turn in enumerate(history[-4:], start=1):
        ctx_parts: list[str] = []
        if turn.location_name:
            ctx_parts.append(f"location={turn.location_name}")
        if turn.obs_date:
            ctx_parts.append(f"date={turn.obs_date}")
        if turn.sources:
            ctx_parts.append(f"cited={turn.sources[:5]}")
        ctx_meta = f" ({', '.join(ctx_parts)})" if ctx_parts else ""
        excerpt = (turn.answer_excerpt or "").strip().replace("\n", " ")[:240]
        excerpt_part = f" -> A: {excerpt}" if excerpt else ""
        lines.append(f"Turn {idx}{ctx_meta}: Q: {turn.question.strip()[:200]}{excerpt_part}")
    return "\n".join(lines)


def build_compose_prompts(
    *,
    question: str,
    lang: str,
    deterministic_json: dict[str, Any],
    context_snippets: str,
    candidate_cite_ids: list[str],
    plan_items: list[PlanItem],
    unindexed: list[str],
    priority_dsos: list[str],
    catalog_summary: list[dict[str, Any]],
    retry_errors: list[str] | None = None,
    conversation_history: list[ConversationTurn] | None = None,
    no_darkness_impossible: bool = False,
) -> tuple[str, str]:
    """Construct `(sys_prompt, user_prompt)` for grounded astronomy synthesis."""
    lang_label = language_name(lang)

    retry_note = ""
    if retry_errors:
        retry_note = (
            "\n\nCORRECTION REQUIRED FROM VALIDATOR: "
            + "; ".join(retry_errors)
            + ". ONLY cite IDs from `Available corpus citation IDs`, and ensure at least one retrieved `[DOC_ID]` is cited inline."
        )

    sys_prompt = (
        "You are AstroPilote, an expert astronomy and astrophotography assistant.\n"
        f"LANGUAGE: Write your ENTIRE response in {lang_label} (prose, headings, lists), keeping bracketed IDs like `[M031]` unchanged.\n"
        "Strict grounding rules:\n"
        "1. Ground coordinates, darkness hours, Moon phase/illumination/rise/set, cloud cover %, clear windows, altitudes, and FOV strictly in `Deterministic Tool Outputs (JSON)`. Never claim live weather/Moon failed when present.\n"
        "2. If `night_weather.available` is false (> `max_weather_forecast_horizon_days` = 14 days), state that weather forecasts only extend 14 days ahead while still answering sky visibility.\n"
        "3. WHITE NIGHTS & MIDNIGHT SUN: If `white_night_warning` is present or `night_ephemeris.has_astronomical_darkness` is false:\n"
        "   - When `min_sun_altitude_deg > -12.0` (Midnight Sun or continuous civil/bright twilight), NEVER invent darkness hours or schedule deep-sky targets; clearly explain why solar altitude prevents deep-sky observing at this latitude and date.\n"
        "   - When `-18.0 < min_sun_altitude_deg <= -12.0` (summer solstice nautical twilight), state that there is no full astronomical darkness (nautical twilight only) and focus on bright targets around local solar midnight.\n"
        "4. When `framing_advice` is present, cite exact sensor FOV arcminutes (e.g. `123.8 x 82.5` arcminutes) and whether the object fits.\n"
        "5. Politely correct any false premise (wrong constellation/type) using `catalog_objects` and cite `[CON_XXX]` if retrieved.\n"
        "6. If `location.provided` is false, NEVER assume or invent a default city; suggest targets and invite the user to share their city/coordinates.\n"
        "7. Cite retrieved sources inline using ONLY exact corpus IDs from `Available corpus citation IDs` (e.g. `[M031]`, `[C020]`). NEVER put JSON key names like `[night_ephemeris]`, `[optics]`, or `[Deterministic Tool Outputs]` in brackets.\n"
        "8. USER EQUIPMENT PROFILE & PHOTOGRAPHY SETTINGS: When `optics.user_equipment_profile` or `focal_length_mm` is set, tailor framing (`fov_arcmin`, % fill), sub-exposure (NPF/500-rule on tripod vs tracked subs), ISO/Gain, and filter advice to that rig. When no telescope/camera is specified, give general naked-eye, binocular, and telescope advice without assuming a specific sensor.\n"
        "9. PRACTICAL OBJECT GUIDE: For each target you present from `catalog_objects`, cover orientation (constellation, direction/altitude, how to find), visual & photo look (magnitude, size vs 30' Moon, morphology/colors), and how to observe/shoot it.\n"
        "10. MULTI-OBJECT DEPTH & PACING: Skip generic greetings and never stop after 1 object when multiple targets fit:\n"
        "   - For session plans or target recommendations (when dark skies exist), start with a concise Night & Weather Conditions summary (3–4 bullets), then write a focused 80–110 word subsection for **EVERY ONE** of the 3–5 scheduled/recommended targets so all targets are completed thoroughly, and end with 2 lines of session tips.\n"
        "   - For single-object questions, give a thorough guide to that primary object AND a concise section highlighting 1–2 nearby companions/neighbors from `catalog_objects`."
    )

    if no_darkness_impossible:
        multi_target_hint = (
            " Explain clearly why Midnight Sun / bright twilight (`min_sun_altitude_deg > -12°`) makes deep-sky observing physically impossible on this date, "
            "mention solar/lunar alternatives or waiting for darker autumn/winter months, and cite at least one retrieved `[DOC_ID]` from `Available corpus citation IDs`."
        )
    elif len(plan_items) >= 2:
        multi_target_hint = (
            f" Cover ALL {len(plan_items)} scheduled targets in order "
            f"({[pi.object_id for pi in plan_items]}) with ~90 words per target after a concise Night & Weather summary."
        )
    elif not unindexed and not priority_dsos and len(catalog_summary) >= 2:
        multi_target_hint = (
            f" Present 3 to {len(catalog_summary)} distinct targets from `catalog_objects` "
            f"({[c['doc_id'] for c in catalog_summary]}) with ~95 words of astronomical and practical detail per target."
        )
    elif not unindexed and priority_dsos and len(catalog_summary) >= 2:
        companion_ids = [c["doc_id"] for c in catalog_summary if c["doc_id"] not in priority_dsos][
            :2
        ]
        multi_target_hint = (
            f" Cover {priority_dsos} in rich detail and also present nearby companion/neighbor target(s) {companion_ids} (citing their `[DOC_ID]`)."
            if companion_ids
            else ""
        )
    else:
        multi_target_hint = ""

    history_str = format_conversation_history_for_composer(conversation_history)
    history_section = (
        f"Previous Conversation Context (for continuity on follow-up questions; ground all numbers & citations in current tool/corpus data):\n{history_str}\n\n"
        if history_str
        else ""
    )

    user_prompt = (
        f"{history_section}"
        f"User Question: {question}\n\n"
        f"Deterministic Tool Outputs (JSON):\n{json.dumps(deterministic_json, ensure_ascii=False, separators=(',', ':'))}\n\n"
        f"Retrieved Corpus Passages:\n{context_snippets}\n\n"
        f"Available corpus citation IDs: {candidate_cite_ids}{retry_note}\n\n"
        f"Write a complete, well-structured response in {lang_label}, grounded in these exact figures and citing `[DOC_ID]` sources inline.{multi_target_hint}"
    )
    return sys_prompt, user_prompt
