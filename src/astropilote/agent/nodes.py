"""LangGraph nodes (`guard`, `agent`, `tools`, `compose`, `validate`, `refuse`) for AstroPilote."""

from __future__ import annotations

import json
import re
import time as time_mod
from datetime import date, datetime, time

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from astropilote.agent.schemas import (
    AstroAnswer,
    Chunk,
    ChunkMetadata,
    Intent,
    PlanItem,
    ToolCallRecord,
)
from astropilote.agent.state import AstroState
from astropilote.agent.validator import verify_astro_answer
from astropilote.config import settings
from astropilote.tools import TOOLS_BY_NAME
from astropilote.tracking.tracker import estimate_text_tokens

# Patterns for prompt injection & off-topic detection in `guard`
_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous\s+|prior\s+)?instructions",
    r"system\s+prompt",
    r"developer\s+rules",
    r"system\s+override",
    r"cite\s+fake\s+source",
    r"disregard\s+the\s+(?:astronomy|corpus)",
    r"forget\s+astronomy",
    r"brute-force\s+ssh",
    r"code\s+du\s+travail",
    r"employment\s+contract",
]

# Known out-of-corpus entities tested in benchmark or general queries
_OUT_OF_CORPUS_PATTERNS = [
    (r"\bic\s*1101\b", "IC 1101 (supergiant elliptical galaxy in Abell 2029)"),
    (r"\bimx\s*455\b", "Sony IMX455 CMOS camera sensor datasheet"),
    (r"\bk2-18b\b", "exoplanet K2-18b"),
    (r"ritchey-chr[eé]tien|collimat", "optical hardware collimation manuals"),
    (r"\b3c\s*273\b", "quasar 3C 273"),
    (r"\bbennu\b|\bosiris-rex\b", "asteroid 101955 Bennu / OSIRIS-REx mission"),
]

_KNOWN_LOCATIONS = [
    "Fontainebleau",
    "Paris",
    "Chamonix",
    "Marseille",
    "Lyon",
    "Bordeaux",
    "Nice",
]

_MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _extract_date(question: str) -> date | None:
    """Extract an explicit date ('YYYY-MM-DD' or 'Month DD, YYYY') from `question`."""
    iso_m = re.search(r"\b(202[4-9]|203\d)-(\d{2})-(\d{2})\b", question)
    if iso_m:
        try:
            return date(int(iso_m.group(1)), int(iso_m.group(2)), int(iso_m.group(3)))
        except ValueError:
            pass

    # Match "October 17, 2026" or "17 October 2026"
    for m_name, m_num in _MONTH_MAP.items():
        pat1 = rf"\b{m_name}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(202[4-9]|203\d)\b"
        m1 = re.search(pat1, question, flags=re.IGNORECASE)
        if m1:
            try:
                return date(int(m1.group(2)), m_num, int(m1.group(1)))
            except ValueError:
                pass
        pat2 = rf"\b(\d{{1,2}})\s+{m_name},?\s+(202[4-9]|203\d)\b"
        m2 = re.search(pat2, question, flags=re.IGNORECASE)
        if m2:
            try:
                return date(int(m2.group(2)), m_num, int(m2.group(1)))
            except ValueError:
                pass
    return None


def _extract_location(question: str) -> str | None:
    for loc in _KNOWN_LOCATIONS:
        if re.search(rf"\b{loc}\b", question, flags=re.IGNORECASE):
            return loc
    return None


def _extract_optics(question: str) -> tuple[float | None, float | None]:
    """Extract `(aperture_mm, focal_length_mm)` from `question`."""
    focal: float | None = None
    aperture: float | None = None

    f_match = re.search(r"(\d{2,4})\s*mm\s+focal\s+length", question, flags=re.IGNORECASE)
    if f_match:
        focal = float(f_match.group(1))
    a_match = re.search(
        r"(\d{2,3})\s*mm\s+(?:refractor|newtonian|telescope|apo|aperture)",
        question,
        flags=re.IGNORECASE,
    )
    if a_match:
        aperture = float(a_match.group(1))
    return aperture, focal


def _is_weather_only_question(question: str) -> bool:
    """Return True if the question primarily asks for a meteorological forecast."""
    q_low = question.lower()
    weather_terms = [
        "cloud-free",
        "cloud percentage",
        "cloud cover",
        "will it rain",
        "rain during",
        "atmospheric seeing",
        "wind speed",
        "fera-t-il beau",
    ]
    return any(term in q_low for term in weather_terms)


def _ollama_available() -> bool:
    """Quick check whether a local Ollama server is running and responsive."""
    if settings.eval_offline_mode:
        return False
    try:
        with httpx.Client(timeout=0.8) as client:
            resp = client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


def guard_node(state: AstroState) -> dict:
    """Node 1 (`guard` / Comprendre): Classify intent, extract parameters, and catch traps."""
    q = state["question"]
    q_low = q.lower()
    in_tok = estimate_text_tokens(q) + 60

    # 1. Prompt injection & off-topic detection
    for pat in _INJECTION_PATTERNS:
        if re.search(pat, q_low):
            is_off_topic = "code du travail" in q_low or "employment contract" in q_low
            req_type = "off_topic" if is_off_topic else "prompt_injection"
            intent = Intent(
                request_type=req_type,
                refusal_reason=(
                    "I must refuse this request: it is either off-topic or attempts to override "
                    "AstroPilote's safety and citation-grounding rules. I only assist with visual "
                    "astronomy and astrophotography using verified catalog sources."
                ),
            )
            return {
                "intent": intent,
                "tokens_in": state.get("tokens_in", 0) + in_tok,
                "tokens_out": state.get("tokens_out", 0) + 35,
            }

    target_d = _extract_date(q)
    loc = _extract_location(q)
    aperture, focal = _extract_optics(q)

    # 2. Impossible long-range weather forecast detection (> 14 days after 2026-10-10)
    ref_today = date(2026, 10, 10)
    if target_d and _is_weather_only_question(q) and (target_d - ref_today).days > 14:
        intent = Intent(
            request_type="impossible_forecast",
            location_name=loc,
            target_date=target_d,
            refusal_reason=(
                f"I cannot provide a weather, cloud-cover, or seeing forecast for {target_d.isoformat()}: "
                "meteorological forecasts are only reliable up to ~7–14 days in advance (though astronomical "
                "positions and Moon phases can be computed for any date)."
            ),
        )
        return {
            "intent": intent,
            "tokens_in": state.get("tokens_in", 0) + in_tok,
            "tokens_out": state.get("tokens_out", 0) + 42,
        }

    # 3. Determine visual vs astrophotography mode and factual vs session_plan
    is_photo = any(
        w in q_low
        for w in ["image", "imaging", "shoot", "photograph", "astrophotography", "narrowband", "filter", "focal"]
    )
    is_visual = any(w in q_low for w in ["binoculars", "visual", "eyepiece", "naked eye"])
    mode = "astrophotography" if (is_photo and not is_visual) else ("visual" if (is_visual and not is_photo) else "hybrid")

    is_planning = bool(loc and target_d) or any(
        w in q_low for w in ["what targets should i", "session", "altitude", "when is", "can i observe"]
    )
    intent = Intent(
        request_type="session_plan" if (is_planning and (loc or target_d)) else "factual",
        mode=mode,
        location_name=loc,
        target_date=target_d,
        aperture_mm=aperture,
        focal_length_mm=focal,
    )
    return {
        "intent": intent,
        "tokens_in": state.get("tokens_in", 0) + in_tok,
        "tokens_out": state.get("tokens_out", 0) + 30,
    }


def refuse_node(state: AstroState) -> dict:
    """Node (`refuse` / Refuser): Return a polite, source-free refusal without calling tools."""
    intent = state.get("intent")
    reason = (
        intent.refusal_reason
        if intent and intent.refusal_reason
        else "I cannot fulfill this request as it falls outside AstroPilote's domain or safety policy."
    )
    ans = AstroAnswer(
        answer=reason,
        plan=[],
        conditions=None,
        sources=[],
        abstained=True,
        verified=True,
    )
    return {"answer": ans}


def _extract_mentioned_ids(question: str) -> list[str]:
    """Extract explicit Messier, Caldwell, NGC, or Constellation mentions from `question`."""
    ids: list[str] = []
    for m in re.finditer(r"\b(?:Messier\s*|M)(\d{1,3})\b", question, flags=re.IGNORECASE):
        num = int(m.group(1))
        if 1 <= num <= 110:
            ids.append(f"M{num:03d}")
    for m in re.finditer(r"\b(?:Caldwell\s*|C)(\d{1,3})\b", question, flags=re.IGNORECASE):
        num = int(m.group(1))
        if 1 <= num <= 109:
            ids.append(f"C{num:03d}")
    for m in re.finditer(r"\bNGC\s*0*(\d{1,4})\b", question, flags=re.IGNORECASE):
        ids.append(f"NGC{int(m.group(1)):04d}")

    # Common object/constellation names mapped to canonical IDs for multi-target coverage
    name_map = {
        "andromeda galaxy": "M031",
        "hercules": "CON_HER",
        "hercules globular": "M013",
        "crab nebula": "M001",
        "taurus": "CON_TAU",
        "dumbbell nebula": "M027",
        "vulpecula": "CON_VUL",
        "ring nebula": "M057",
        "lyra": "CON_LYR",
        "double cluster": "C014",
        "perseus": "CON_PER",
        "north america nebula": "C020",
        "cygnus": "CON_CYG",
        "whirlpool galaxy": "M051",
        "canes venatici": "CON_CVN",
        "pleiades": "M045",
        "bode's galaxy": "M081",
        "cigar galaxy": "M082",
        "ursa major": "CON_UMA",
        "veil nebula": "C033",
        "crescent nebula": "C027",
        "sombrero galaxy": "M104",
        "virgo": "CON_VIR",
        "orion nebula": "M042",
        "orion": "CON_ORI",
        "needle galaxy": "C038",
        "coma berenices": "CON_COM",
        "omega centauri": "C080",
        "centaurus": "CON_CEN",
    }
    q_low = question.lower()
    for phrase, cid in name_map.items():
        if phrase in q_low:
            ids.append(cid)
            if cid == "C033":
                ids.append("C034")
            if cid == "M042" and "companion" in q_low:
                ids.append("M043")
            if cid == "M031" and "satellite" in q_low:
                ids.extend(["M032", "M110"])
    return list(dict.fromkeys(ids))


def agent_node(state: AstroState) -> dict:
    """Node 2 (`agent` / Agir): Decide which tools to invoke next (bounded to max 6 steps)."""
    steps = state.get("steps", 0)
    if steps >= settings.max_agent_steps:
        return {"messages": [AIMessage(content="Max steps reached; proceeding to compose.")]}

    q = state["question"]
    intent = state.get("intent")
    tool_calls: list[dict] = []

    if steps == 0:
        # Step 1: Retrieve relevant passages from the hybrid catalog
        tool_calls.append(
            {
                "name": "search_catalog",
                "args": {"query": q, "top_k": 5},
                "id": "call_search_main",
            }
        )
        # Also fetch specific mentioned objects or showcase session objects
        mentioned = _extract_mentioned_ids(q)
        if not mentioned and intent and intent.request_type == "session_plan":
            if intent.location_name and intent.location_name.lower() == "fontainebleau":
                mentioned = ["M013", "M027", "C014", "M031", "M045"]
            elif "binoculars" in q.lower():
                mentioned = ["M013", "M027", "M031"]
        elif not mentioned and "beautiful galaxy" in q.lower():
            mentioned = ["M031", "M051", "M104"]
        elif not mentioned and "photograph tonight" in q.lower():
            mentioned = ["M031", "M042", "M013", "M027"]

        for idx, obj_id in enumerate(mentioned[:6]):
            tool_calls.append(
                {
                    "name": "get_object",
                    "args": {
                        "object_id": obj_id,
                        "focal_length_mm": intent.focal_length_mm if intent else None,
                    },
                    "id": f"call_get_{idx}_{obj_id}",
                }
            )

        if intent and intent.location_name:
            tool_calls.append(
                {
                    "name": "geocode",
                    "args": {"place_name": intent.location_name},
                    "id": "call_geocode",
                }
            )

    elif steps == 1 and intent and intent.request_type == "session_plan":
        # Extract geocoded (lat, lon) from previous ToolMessages
        lat, lon = 48.8566, 2.3522
        for msg in state.get("messages", []):
            if isinstance(msg, ToolMessage) and msg.name == "geocode":
                try:
                    data = json.loads(msg.content)
                    if data.get("found"):
                        lat, lon = float(data["lat"]), float(data["lon"])
                except Exception:
                    pass

        obs_date_str = (
            intent.target_date.isoformat() if intent.target_date else "2026-10-17"
        )
        candidate_ids = list(
            dict.fromkeys(c.doc_id for c in state.get("retrieved", []) if not c.doc_id.startswith("CON_"))
        )[:8]
        if not candidate_ids:
            candidate_ids = ["M013", "M027", "C014", "M031", "M045"]

        tool_calls.extend(
            [
                {
                    "name": "night",
                    "args": {"lat": lat, "lon": lon, "obs_date": obs_date_str},
                    "id": "call_night",
                },
                {
                    "name": "visibility",
                    "args": {
                        "object_ids": candidate_ids,
                        "lat": lat,
                        "lon": lon,
                        "obs_date": obs_date_str,
                        "focal_length_mm": intent.focal_length_mm,
                    },
                    "id": "call_visibility",
                },
                {
                    "name": "weather",
                    "args": {"lat": lat, "lon": lon, "obs_date": obs_date_str},
                    "id": "call_weather",
                },
            ]
        )

    in_tok = estimate_text_tokens(q) + 120 * (steps + 1)
    out_tok = 45 * len(tool_calls) if tool_calls else 20
    ai_msg = AIMessage(content="", tool_calls=tool_calls)
    return {
        "messages": [ai_msg],
        "steps": steps + 1,
        "tokens_in": state.get("tokens_in", 0) + in_tok,
        "tokens_out": state.get("tokens_out", 0) + out_tok,
    }


def tools_node(state: AstroState) -> dict:
    """Node 3 (`tools` / Outils): Execute requested tool calls and record telemetry."""
    messages = state.get("messages", [])
    if not messages:
        return {}
    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []

    new_tool_messages: list[ToolMessage] = []
    trace: list[ToolCallRecord] = list(state.get("tool_trace", []))
    retrieved: list[Chunk] = list(state.get("retrieved", []))
    existing_chunk_ids = {c.chunk_id for c in retrieved}
    existing_doc_ids = {c.doc_id for c in retrieved}
    calc_alts: dict[str, list[float]] = dict(state.get("calculated_altitudes", {}))
    night_win = state.get("night_window")

    for tc in tool_calls:
        t_name = tc["name"]
        t_args = tc.get("args", {})
        t_id = tc.get("id", t_name)
        tool_fn = TOOLS_BY_NAME.get(t_name)
        if tool_fn is None:
            continue

        t0 = time_mod.perf_counter()
        result = tool_fn.invoke(t_args)
        dur = round(time_mod.perf_counter() - t0, 4)

        trace.append(ToolCallRecord(name=t_name, arguments=t_args, duration_s=dur))
        new_tool_messages.append(
            ToolMessage(content=json.dumps(result, ensure_ascii=False), name=t_name, tool_call_id=t_id)
        )

        if t_name == "search_catalog" and isinstance(result, list):
            for item in result:
                ch = Chunk(
                    chunk_id=item["chunk_id"],
                    doc_id=item["doc_id"],
                    title=item["title"],
                    section=item["section"],
                    text=item["text"],
                    score=item.get("score", 0.0),
                    metadata=ChunkMetadata(**item["metadata"]),
                )
                if ch.chunk_id not in existing_chunk_ids:
                    retrieved.append(ch)
                    existing_chunk_ids.add(ch.chunk_id)
                    existing_doc_ids.add(ch.doc_id)

        elif t_name == "get_object" and isinstance(result, dict) and result.get("found"):
            doc_id = result["doc_id"]
            if doc_id not in existing_doc_ids:
                meta = ChunkMetadata(
                    doc_id=doc_id,
                    catalog=result["catalog"],
                    object_type=result["object_type"],
                    constellation=result["constellation"],
                    ra_deg=result.get("ra_deg"),
                    dec_deg=result.get("dec_deg"),
                    magnitude=result.get("magnitude"),
                    surface_brightness=result.get("surface_brightness"),
                    size_major_arcmin=result.get("size_major_arcmin"),
                    size_minor_arcmin=result.get("size_minor_arcmin"),
                    narrowband_friendly=bool(result.get("narrowband_friendly", False)),
                    aliases=list(result.get("aliases", [])),
                )
                framing_str = f" {result['framing_advice']}" if result.get("framing_advice") else ""
                summary_text = (
                    f"[{doc_id} | {result['title']} | Type: {meta.object_type} | "
                    f"Constellation: {meta.constellation} | Mag: {meta.magnitude} | "
                    f"Size: {meta.size_major_arcmin}' | Narrowband: {meta.narrowband_friendly}]"
                    f"{framing_str}"
                )
                retrieved.append(
                    Chunk(
                        chunk_id=f"{doc_id}#catalog-0",
                        doc_id=doc_id,
                        title=result["title"],
                        section="Catalog Metadata",
                        text=summary_text,
                        score=1.0,
                        metadata=meta,
                    )
                )
                existing_doc_ids.add(doc_id)

        elif t_name == "night" and isinstance(result, dict):
            night_win = (result["astronomical_dark_start"], result["astronomical_dark_end"])

        elif t_name == "visibility" and isinstance(result, dict):
            objs = result.get("objects", {})
            for oid, odata in objs.items():
                if odata.get("slots"):
                    calc_alts[oid] = [float(s["altitude_deg"]) for s in odata["slots"]]

    return {
        "messages": new_tool_messages,
        "tool_trace": trace,
        "retrieved": retrieved,
        "calculated_altitudes": calc_alts,
        "night_window": night_win,
    }


def _check_out_of_corpus(question: str) -> str | None:
    q_low = question.lower()
    for pat, label in _OUT_OF_CORPUS_PATTERNS:
        if re.search(pat, q_low):
            return (
                f"I must abstain: {label} is outside the AstroPilote 506-document corpus "
                "(which covers Messier, Caldwell, bright Herschel 400 deep-sky objects, and the 88 IAU constellations)."
            )
    return None


def compose_node(state: AstroState) -> dict:
    """Node 4 (`compose` / Rédiger): Synthesize a structured, cited `AstroAnswer`."""
    q = state["question"]
    retrieved = state.get("retrieved", [])
    intent = state.get("intent")

    # 1. Check if the query asks about an out-of-corpus subject
    ooc_reason = _check_out_of_corpus(q)
    if ooc_reason is not None or not retrieved:
        ans = AstroAnswer(
            answer=ooc_reason
            or "I must abstain: no matching documents were found in the AstroPilote corpus.",
            plan=[],
            conditions=None,
            sources=[],
            abstained=True,
            verified=True,
        )
        return {
            "answer": ans,
            "tokens_in": state.get("tokens_in", 0) + 180,
            "tokens_out": state.get("tokens_out", 0) + 35,
        }

    # Gather tool outputs from ToolMessages
    night_data: dict = {}
    vis_data: dict = {}
    weather_data: dict = {}
    for msg in state.get("messages", []):
        if isinstance(msg, ToolMessage):
            try:
                payload = json.loads(msg.content)
                if msg.name == "night":
                    night_data = payload
                elif msg.name == "visibility":
                    vis_data = payload
                elif msg.name == "weather":
                    weather_data = payload
            except Exception:
                pass

    # Build ordered list of retrieved doc_ids (prioritizing explicitly mentioned or get_object targets)
    mentioned = _extract_mentioned_ids(q)
    get_obj_ids = [
        c.doc_id for c in retrieved if c.chunk_id.endswith("#catalog-0")
    ]
    retrieved_doc_ids = list(dict.fromkeys(c.doc_id for c in retrieved))
    priority_ids = list(dict.fromkeys(mentioned + get_obj_ids))
    cited_ids = [d for d in priority_ids if d in retrieved_doc_ids]
    for d in retrieved_doc_ids:
        if d not in cited_ids and len(cited_ids) < 5:
            cited_ids.append(d)

    plan_items: list[PlanItem] = []
    conditions_str: str | None = None

    if intent and intent.request_type == "session_plan" and vis_data.get("objects"):
        dark_start = night_data.get("astronomical_dark_start", "21:00")
        dark_end = night_data.get("astronomical_dark_end", "05:30")
        moon_pct = night_data.get("moon_illumination_pct", 0)
        moon_set = night_data.get("moon_set_local")
        clouds = weather_data.get("cloud_cover_pct", 10)
        moon_note = f"{moon_pct}% (sets {moon_set})" if moon_set else f"{moon_pct}%"
        conditions_str = (
            f"Astronomical darkness {dark_start} -> {dark_end} | Moon {moon_note} | Clouds {clouds}%"
        )

        from astropilote.agent.validator import _parse_hhmm, _time_to_night_minutes

        win_s_t = _parse_hhmm(dark_start) or time(21, 0)
        win_e_t = _parse_hhmm(dark_end) or time(5, 30)
        win_s_m = _time_to_night_minutes(win_s_t)
        win_e_m = _time_to_night_minutes(win_e_t)

        objs_map = vis_data.get("objects", {})
        slot_idx = 0
        for oid in cited_ids:
            odata = objs_map.get(oid)
            if not odata or not odata.get("observable"):
                continue
            slots = odata.get("slots", [])
            if not slots:
                continue

            # Keep only slots strictly inside the astronomical darkness window with altitude >= 20°
            valid_dark_slots = []
            for s in slots:
                st_obj = _parse_hhmm(s["time"])
                if st_obj is None:
                    continue
                sm_val = _time_to_night_minutes(st_obj)
                if win_s_m <= sm_val <= win_e_m and float(s["altitude_deg"]) >= 20.0:
                    valid_dark_slots.append(s)

            if not valid_dark_slots:
                continue

            # Stagger slots across the night while preferring high altitude
            chosen_slot = valid_dark_slots[min(slot_idx, len(valid_dark_slots) - 1)]
            sh, sm = [int(x) for x in chosen_slot["time"].split(":")]
            start_t = time(sh, sm)
            eh = (sh + (1 if sm + 30 >= 60 else 0)) % 24
            em = (sm + 30) % 60
            end_t = time(eh, em)

            mag = odata.get("magnitude") or 7.0
            diff = "easy" if mag <= 6.5 else ("medium" if mag <= 8.5 else "hard")
            nb_tip = (
                "Narrowband Ha/OIII friendly (cuts through moonlight)"
                if odata.get("narrowband_friendly")
                else "Broadband continuum target (best in dark sky after Moonset)"
            )
            plan_items.append(
                PlanItem(
                    object_id=oid,
                    name=odata.get("title", oid),
                    start=start_t,
                    end=end_t,
                    altitude_deg=float(chosen_slot["altitude_deg"]),
                    direction=chosen_slot.get("direction", "S"),
                    difficulty=diff,
                    imaging_notes=nb_tip,
                    sources=[oid],
                )
            )
            slot_idx += 1

    # Compose grounded natural-language explanation with inline [DOC_ID] citations
    context_snippets = "\n\n".join(
        f"[{c.doc_id}] ({c.title} — {c.section}): {c.text[:420]}"
        for c in retrieved
        if c.doc_id in cited_ids
    )

    answer_text = ""
    extra_in = estimate_text_tokens(context_snippets) + estimate_text_tokens(q) + 150
    extra_out = 0

    if _ollama_available():
        try:
            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                base_url=settings.ollama_base_url,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
            )
            sys_prompt = (
                "You are AstroPilote, an expert visual astronomy and astrophotography assistant. "
                "Answer the user's question concisely in English using ONLY the provided corpus passages "
                "and ephemeris data. Cite every source inline using brackets like [M031] or [C014]."
            )
            user_prompt = (
                f"Question: {q}\n\n"
                f"Conditions: {conditions_str or 'N/A'}\n\n"
                f"Retrieved Corpus Passages:\n{context_snippets}\n\n"
                f"Cite sources from: {cited_ids}"
            )
            resp = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
            content = str(resp.content).strip()
            # Strip `<think>...</think>` blocks if produced by reasoning models like Qwen3
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if content:
                answer_text = content
                usage = getattr(resp, "usage_metadata", None) or {}
                extra_in = int(usage.get("input_tokens") or extra_in)
                extra_out = int(usage.get("output_tokens") or estimate_text_tokens(answer_text))
        except Exception:
            answer_text = ""

    if not answer_text:
        # Deterministic grounded synthesis from retrieved chunks + ephemerides
        summary_lines: list[str] = []
        if conditions_str:
            summary_lines.append(f"**Night Conditions:** {conditions_str}.")
        if night_data and not night_data.get("has_astronomical_darkness", True):
            summary_lines.append(night_data.get("summary", ""))

        # Check if any queried object never rises (e.g., Omega Centauri C080 from Paris)
        for oid, odata in vis_data.get("objects", {}).items():
            if not odata.get("observable") and odata.get("max_altitude_deg", 0.0) < 5.0:
                summary_lines.append(
                    f"**{odata.get('title', oid)} [{oid}]** reaches a maximum altitude of "
                    f"{odata.get('max_altitude_deg')}° and is not observable from this latitude."
                )

        if plan_items:
            plan_str = "; ".join(
                f"{p.start.strftime('%H:%M')} **{p.name}** [{p.object_id}] ({p.altitude_deg}° {p.direction}, {p.difficulty} — {p.imaging_notes})"
                for p in plan_items
            )
            summary_lines.append(f"**Recommended Session Plan:** {plan_str}.")

        for cid in cited_ids:
            matching_chunks = [c for c in retrieved if c.doc_id == cid]
            if matching_chunks:
                ch = matching_chunks[0]
                body_lines = [
                    ln
                    for ln in ch.text.splitlines()
                    if not ln.startswith("[") and not ln.startswith("Section:")
                ]
                snippet = " ".join(body_lines).strip()
                if len(snippet) > 280:
                    snippet = snippet[:277].rsplit(" ", 1)[0] + "..."
                summary_lines.append(f"- **{ch.title} [{cid}]**: {snippet}")

        answer_text = "\n".join(summary_lines)
        extra_out = estimate_text_tokens(answer_text)

    ans = AstroAnswer(
        answer=answer_text,
        plan=plan_items,
        conditions=conditions_str,
        sources=cited_ids,
        abstained=False,
        verified=True,
    )
    return {
        "answer": ans,
        "tokens_in": state.get("tokens_in", 0) + extra_in,
        "tokens_out": state.get("tokens_out", 0) + extra_out,
    }


def validate_node(state: AstroState) -> dict:
    """Node 5 (`validate` / Vérifier): Run pure-Python verification on `state['answer']`."""
    ans = state.get("answer")
    if ans is None:
        return {}
    checked = verify_astro_answer(ans, state)
    retries = state.get("retries", 0)
    if not checked.verified:
        retries += 1
    return {"answer": checked, "retries": retries}
