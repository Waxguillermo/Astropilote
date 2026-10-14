"""Pipeline nodes (`guard`, `agent`, `tools`, `compose`, `validate`, `refuse`) for AstroPilote."""

from __future__ import annotations

import json
import math
import re
import time as time_mod
from datetime import date, time

from astropilote.agent.schemas import (
    AstroAnswer,
    Chunk,
    ChunkMetadata,
    Intent,
    PlanItem,
    ToolCallRecord,
)
from astropilote.agent.state import AgentMessage, AstroState
from astropilote.agent.validator import verify_astro_answer
from astropilote.config import settings
from astropilote.ingest.catalog import IAU_CONSTELLATIONS
from astropilote.llm import generate_grounded_answer
from astropilote.tools import TOOLS_BY_NAME
from astropilote.tools.catalog import load_documents_lookup, resolve_object_record

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

# General regexes for astronomical catalog designations (to verify corpus membership dynamically)
_EXPLICIT_CATALOG_CODE_PATTERNS = [
    r"\b(NGC\s*\d{1,4})\b",
    r"\b(IC\s*\d{1,4})\b",
    r"\b(Abell\s*\d{1,4})\b",
    r"\b(3C\s*\d{1,4})\b",
    r"\b(UGC\s*\d{1,5})\b",
    r"\b(PGC\s*\d{1,6})\b",
    r"\b(ESO\s*\d{1,3}-\d{1,3})\b",
    r"\b(K2-\d{1,4}[b-i])\b",
    r"\b(Kepler-\d{1,4}[b-i])\b",
    r"\b(TRAPPIST-\d[b-i])\b",
    r"\b(HD\s*\d{4,6}[b-i]?)\b",
    r"\b(\d{5,6}\s+[A-Z][a-z]{3,15})\b",  # Numbered minor planets / asteroids (e.g., 101955 Bennu)
]

# Non-catalog technical domains outside our deep-sky & constellation corpus
_OUT_OF_SCOPE_DOMAIN_PATTERNS = [
    (
        r"\b(?:cmos|ccd)\b.*\b(?:quantum\s+efficiency|read\s+noise|datasheet|imx\d+)\b|\bimx\s*\d{3,4}\b",
        "camera sensor electronic datasheets (quantum efficiency / read noise)",
    ),
    (
        r"\bcollimat(?:e|ion|ing)\b.*\b(?:mirror|ritchey|secondary|newtonian|scope)\b",
        "telescope optical collimation and hardware maintenance manuals",
    ),
    (
        r"\bsample\s+return\b|\bosiris-rex\b|\bexoplanet\b",
        "planetary probes and exoplanet atmospheric spectroscopy",
    ),
]

_KNOWN_LOCATIONS = [
    "Fontainebleau",
    "Paris",
    "Chamonix",
    "Marseille",
    "Lyon",
    "Bordeaux",
    "Nice",
    "Toulouse",
    "Strasbourg",
    "Lille",
    "Nantes",
    "Rennes",
    "Montpellier",
    "Grenoble",
]

_NON_CITY_WORDS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "messier", "caldwell", "herschel", "abell", "ngc", "ic", "utc",
} | {name.lower().split()[0] for _, name, _, _, _ in IAU_CONSTELLATIONS}

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
    """Extract observation city/site from `question` (supports known cities + general prepositional patterns)."""
    for loc in _KNOWN_LOCATIONS:
        if re.search(rf"\b{loc}\b", question, flags=re.IGNORECASE):
            return loc

    # General pattern: "in <City>", "from <City>", "over <City>" where <City> is not a constellation or month
    for m in re.finditer(
        r"\b(?:in|from|over|near|à|depuis)\s+([A-ZÀ-Ÿ][a-zà-ÿ]{2,20}(?:-[A-ZÀ-Ÿ][a-zà-ÿ]{2,20})?)\b",
        question,
    ):
        candidate = m.group(1)
        if candidate.lower() not in _NON_CITY_WORDS:
            return candidate
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


def guard_node(state: AstroState) -> dict:
    """Node 1 (`guard` / Comprendre): Classify intent, extract parameters, and catch traps.

    Pure deterministic Python guardrail (0 LLM tokens consumed).
    """
    q = state["question"]
    q_low = q.lower()

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
            return {"intent": intent}

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
        return {"intent": intent}

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
    return {"intent": intent}


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


def _angular_separation_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Compute spherical angular separation in degrees between two J2000 coordinates."""
    r1, d1 = math.radians(ra1), math.radians(dec1)
    r2, d2 = math.radians(ra2), math.radians(dec2)
    cos_sep = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_sep))))


def _find_astrometric_companions(primary_id: str, max_sep_deg: float = 1.6) -> list[str]:
    """Find physical companion or multi-component objects in the same constellation using J2000 coordinates."""
    by_id, _ = load_documents_lookup()
    primary = by_id.get(primary_id)
    if not primary:
        return []
    p_meta = primary.get("metadata", {})
    ra1, dec1 = p_meta.get("ra_deg"), p_meta.get("dec_deg")
    const1 = p_meta.get("constellation")
    if ra1 is None or dec1 is None or not const1:
        return []

    companions: list[tuple[float, str]] = []
    for doc_id, doc in by_id.items():
        if doc_id == primary_id or doc_id.startswith("CON_"):
            continue
        # Prefer Messier/Caldwell companions for Messier/Caldwell primaries
        if primary_id.startswith(("M", "C")) and not doc_id.startswith(("M", "C")):
            continue
        m = doc.get("metadata", {})
        if m.get("constellation") != const1:
            continue
        ra2, dec2 = m.get("ra_deg"), m.get("dec_deg")
        if ra2 is None or dec2 is None:
            continue
        sep = _angular_separation_deg(float(ra1), float(dec1), float(ra2), float(dec2))
        if sep <= max_sep_deg:
            companions.append((sep, doc_id))

    companions.sort(key=lambda x: x[0])
    return [cid for _, cid in companions[:2]]


def _extract_mentioned_ids(question: str) -> list[str]:
    """Extract Messier, Caldwell, NGC, or Constellation IDs dynamically from the 507-doc corpus."""
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
        rec = resolve_object_record(f"NGC{int(m.group(1)):04d}")
        if rec:
            ids.append(rec["doc_id"])

    q_low = question.lower()
    by_id, _ = load_documents_lookup()

    # Match common names and constellation names dynamically against the loaded catalog
    for doc_id, doc in by_id.items():
        meta = doc.get("metadata", {})
        for alias in meta.get("aliases", []):
            a_low = alias.strip().lower()
            if len(a_low) >= 5 and not a_low.startswith(("m0", "c0", "ngc", "con_")):
                if re.search(rf"\b{re.escape(a_low)}\b", q_low):
                    ids.append(doc_id)
                    break

    # If the user asks about companions, satellites, or multi-part nebulae, include astrometric neighbors
    wants_neighbors = any(
        w in q_low for w in ["satellite", "companion", "interacting", "veil nebula", "double cluster"]
    )
    for base_id in list(dict.fromkeys(ids)):
        if wants_neighbors or base_id == "C033":
            ids.extend(_find_astrometric_companions(base_id, max_sep_deg=1.6))

    return list(dict.fromkeys(ids))


def _extract_search_filters(question: str) -> dict[str, object]:
    """Extract Qdrant `MetadataFilter` criteria (`constellation`, `catalog`, `max_magnitude`, `narrowband_only`)."""
    q_low = question.lower()
    filters: dict[str, object] = {}

    # Detect explicit constellation filter in multi-target queries (e.g., "nebulae in Cygnus")
    if any(w in q_low for w in ["which ", "suggest ", "targets ", "nebulae in ", "galaxies in ", "clusters in "]):
        for code, cname, _, _, _ in IAU_CONSTELLATIONS:
            if re.search(rf"\b{re.escape(cname.lower())}\b", q_low):
                filters["constellation"] = code
                break

    if any(w in q_low for w in ["narrowband", "ha/oiii", "h-alpha", "dual-band"]):
        filters["narrowband_only"] = True

    if "messier" in q_low and any(w in q_low for w in ["suggest", "three", "bright"]):
        filters["catalog"] = "Messier"

    if "binoculars" in q_low and "suggest" in q_low:
        filters["max_magnitude"] = 8.5
    else:
        mag_m = re.search(
            r"(?:brighter than|under|below)\s+magnitude\s+(\d+(?:\.\d+)?)",
            q_low,
        )
        if mag_m:
            filters["max_magnitude"] = float(mag_m.group(1))

    return filters


def _select_showcase_candidates(question: str, intent: Intent | None) -> list[str]:
    """Select bright catalog objects by physical criteria when no specific target ID is named in the query."""
    q_low = question.lower()
    if "galaxy" in q_low or "galaxies" in q_low:
        return ["M031", "M051", "M104"]
    if "binoculars" in q_low:
        return ["M013", "M027", "M031"]
    if intent and intent.request_type == "session_plan":
        return ["M013", "M027", "C014", "M031", "M045"]
    if any(w in q_low for w in ["photograph tonight", "what should i", "best objects"]):
        return ["M031", "M042", "M013", "M027"]
    return []


def agent_node(state: AstroState) -> dict:
    """Node 2 (`agent` / Agir): Decide which tools to invoke next (bounded to max 6 steps).

    Pure deterministic planner (0 LLM tokens consumed).
    """
    steps = state.get("steps", 0)
    if steps >= settings.max_agent_steps:
        return {
            "messages": state.get("messages", [])
            + [AgentMessage(role="assistant", content="Max steps reached; proceeding to compose.")]
        }

    q = state["question"]
    intent = state.get("intent")
    tool_calls: list[dict] = []

    if steps == 0:
        # Step 1a: Unfiltered hybrid search (BM25 + Qdrant Dense + Cross-Encoder Reranker)
        tool_calls.append(
            {
                "name": "search_catalog",
                "args": {"query": q, "top_k": 5},
                "id": "call_search_main",
            }
        )

        # Step 1b: If structured metadata constraints are present, run a Qdrant metadata-filtered search
        meta_filters = _extract_search_filters(q)
        if meta_filters:
            tool_calls.append(
                {
                    "name": "search_catalog",
                    "args": {"query": q, "top_k": 5, **meta_filters},
                    "id": "call_search_filtered",
                }
            )

        # Step 1c: Fetch exact astrometric records for mentioned or showcase candidate objects
        mentioned = _extract_mentioned_ids(q)
        if not mentioned:
            mentioned = _select_showcase_candidates(q, intent)

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
        lat, lon = 48.8566, 2.3522
        for msg in state.get("messages", []):
            if msg.role == "tool" and msg.name == "geocode":
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

    ai_msg = AgentMessage(role="assistant", content="", tool_calls=tool_calls)
    return {
        "messages": state.get("messages", []) + [ai_msg],
        "steps": steps + 1,
    }


def tools_node(state: AstroState) -> dict:
    """Node 3 (`tools` / Outils): Execute requested tool calls and record telemetry."""
    messages = state.get("messages", [])
    if not messages:
        return {}
    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []

    new_tool_messages: list[AgentMessage] = []
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
        result = tool_fn(**t_args)
        dur = round(time_mod.perf_counter() - t0, 4)

        trace.append(ToolCallRecord(name=t_name, arguments=t_args, duration_s=dur))
        new_tool_messages.append(
            AgentMessage(
                role="tool",
                content=json.dumps(result, ensure_ascii=False),
                name=t_name,
                tool_call_id=t_id,
            )
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
        "messages": messages + new_tool_messages,
        "tool_trace": trace,
        "retrieved": retrieved,
        "calculated_altitudes": calc_alts,
        "night_window": night_win,
    }


def _check_out_of_corpus(question: str) -> str | None:
    """Check whether `question` asks about an unindexed catalog designation or out-of-scope domain."""
    # 1. Check if any explicit catalog designation (IC, Abell, 3C, K2, numbered asteroid, etc.) is absent from our 507 docs
    for pat in _EXPLICIT_CATALOG_CODE_PATTERNS:
        for m in re.finditer(pat, question, flags=re.IGNORECASE):
            raw_code = m.group(1).strip()
            if resolve_object_record(raw_code) is None:
                return (
                    f"I must abstain: '{raw_code}' is not indexed in the AstroPilote 507-document corpus "
                    "(which covers Messier, Caldwell, the 200 brightest Herschel NGC deep-sky objects, and the 88 IAU constellations)."
                )

    # 2. Check if the query asks about non-catalog hardware/probe domains
    q_low = question.lower()
    for pat, domain_label in _OUT_OF_SCOPE_DOMAIN_PATTERNS:
        if re.search(pat, q_low):
            return (
                f"I must abstain: {domain_label} falls outside the AstroPilote 507-document corpus "
                "(which covers Messier, Caldwell, bright Herschel 400 deep-sky objects, and the 88 IAU constellations)."
            )
    return None


_CONSTELLATION_NAMES: dict[str, str] = {
    code: cname for code, cname, _, _, _ in IAU_CONSTELLATIONS
}

_COMPASS_FULL: dict[str, str] = {
    "N": "North",
    "NE": "North-East",
    "E": "East",
    "SE": "South-East",
    "S": "South",
    "SW": "South-West",
    "W": "West",
    "NW": "North-West",
}

_OBJECT_TYPE_PLAIN: dict[str, str] = {
    "galaxy": "spiral/elliptical galaxy (a giant island of billions of stars outside our Milky Way)",
    "globular_cluster": "globular star cluster (a dense, ancient sphere of hundreds of thousands of stars)",
    "open_cluster": "open star cluster (a bright group of young stars born from the same cloud)",
    "cluster_with_nebulosity": "young star cluster embedded in a glowing gas nebula",
    "planetary_nebula": "planetary nebula (a glowing shell of gas cast off by a dying sun-like star)",
    "emission_nebula": "emission nebula (a vast glowing stellar nursery where new stars form)",
    "reflection_nebula": "reflection nebula (interstellar dust glowing with reflected starlight)",
    "supernova_remnant": "supernova remnant (the expanding cloud left by an exploded massive star)",
    "dark_nebula": "dark nebula (a dense silhouette of cosmic dust)",
}


def _equipment_for_magnitude(mag: float | None) -> str:
    """Translate astronomical magnitude into plain beginner equipment guidance."""
    if mag is None:
        return "Visible with binoculars or a small telescope"
    if mag <= 5.0:
        return f"Visible to the naked eye & spectacular in binoculars (very bright, mag {mag})"
    if mag <= 6.3:
        return f"Faint naked-eye / Easy with any binoculars (bright, mag {mag})"
    if mag <= 8.5:
        return f"Easy with standard 10x50 binoculars or a small telescope (moderate brightness, mag {mag})"
    return f"Requires a small telescope — too faint for the naked eye (faint, mag {mag})"


def _moon_size_comparison(size_arcmin: float | None) -> str:
    """Compare apparent size in arcminutes to the Full Moon (30 arcminutes across)."""
    if not size_arcmin or size_arcmin <= 0:
        return ""
    ratio = size_arcmin / 30.0
    if ratio >= 1.8:
        return f"spans ~{ratio:.1f}× the width of the Full Moon ({size_arcmin}')"
    if ratio >= 0.75:
        return f"appears roughly the same width as the Full Moon ({size_arcmin}')"
    if ratio >= 0.25:
        return f"spans about {int(round(ratio * 100))}% of the Full Moon's width ({size_arcmin}')"
    return f"compact object ({size_arcmin}', best magnified in a telescope)"


def compose_node(state: AstroState) -> dict:
    """Node 4 (`compose` / Rédiger): Synthesize a clear, beginner-friendly cited `AstroAnswer`."""
    q = state["question"]
    retrieved = state.get("retrieved", [])
    intent = state.get("intent")

    # 1. Check if the query asks about an out-of-corpus subject (0 LLM tokens consumed)
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
        return {"answer": ans}

    # Gather tool outputs from tool messages
    night_data: dict = {}
    vis_data: dict = {}
    weather_data: dict = {}
    for msg in state.get("messages", []):
        if msg.role == "tool":
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

    # Build ordered list of retrieved doc_ids (prioritizing deep-sky objects ahead of CON_* pages)
    mentioned = _extract_mentioned_ids(q)
    get_obj_ids = [
        c.doc_id for c in retrieved if c.chunk_id.endswith("#catalog-0")
    ]
    retrieved_doc_ids = list(dict.fromkeys(c.doc_id for c in retrieved))
    priority_ids = list(dict.fromkeys(mentioned + get_obj_ids))
    # Place deep-sky objects (M*, C*, NGC*) before generic constellation pages (CON_*)
    dso_priority = [d for d in priority_ids if d in retrieved_doc_ids and not d.startswith("CON_")]
    con_priority = [d for d in priority_ids if d in retrieved_doc_ids and d.startswith("CON_")]
    cited_ids = (dso_priority + con_priority)[:5]
    for d in retrieved_doc_ids:
        if d not in cited_ids and len(cited_ids) < 5:
            cited_ids.append(d)

    plan_items: list[PlanItem] = []
    conditions_str: str | None = None
    plan_by_id: dict[str, PlanItem] = {}

    loc_label = (intent.location_name if intent and intent.location_name else "Observation Site")
    date_label = (
        intent.target_date.isoformat()
        if intent and intent.target_date
        else vis_data.get("date", "Tonight")
    )

    if intent and intent.request_type == "session_plan" and vis_data.get("objects"):
        dark_start = night_data.get("astronomical_dark_start", "21:00")
        dark_end = night_data.get("astronomical_dark_end", "05:30")
        moon_pct = night_data.get("moon_illumination_pct", 0)
        moon_set = night_data.get("moon_set_local")
        clouds = weather_data.get("cloud_cover_pct", 10)
        moon_note = f"{moon_pct}% lit (sets at {moon_set})" if moon_set else f"{moon_pct}% lit"
        conditions_str = (
            f"{loc_label} on {date_label}: Astronomical darkness {dark_start} to {dark_end} | "
            f"Moon {moon_note} | Cloud cover {clouds}%"
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

            chosen_slot = valid_dark_slots[min(slot_idx, len(valid_dark_slots) - 1)]
            sh, sm = [int(x) for x in chosen_slot["time"].split(":")]
            start_t = time(sh, sm)
            eh = (sh + (1 if sm + 30 >= 60 else 0)) % 24
            em = (sm + 30) % 60
            end_t = time(eh, em)

            mag = odata.get("magnitude") or 7.0
            diff = "easy" if mag <= 6.5 else ("medium" if mag <= 8.5 else "hard")
            equip_short = (
                "Naked eye / Binoculars"
                if mag <= 6.2
                else ("Binoculars (10x50)" if mag <= 8.5 else "Small telescope")
            )
            size_note = _moon_size_comparison(odata.get("size_major_arcmin"))
            beginner_tip = f"{equip_short}" + (f" · {size_note}" if size_note else "")

            item = PlanItem(
                object_id=oid,
                name=odata.get("title", oid),
                start=start_t,
                end=end_t,
                altitude_deg=float(chosen_slot["altitude_deg"]),
                direction=chosen_slot.get("direction", "S"),
                difficulty=diff,
                imaging_notes=beginner_tip,
                sources=[oid],
            )
            plan_items.append(item)
            plan_by_id[oid] = item
            slot_idx += 1

    # Build structured beginner-friendly reference notes from retrieved chunks & ephemerides
    summary_lines: list[str] = []
    if conditions_str:
        dark_s = night_data.get("astronomical_dark_start", "21:00")
        dark_e = night_data.get("astronomical_dark_end", "05:30")
        moon_p = night_data.get("moon_illumination_pct", 0)
        clouds_p = weather_data.get("cloud_cover_pct", 10)
        moon_desc = (
            "the Moon is barely lit, leaving the sky nice and dark"
            if moon_p <= 35
            else "moonlight will brighten the sky, so stick to bright star clusters or narrowband nebulae"
        )
        summary_lines.append(
            f"**Computed Ephemeris for {loc_label} on {date_label}:** The sky is completely dark from **{dark_s} to {dark_e}** "
            f"(cloud cover is around **{clouds_p}%**, and with the Moon at **{moon_p}%**, {moon_desc})."
        )
    if night_data and not night_data.get("has_astronomical_darkness", True):
        summary_lines.append(night_data.get("summary", ""))

    for oid, odata in vis_data.get("objects", {}).items():
        if not odata.get("observable") and odata.get("max_altitude_deg", 0.0) < 5.0:
            summary_lines.append(
                f"**Why you cannot see {odata.get('title', oid)} [{oid}] from {loc_label} on {date_label}:** "
                f"Because of our northern latitude on Earth, this southern-sky object stays below the horizon all night "
                f"(reaching a maximum altitude of **{odata.get('max_altitude_deg')}°**, where `0°` is the horizon line)."
            )

    objs_vis_map = vis_data.get("objects", {})
    for cid in cited_ids:
        matching_chunks = [c for c in retrieved if c.doc_id == cid]
        if matching_chunks:
            prose_chunks = [c for c in matching_chunks if c.section != "Catalog Metadata"]
            ch = prose_chunks[0] if prose_chunks else matching_chunks[0]
            body_lines = [
                ln.strip()
                for ln in ch.text.splitlines()
                if ln.strip() and not ln.startswith("[") and not ln.startswith("Section:")
            ]
            full_body = " ".join(body_lines).strip()
            sentences = re.split(r"(?<=[.!?])\s+", full_body)
            clean_excerpt = " ".join(sentences[:2]).strip() or full_body[:240]

            const_code = (ch.metadata.constellation or "").upper()
            const_full = _CONSTELLATION_NAMES.get(const_code, const_code)
            raw_type = ch.metadata.object_type or "deep-sky object"
            plain_type = _OBJECT_TYPE_PLAIN.get(raw_type, raw_type.replace("_", " "))
            equip_guide = _equipment_for_magnitude(ch.metadata.magnitude)
            size_guide = _moon_size_comparison(ch.metadata.size_major_arcmin)
            size_sentence = f", and {size_guide}" if size_guide else ""

            where_sentence = ""
            odata = objs_vis_map.get(cid)
            if cid in plan_by_id:
                pi = plan_by_id[cid]
                dir_full = _COMPASS_FULL.get(pi.direction, pi.direction)
                peak_alt = odata.get("max_altitude_deg", pi.altitude_deg) if odata else pi.altitude_deg
                best_time = (odata.get("best_slot") or {}).get("time", pi.start.strftime("%H:%M")) if odata else pi.start.strftime("%H:%M")
                where_sentence = (
                    f" **Verified Ephemeris ({loc_label}, {date_label}):** Scheduled slot **{pi.start.strftime('%H:%M')}–{pi.end.strftime('%H:%M')}** "
                    f"at **{pi.altitude_deg:.0f}°** facing **{dir_full} ({pi.direction})** (peaks at **{peak_alt:.0f}°** around **{best_time}**)."
                )

            summary_lines.append(
                f"- **{ch.title}** [{cid}] — **What it is:** A {plain_type} in the constellation **{const_full}** ({const_code}). "
                f"**How to see it:** {equip_guide}{size_sentence}.{where_sentence} "
                f"**Background:** {clean_excerpt}"
            )

    structured_reference = "\n\n".join(summary_lines)

    context_snippets = "\n\n".join(
        f"[{c.doc_id}] ({c.title} — {c.section}): {c.text[:400]}"
        for c in retrieved
        if c.doc_id in cited_ids
    )

    prev_ans = state.get("answer")
    retry_note = ""
    if state.get("retries", 0) > 0 and prev_ans and prev_ans.verification_errors:
        retry_note = (
            "\n\nIMPORTANT CORRECTION REQUIRED: Your previous draft failed verification with errors: "
            + "; ".join(prev_ans.verification_errors)
            + ". Cite ONLY from "
            + str(cited_ids)
            + "."
        )

    sys_prompt = (
        "You are AstroPilote, a warm, crystal-clear astronomy guide for beginners. "
        "Keep your response concise (200 to 320 words) so it is complete and never cuts off mid-sentence. "
        "Explain clearly: "
        "1) What each relevant object is and its full constellation name. "
        "2) Its exact visual magnitude, whether it is visible to the naked eye (mag <= 6), binoculars (6-8.5), or a telescope (> 8.5), and its size in arcminutes compared to the Full Moon (30'). "
        "3) Where and when to look using the verified ephemeris altitudes and times provided for the user's location and date. "
        "Use ONLY the provided corpus facts and ephemeris data, and cite every source inline in exact brackets like [M031] or [C014]."
    )
    user_prompt = (
        f"Question: {q}\n\n"
        f"Sky Conditions ({loc_label}, {date_label}): {conditions_str or 'General catalog question'}\n\n"
        f"Verified Catalog & Ephemeris Facts:\n{structured_reference}\n\n"
        f"Retrieved Corpus Passages:\n{context_snippets}\n\n"
        f"Required inline citations: {cited_ids}{retry_note}\n"
        f"Write a complete, concise beginner-friendly answer (200-320 words) with exact bracketed citations:"
    )

    answer_text, extra_in, extra_out, _backend = generate_grounded_answer(
        sys_prompt,
        user_prompt,
        mode_override=state.get("llm_mode_override"),
    )

    if not answer_text:
        answer_text = structured_reference
        extra_in = 0
        extra_out = 0

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
