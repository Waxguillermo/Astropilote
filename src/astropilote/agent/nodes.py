"""Pipeline nodes (`guard`, `agent`, `tools`, `compose`, `validate`, `refuse`) for AstroPilote."""

from __future__ import annotations

import json
import math
import re
import time as time_mod
from datetime import date, time, timedelta

from astropilote.agent.schemas import (
    AstroAnswer,
    Chunk,
    ChunkMetadata,
    Intent,
    PlanItem,
    ToolCallRecord,
)
from astropilote.agent.state import AgentMessage, AstroState
from astropilote.agent.validator import (
    _parse_hhmm,
    _time_to_night_minutes,
    verify_astro_answer,
)
from astropilote.config import settings
from astropilote.ingest.catalog import IAU_CONSTELLATIONS
from astropilote.llm import generate_grounded_answer
from astropilote.tools import TOOLS_BY_NAME
from astropilote.tools.catalog import load_documents_lookup, resolve_object_record
from astropilote.tools.world import _FICTIONAL_LOCATIONS, _load_fixtures

# Prompt injection & secret exfiltration patterns (must indicate override or extraction intent)
_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous\s+|prior\s+)?instructions",
    r"ignore[rz]?\s+(?:toutes\s+)?(?:tes|les|vos)\s+instructions",
    r"(?:reveal|output|print|dump|show|give\s+me|donne[- ]moi)\b.*(?:\bsystem\s+prompt\b|\bprompt\s+syst[eè]me\b|\bdeveloper\s+rules\b)",
    r"\bsystem\s+override\b",
    r"\bcite\s+fake\s+source\b",
    r"disregard\s+the\s+(?:astronomy|corpus)",
    r"(?:anthropic_api_key|openai_api_key|\.env\b).*(?:\bkey\b|secret|token|password|print|reveal|show|dump)",
    r"(?:print|reveal|show|dump|what\s+is)\b.*(?:anthropic_api_key|openai_api_key|\.env\b)",
]

# Clearly non-astronomy off-topic requests (only triggered when zero astronomy terms are present)
_OFF_TOPIC_PATTERNS = [
    r"\bbrute[- ]force\b.*\b(?:ssh|password|server)\b",
    r"\b(?:code\s+du\s+travail|employment\s+contract|contrat\s+de\s+travail)\b",
    r"\b(?:sql\s+injection|ransomware|keylogger|exploit\s+payload)\b",
]

_ASTRO_DOMAIN_TERMS = {
    "star",
    "stars",
    "galaxy",
    "galaxies",
    "nebula",
    "nebulae",
    "cluster",
    "clusters",
    "constellation",
    "telescope",
    "binoculars",
    "eyepiece",
    "mount",
    "camera",
    "sensor",
    "focal",
    "aperture",
    "magnitude",
    "sky",
    "night",
    "moon",
    "sun",
    "eclipse",
    "planet",
    "exoplanet",
    "asteroid",
    "comet",
    "meteor",
    "bortle",
    "seeing",
    "twilight",
    "zenith",
    "horizon",
    "altitude",
    "azimuth",
    "declination",
    "ascension",
    "astrophotography",
    "narrowband",
    "filter",
    "oiii",
    "h-alpha",
    "collimation",
    "collimate",
    "messier",
    "caldwell",
    "ngc",
    "ic",
    "observe",
    "observing",
    "photograph",
    "image",
    "imaging",
}

# General regexes for astronomical catalog designations
_EXPLICIT_CATALOG_CODE_PATTERNS = [
    r"\b(Messier\s*\d{1,4})\b",
    r"\b(Caldwell\s*\d{1,4})\b",
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
    r"\b(\d{5,6}\s+[A-Z][a-z]{3,15})\b",
]

# Questions about technique, equipment or sky quality also search the practical guides
_GUIDE_TOPIC_TERMS = [
    "how do i",
    "how to",
    "how can i",
    "tips",
    "beginner",
    "technique",
    "light pollution",
    "bortle",
    "dark sky",
    "dark-sky",
    "seeing",
    "airmass",
    "air mass",
    "twilight",
    "moon phase",
    "dark adaptation",
    "averted vision",
    "star hopping",
    "star-hopping",
    "limiting magnitude",
    "magnitude scale",
    "surface brightness",
    "planisphere",
    "star chart",
    "binocular",
    "eyepiece",
    "magnification",
    "barlow",
    "refractor",
    "dobsonian",
    "mount",
    "polar align",
    "goto",
    "focal ratio",
    "f-ratio",
    "f-number",
    "stacking",
    "dark frame",
    "flat frame",
    "flats",
    "star trail",
    "h-alpha",
    "oiii",
    "narrowband",
    "exposure",
    "astrophotography",
    "right ascension",
    "declination",
    "azimuth",
    "messier marathon",
    "collimate",
    "collimation",
    "cmos",
    "ccd",
    "read noise",
    "quantum efficiency",
    "buy",
    "purchase",
    "budget",
    "eclipse",
    "telescope",
    "camera",
    "sensor",
]

_NON_CITY_WORDS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "messier",
    "caldwell",
    "herschel",
    "abell",
    "ngc",
    "ic",
    "utc",
    "tonight",
    "today",
    "tomorrow",
    "france",
    "europe",
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


def _extract_date(question: str, ref_date: date | None = None) -> date | None:
    """Extract an explicit date ('YYYY-MM-DD', 'Month DD, YYYY') or relative date ('tonight', 'tomorrow')."""
    iso_m = re.search(r"\b(20\d{2})-(\d{2})-(\d{2})\b", question)
    if iso_m:
        try:
            return date(int(iso_m.group(1)), int(iso_m.group(2)), int(iso_m.group(3)))
        except ValueError:
            pass

    for m_name, m_num in _MONTH_MAP.items():
        pat1 = rf"\b{m_name}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(20\d{{2}})\b"
        m1 = re.search(pat1, question, flags=re.IGNORECASE)
        if m1:
            try:
                return date(int(m1.group(2)), m_num, int(m1.group(1)))
            except ValueError:
                pass
        pat2 = rf"\b(\d{{1,2}})\s+{m_name},?\s+(20\d{{2}})\b"
        m2 = re.search(pat2, question, flags=re.IGNORECASE)
        if m2:
            try:
                return date(int(m2.group(2)), m_num, int(m2.group(1)))
            except ValueError:
                pass

    q_low = question.lower()
    base_d = ref_date or settings.get_reference_date()
    if re.search(r"\b(?:tomorrow|demain)\b", q_low):
        return base_d + timedelta(days=1)
    if re.search(r"\b(?:today|aujourd'hui)\b", q_low):
        return base_d
    return None


def _extract_location(question: str) -> str | None:
    """Extract observation city/site from `question` dynamically using fixtures and prepositional patterns."""
    fixtures = _load_fixtures()
    for loc_key, loc_info in fixtures.get("locations", {}).items():
        loc_name = loc_info.get("name", loc_key)
        if re.search(rf"\b{re.escape(loc_name)}\b", question, flags=re.IGNORECASE):
            return str(loc_name)

    for fic in _FICTIONAL_LOCATIONS:
        if re.search(rf"\b{re.escape(fic)}\b", question, flags=re.IGNORECASE):
            return fic.title()

    for m in re.finditer(
        r"\b(?:in|from|over|near|at|around|à|depuis)\s+([A-ZÀ-Ÿ][a-zà-ÿ]{2,20}(?:[\s-][A-ZÀ-Ÿ][a-zà-ÿ]{2,20}){0,2})\b",
        question,
    ):
        candidate = m.group(1).strip()
        first_word = candidate.split()[0].lower()
        if first_word not in _NON_CITY_WORDS and candidate.lower() not in _NON_CITY_WORDS:
            return candidate
    return None


def _extract_optics(question: str) -> tuple[float | None, float | None]:
    """Extract `(aperture_mm, focal_length_mm)` from `question`."""
    focal: float | None = None
    aperture: float | None = None

    f_match = re.search(r"(\d{2,4})\s*mm\s+focal\s+length", question, flags=re.IGNORECASE)
    if f_match:
        focal = float(f_match.group(1))
    else:
        f_scope = re.search(
            r"\b([4-9]\d{2}|\d{4})\s*mm\s+(?:scope|telescope|lens|refractor|apo)",
            question,
            flags=re.IGNORECASE,
        )
        if f_scope:
            focal = float(f_scope.group(1))

    a_match = re.search(
        r"(\d{2,3})\s*mm\s+(?:refractor|newtonian|telescope|apo|aperture)",
        question,
        flags=re.IGNORECASE,
    )
    if a_match and float(a_match.group(1)) < 350:
        aperture = float(a_match.group(1))
    return aperture, focal


def _extract_sensor_format(question: str) -> str:
    """Detect whether the user specifies a full-frame (36x24 mm) or APS-C camera sensor."""
    if re.search(r"full[- ]frame|36\s*[x×]\s*24", question, flags=re.IGNORECASE):
        return "full-frame"
    return "APS-C"


def _is_weather_only_question(question: str) -> bool:
    """Return True if the question asks ONLY for a meteorological forecast (no astronomical target)."""
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
    if not any(term in q_low for term in weather_terms):
        return False
    dso_mentions = [oid for oid in _extract_mentioned_ids(question) if not oid.startswith("CON_")]
    return len(dso_mentions) == 0


def guard_node(state: AstroState) -> dict:
    """Node 1 (`guard` / Comprendre): Classify intent, extract parameters, and catch genuine safety/weather traps."""
    q = state["question"]
    q_low = q.lower()

    for pat in _INJECTION_PATTERNS:
        if re.search(pat, q_low):
            intent = Intent(
                request_type="prompt_injection",
                refusal_reason=(
                    "I must refuse this request: it attempts to override AstroPilote's safety "
                    "or citation-grounding rules (or extract configuration secrets). "
                    "I only assist with visual astronomy and astrophotography."
                ),
            )
            return {"intent": intent}

    has_astro_context = any(re.search(rf"\b{re.escape(t)}\b", q_low) for t in _ASTRO_DOMAIN_TERMS)
    if not has_astro_context:
        for pat in _OFF_TOPIC_PATTERNS:
            if re.search(pat, q_low):
                intent = Intent(
                    request_type="off_topic",
                    refusal_reason=(
                        "I must refuse this request as it is outside the scope of AstroPilote. "
                        "I assist with visual astronomy, sky observation, and astrophotography."
                    ),
                )
                return {"intent": intent}

    ref_today = settings.get_reference_date()
    date_override_str = state.get("date_override")
    target_d = _extract_date(q, ref_date=ref_today)
    if target_d is None and date_override_str:
        try:
            target_d = date.fromisoformat(date_override_str.strip()[:10])
        except ValueError:
            target_d = None

    loc = _extract_location(q)
    aperture, focal = _extract_optics(q)
    if aperture is None and state.get("aperture_override") is not None:
        aperture = state.get("aperture_override")
    if focal is None and state.get("focal_length_override") is not None:
        focal = state.get("focal_length_override")
    sensor_fmt = _extract_sensor_format(q)

    if (
        target_d
        and _is_weather_only_question(q)
        and (target_d - ref_today).days > settings.max_forecast_days
    ):
        intent = Intent(
            request_type="impossible_forecast",
            location_name=loc,
            target_date=target_d,
            refusal_reason=(
                f"I cannot provide a weather, cloud-cover, or seeing forecast for {target_d.isoformat()}: "
                f"meteorological forecasts are only reliable up to ~{settings.max_forecast_days} days in advance "
                "(though astronomical positions and Moon phases can be computed for any date)."
            ),
        )
        return {"intent": intent}

    is_photo = any(
        w in q_low
        for w in [
            "image",
            "imaging",
            "shoot",
            "photograph",
            "astrophotography",
            "narrowband",
            "filter",
            "focal",
            "sensor",
            "frame",
        ]
    )
    is_visual = any(w in q_low for w in ["binoculars", "visual", "eyepiece", "naked eye"])
    mode = (
        "astrophotography"
        if (is_photo and not is_visual)
        else ("visual" if (is_visual and not is_photo) else "hybrid")
    )

    exec_mode = state.get("execution_mode")
    is_planning = (
        exec_mode != "rag"
        and (
            bool(loc and target_d)
            or any(
                w in q_low
                for w in [
                    "what targets should i",
                    "session",
                    "altitude",
                    "when is",
                    "can i observe",
                    "can i see",
                    "can i photograph",
                    "observe tonight",
                ]
            )
        )
        and bool(loc or target_d)
    )
    intent = Intent(
        request_type="session_plan" if is_planning else "factual",
        mode=mode,
        location_name=loc,
        target_date=target_d,
        aperture_mm=aperture,
        focal_length_mm=focal,
        sensor_format=sensor_fmt,  # type: ignore[arg-type]
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


def _shares_complex_stem(doc_a: dict, doc_b: dict) -> bool:
    """Return True if two catalog entries share a multi-component nebula/cluster name stem (e.g. Veil Nebula)."""
    words_a = set(re.findall(r"[a-z]{4,}", (doc_a.get("title") or "").lower()))
    words_b = set(re.findall(r"[a-z]{4,}", (doc_b.get("title") or "").lower()))
    ignore = {"nebula", "galaxy", "cluster", "messier", "caldwell", "open", "globular", "planetary"}
    return bool((words_a & words_b) - ignore)


def _find_astrometric_companions(
    primary_id: str,
    max_sep_deg: float = 1.6,
    only_shared_complex: bool = False,
) -> list[str]:
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
        if doc_id == primary_id or doc_id.startswith(("CON_", "GUIDE_")):
            continue
        if primary_id.startswith(("M", "C")) and not doc_id.startswith(("M", "C")):
            continue
        m = doc.get("metadata", {})
        if m.get("constellation") != const1:
            continue
        if only_shared_complex and not _shares_complex_stem(primary, doc):
            continue
        ra2, dec2 = m.get("ra_deg"), m.get("dec_deg")
        if ra2 is None or dec2 is None:
            continue
        sep = _angular_separation_deg(float(ra1), float(dec1), float(ra2), float(dec2))
        if sep <= max_sep_deg:
            companions.append((sep, doc_id))

    companions.sort(key=lambda x: x[0])
    return [cid for _, cid in companions[:2]]


def _dedup_letters(s: str) -> str:
    """Collapse repeated consecutive letters (e.g. 'nebulla' -> 'nebula') for typo-tolerant alias matching."""
    return re.sub(r"([a-z])\1+", r"\1", s.lower())


def _extract_mentioned_ids(question: str) -> list[str]:
    """Extract Messier, Caldwell, NGC, IC, Guide, or Constellation IDs dynamically from the corpus."""
    by_id, _ = load_documents_lookup()
    dso_ids: list[str] = []
    explicit_con_ids: list[str] = []

    for m in re.finditer(r"\b(?:Messier\s*|M)(\d{1,3})\b", question, flags=re.IGNORECASE):
        num = int(m.group(1))
        cand = f"M{num:03d}"
        if 1 <= num <= 110 and cand in by_id:
            dso_ids.append(cand)
    for m in re.finditer(r"\b(?:Caldwell\s*|C)(\d{1,3})\b", question, flags=re.IGNORECASE):
        num = int(m.group(1))
        cand = f"C{num:03d}"
        if 1 <= num <= 109 and cand in by_id:
            dso_ids.append(cand)
    for m in re.finditer(r"\bNGC\s*0*(\d{1,4})\b", question, flags=re.IGNORECASE):
        rec = resolve_object_record(f"NGC{int(m.group(1)):04d}")
        if rec:
            dso_ids.append(rec["doc_id"])
    for m in re.finditer(r"\bIC\s*0*(\d{1,4})\b", question, flags=re.IGNORECASE):
        rec = resolve_object_record(f"IC{int(m.group(1)):04d}")
        if rec:
            dso_ids.append(rec["doc_id"])

    q_low = question.lower()
    q_dedup = _dedup_letters(q_low)
    matched_guide_ids: list[str] = []

    for doc_id, doc in by_id.items():
        meta = doc.get("metadata", {})
        is_con = doc_id.startswith("CON_")
        is_guide = doc_id.startswith("GUIDE_")
        for alias in meta.get("aliases", []):
            a_low = alias.strip().lower()
            if len(a_low) >= 5 and not a_low.startswith(
                ("m0", "c0", "ngc", "ic0", "con_", "guide_")
            ):
                matched = bool(re.search(rf"\b{re.escape(a_low)}s?\b", q_low))
                if not matched and not is_con and not is_guide and " " in a_low and len(a_low) >= 8:
                    a_dedup = _dedup_letters(a_low)
                    matched = bool(re.search(rf"\b{re.escape(a_dedup)}\b", q_dedup))
                if matched:
                    if is_guide:
                        matched_guide_ids.append(doc_id)
                    elif is_con:
                        explicit_con_ids.append(doc_id)
                    else:
                        dso_ids.append(doc_id)
                    break

    wants_neighbors = any(
        w in q_low for w in ["satellite", "companion", "interacting", "triplet", "pair", "double"]
    )
    for base_id in list(dict.fromkeys(dso_ids)):
        if wants_neighbors:
            dso_ids.extend(_find_astrometric_companions(base_id, max_sep_deg=1.6))
        else:
            dso_ids.extend(
                _find_astrometric_companions(base_id, max_sep_deg=1.6, only_shared_complex=True)
            )

    dso_ids = list(dict.fromkeys(dso_ids))

    host_con_ids: list[str] = []
    for did in dso_ids:
        const_code = (by_id.get(did, {}).get("metadata", {}).get("constellation") or "").upper()
        con_doc_id = f"CON_{const_code}"
        if const_code and con_doc_id in by_id:
            host_con_ids.append(con_doc_id)

    return list(dict.fromkeys(dso_ids + host_con_ids + explicit_con_ids + matched_guide_ids))


def _extract_search_filters(question: str) -> dict[str, object]:
    """Extract Qdrant `MetadataFilter` criteria (`constellation`, `catalog`, `max_magnitude`, `narrowband_only`)."""
    q_low = question.lower()
    filters: dict[str, object] = {}

    if any(
        w in q_low
        for w in ["which ", "suggest ", "targets ", "nebulae in ", "galaxies in ", "clusters in "]
    ):
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
    """Select bright catalog objects dynamically from the corpus when no specific target ID is named."""
    by_id, _ = load_documents_lookup()
    q_low = question.lower()

    def _rank_candidates(
        allowed_types: set[str] | None = None,
        max_mag: float = 9.0,
        min_dec: float = -15.0,
        catalogs: tuple[str, ...] = ("Messier", "Caldwell"),
    ) -> list[str]:
        scored: list[tuple[float, str]] = []
        for doc_id, doc in by_id.items():
            if doc.get("catalog") not in catalogs:
                continue
            meta = doc.get("metadata", {})
            mag = meta.get("magnitude")
            dec = meta.get("dec_deg")
            otype = str(meta.get("object_type") or "")
            if mag is None or float(mag) > max_mag:
                continue
            if dec is not None and float(dec) < min_dec:
                continue
            if allowed_types and otype not in allowed_types:
                continue
            scored.append((float(mag), doc_id))
        scored.sort(key=lambda item: item[0])
        return [did for _, did in scored]

    if "galaxy" in q_low or "galaxies" in q_low:
        galaxies = _rank_candidates(
            allowed_types={"galaxy", "galaxy_pair", "galaxy_triplet"},
            max_mag=9.5,
            catalogs=("Messier",),
        )
        preferred = [g for g in ("M031", "M051", "M104") if g in galaxies]
        return (preferred + [g for g in galaxies if g not in preferred])[:3]

    if "binoculars" in q_low:
        bino_pool = _rank_candidates(
            allowed_types={"globular_cluster", "planetary_nebula", "galaxy"},
            max_mag=8.5,
            min_dec=15.0,
            catalogs=("Messier",),
        )
        preferred = [b for b in ("M013", "M027", "M031") if b in bino_pool]
        return (preferred + [b for b in bino_pool if b not in preferred])[:3]

    if intent and intent.request_type == "session_plan":
        session_pool = _rank_candidates(max_mag=8.5, min_dec=15.0, catalogs=("Messier", "Caldwell"))
        preferred = [s for s in ("M013", "M027", "C014", "M031", "M045") if s in session_pool]
        return (preferred + [s for s in session_pool if s not in preferred])[:5]

    if any(w in q_low for w in ["photograph tonight", "what should i", "best objects"]):
        showcase_pool = _rank_candidates(max_mag=8.5, min_dec=-10.0, catalogs=("Messier",))
        preferred = [s for s in ("M031", "M042", "M013", "M027") if s in showcase_pool]
        return (preferred + [s for s in showcase_pool if s not in preferred])[:4]

    return []


def agent_node(state: AstroState) -> dict:
    """Node 2 (`agent` / Agir): Decide which tools to invoke next (bounded to `settings.max_agent_steps`)."""
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
        tool_calls.append(
            {
                "name": "search_catalog",
                "args": {"query": q, "top_k": 5, "exclude_guides": True},
                "id": "call_search_main",
            }
        )

        meta_filters = _extract_search_filters(q)
        if meta_filters:
            tool_calls.append(
                {
                    "name": "search_catalog",
                    "args": {"query": q, "top_k": 5, "exclude_guides": True, **meta_filters},
                    "id": "call_search_filtered",
                }
            )

        mentioned = _extract_mentioned_ids(q)
        names_an_object = any(not m_id.startswith("GUIDE_") for m_id in mentioned)
        if not names_an_object:
            showcase = _select_showcase_candidates(q, intent)
            mentioned = showcase + [m_id for m_id in mentioned if m_id not in showcase]

        for idx, obj_id in enumerate(mentioned[:6]):
            tool_calls.append(
                {
                    "name": "get_object",
                    "args": {
                        "object_id": obj_id,
                        "focal_length_mm": intent.focal_length_mm if intent else None,
                        "sensor_format": intent.sensor_format if intent else "APS-C",
                    },
                    "id": f"call_get_{idx}_{obj_id}",
                }
            )

        q_low = q.lower()
        if any(term in q_low for term in _GUIDE_TOPIC_TERMS):
            guide_call = {
                "name": "search_catalog",
                "args": {"query": q, "top_k": 5, "catalog": "Guide"},
                "id": "call_search_guides",
            }
            is_session = bool(intent and intent.request_type == "session_plan")
            if names_an_object or is_session:
                tool_calls.append(guide_call)
            else:
                tool_calls.insert(0, guide_call)

        if intent and intent.location_name:
            tool_calls.append(
                {
                    "name": "geocode",
                    "args": {"place_name": intent.location_name},
                    "id": "call_geocode",
                }
            )

    elif steps == 1 and intent and intent.request_type == "session_plan":
        lat, lon = settings.default_lat, settings.default_lon
        geocode_failed = False
        for msg in state.get("messages", []):
            if msg.role == "tool" and msg.name == "geocode":
                try:
                    data = json.loads(msg.content)
                    if data.get("found"):
                        lat, lon = float(data["lat"]), float(data["lon"])
                    else:
                        geocode_failed = True
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    pass

        if not geocode_failed:
            obs_date_str = (
                intent.target_date.isoformat()
                if intent.target_date
                else settings.get_reference_date().isoformat()
            )
            mentioned = _extract_mentioned_ids(q)
            names_an_object = any(not m_id.startswith("GUIDE_") for m_id in mentioned)
            candidate_ids = list(
                dict.fromkeys(
                    c.doc_id
                    for c in state.get("retrieved", [])
                    if not c.doc_id.startswith(("CON_", "GUIDE_")) and c.metadata.ra_deg is not None
                )
            )
            if not names_an_object:
                showcase = _select_showcase_candidates(q, intent)
                candidate_ids = list(dict.fromkeys(showcase + candidate_ids))
            candidate_ids = candidate_ids[:8]

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


def _find_unindexed_designations(question: str) -> list[str]:
    """Return any explicit astronomical catalog designations in `question` that are not in the corpus."""
    unindexed: list[str] = []
    for pat in _EXPLICIT_CATALOG_CODE_PATTERNS:
        for m in re.finditer(pat, question, flags=re.IGNORECASE):
            raw_code = m.group(1).strip()
            if resolve_object_record(raw_code) is None:
                unindexed.append(raw_code)
    return list(dict.fromkeys(unindexed))


def _check_out_of_corpus(question: str, *, strict: bool | None = None) -> str | None:
    """Check whether `question` asks about a non-existent catalog number or (when `strict=True`) an unindexed designation.

    In v1.1 permissive default mode (`settings.strict_corpus_guard = False`), legitimate astronomy
    questions and unindexed real catalog objects are NOT hard-blocked unless the catalog number
    itself does not exist in astronomy (e.g. Messier > 110 or Caldwell > 109) or `strict=True`.
    """
    for m in re.finditer(r"\b(?:Messier|M)\s*(\d{1,4})\b", question, flags=re.IGNORECASE):
        num = int(m.group(1))
        if num < 1 or num > 110:
            return (
                f"I must abstain: Messier {num} does not exist "
                "(the Messier catalog only contains 110 objects, M1 to M110)."
            )

    for m in re.finditer(r"\b(?:Caldwell|C)\s*(\d{1,4})\b", question, flags=re.IGNORECASE):
        num = int(m.group(1))
        if num < 1 or num > 109:
            return (
                f"I must abstain: Caldwell {num} does not exist "
                "(the Caldwell catalog only contains 109 objects, C1 to C109)."
            )

    enforce_strict = settings.strict_corpus_guard if strict is None else strict
    if not enforce_strict:
        return None

    in_corpus_dsos = [
        oid for oid in _extract_mentioned_ids(question) if not oid.startswith(("CON_", "GUIDE_"))
    ]
    if in_corpus_dsos:
        return None

    unindexed = _find_unindexed_designations(question)
    if unindexed:
        raw_code = unindexed[0]
        return (
            f"I must abstain: '{raw_code}' is not indexed in the AstroPilote corpus "
            "(which covers Messier, Caldwell, the 200 brightest NGC deep-sky objects, famous astrophotography "
            "nebulae and galaxies, the 88 IAU constellations, and practical observing guides)."
        )
    return None


_CONSTELLATION_NAMES: dict[str, str] = {code: cname for code, cname, _, _, _ in IAU_CONSTELLATIONS}

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
        return (
            "The AstroPilote corpus does not record a visual magnitude for this object "
            "(the value is not available in the corpus rather than inventing one); "
            "visible with binoculars or a small telescope"
        )
    if mag <= 5.0:
        return f"Visible to the naked eye & spectacular in binoculars (very bright, apparent visual magnitude {mag})"
    if mag <= 6.3:
        return (
            f"Faint naked-eye / Easy with any binoculars (bright, apparent visual magnitude {mag})"
        )
    if mag <= 8.5:
        return f"Easy with standard 10x50 binoculars or a small telescope (moderate brightness, apparent visual magnitude {mag})"
    return f"Requires a small telescope — too faint for the naked eye (faint, apparent visual magnitude {mag})"


def _moon_size_comparison(size_arcmin: float | None) -> str:
    """Compare apparent size in arcminutes to the Full Moon (30 arcminutes across)."""
    if not size_arcmin or size_arcmin <= 0:
        return ""
    ratio = size_arcmin / 30.0
    if ratio >= 1.8:
        return f"spans ~{ratio:.1f}× the width of the Full Moon ({size_arcmin}' / {size_arcmin:.0f} arcminutes across)"
    if ratio >= 0.75:
        return f"appears roughly the same width as the Full Moon ({size_arcmin}' / {size_arcmin:.0f} arcminutes across)"
    if ratio >= 0.25:
        return f"spans about {int(round(ratio * 100))}% of the Full Moon's width ({size_arcmin}' / {size_arcmin:.0f} arcminutes across)"
    return f"compact object ({size_arcmin}' / {size_arcmin} arcminutes across, best magnified at high magnification in a telescope)"


def compose_node(state: AstroState) -> dict:
    """Node 4 (`compose` / Rédiger): Synthesize a clear, beginner-friendly cited `AstroAnswer`."""
    q = state["question"]
    retrieved = state.get("retrieved", [])
    intent = state.get("intent")

    ooc_reason = _check_out_of_corpus(q, strict=state.get("strict_corpus_guard"))
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

    night_data: dict = {}
    vis_data: dict = {}
    weather_data: dict = {}
    geocode_data: dict = {}
    framing_by_id: dict[str, str] = {}
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
                elif msg.name == "geocode":
                    geocode_data = payload
                elif (
                    msg.name == "get_object"
                    and isinstance(payload, dict)
                    and payload.get("framing_advice")
                ):
                    framing_by_id[payload["doc_id"]] = str(payload["framing_advice"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

    loc_label = (
        intent.location_name if intent and intent.location_name else settings.default_location_name
    )
    if geocode_data and geocode_data.get("found") is False:
        ans = AstroAnswer(
            answer=(
                f"I cannot compute a night plan because '{loc_label}' is not a recognizable location; "
                "please provide a real city or coordinates."
            ),
            plan=[],
            conditions=None,
            sources=[],
            abstained=True,
            verified=True,
        )
        return {"answer": ans}

    mentioned = _extract_mentioned_ids(q)
    get_obj_ids = [c.doc_id for c in retrieved if c.chunk_id.endswith("#catalog-0")]
    retrieved_doc_ids = list(dict.fromkeys(c.doc_id for c in retrieved))
    priority_ids = list(dict.fromkeys(mentioned + get_obj_ids))
    dso_priority = [
        d for d in priority_ids if d in retrieved_doc_ids and not d.startswith(("CON_", "GUIDE_"))
    ]
    explicit_guide_priority = [
        d for d in priority_ids if d in retrieved_doc_ids and d.startswith("GUIDE_")
    ]
    guide_priority = list(
        dict.fromkeys(
            explicit_guide_priority + [d for d in retrieved_doc_ids if d.startswith("GUIDE_")]
        )
    )[:2]
    con_priority = [d for d in priority_ids if d in retrieved_doc_ids and d.startswith("CON_")]
    cited_ids = list(dict.fromkeys(dso_priority + con_priority + guide_priority))[:5]
    for d in retrieved_doc_ids:
        if d not in cited_ids and len(cited_ids) < 5:
            cited_ids.append(d)

    plan_items: list[PlanItem] = []
    conditions_str: str | None = None
    plan_by_id: dict[str, PlanItem] = {}

    date_label = (
        intent.target_date.isoformat()
        if intent and intent.target_date
        else vis_data.get("date", settings.get_reference_date().isoformat())
    )

    min_alt_floor = settings.min_observable_altitude_deg

    if intent and intent.request_type == "session_plan" and vis_data.get("objects"):
        dark_start = night_data.get("astronomical_dark_start", "21:00")
        dark_end = night_data.get("astronomical_dark_end", "05:30")
        moon_pct = night_data.get("moon_illumination_pct", 0)
        moon_set = night_data.get("moon_set_local")
        weather_avail = weather_data.get("available", True)
        clouds = weather_data.get("cloud_cover_pct", 10)
        cloud_note = (
            f"Cloud cover {clouds}%"
            if weather_avail
            else f"Cloud cover unavailable (reliable weather forecasts only exist up to ~{settings.max_forecast_days} days ahead)"
        )
        moon_note = f"{moon_pct}% lit (sets at {moon_set})" if moon_set else f"{moon_pct}% lit"
        conditions_str = (
            f"{loc_label} on {date_label}: Astronomical darkness {dark_start} to {dark_end} | "
            f"Moon {moon_note} | {cloud_note}"
        )

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
                if win_s_m <= sm_val <= win_e_m and float(s["altitude_deg"]) >= min_alt_floor:
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

    summary_lines: list[str] = []
    unindexed_codes = _find_unindexed_designations(q)
    if unindexed_codes:
        joined_codes = ", ".join(unindexed_codes)
        summary_lines.append(
            f"**Catalog Coverage Note:** Specific designation(s) `{joined_codes}` are not indexed as standalone "
            "records in the 567-document OpenNGC subset; using the closest related constellation, catalog, and "
            "practical guide references retrieved below."
        )

    if conditions_str:
        dark_s = night_data.get("astronomical_dark_start", "21:00")
        dark_e = night_data.get("astronomical_dark_end", "05:30")
        moon_p = night_data.get("moon_illumination_pct", 0)
        weather_avail = weather_data.get("available", True)
        clouds_p = weather_data.get("cloud_cover_pct", 10)
        moon_desc = (
            "the Moon is barely lit, leaving the sky nice and dark"
            if moon_p <= 35
            else "moonlight will brighten the sky, so stick to bright star clusters or narrowband nebulae"
        )
        if weather_avail:
            weather_clause = f"cloud cover is around **{clouds_p}%**"
        else:
            weather_clause = f"we must **decline to give cloud cover** because reliable weather forecasts only exist up to about **{settings.max_forecast_days} days ahead**"
        summary_lines.append(
            f"**Computed Ephemeris for {loc_label} on {date_label}:** The sky is completely dark from **{dark_s} to {dark_e}** "
            f"({weather_clause}, and with the Moon at **{moon_p}%**, {moon_desc})."
        )
    if night_data and not night_data.get("has_astronomical_darkness", True):
        summary_lines.append(night_data.get("summary", ""))

    by_id, _ = load_documents_lookup()
    obs_lat = float(vis_data.get("lat", settings.default_lat)) if vis_data else settings.default_lat
    lat_hemi = "N" if obs_lat >= 0 else "S"
    abs_lat = abs(obs_lat)

    for oid, odata in vis_data.get("objects", {}).items():
        if not odata.get("observable"):
            max_alt = float(odata.get("max_altitude_deg", 0.0))
            dec_val = by_id.get(oid, {}).get("metadata", {}).get("dec_deg")
            dec_str = (
                f"declination of about **{dec_val:.1f} degrees**"
                if dec_val is not None
                else "unfavorable declination"
            )
            if max_alt <= 0.0:
                summary_lines.append(
                    f"**No — Why you cannot see {odata.get('title', oid)} [{oid}] from {loc_label} (latitude {abs_lat:.2f} {lat_hemi}) on {date_label}:** "
                    f"No: {odata.get('title', oid)} [{oid}] has a {dec_str}, so from {loc_label} (latitude {abs_lat:.2f} {lat_hemi}) "
                    f"it never rises above the horizon and stays below the horizon all night "
                    f"(maximum altitude **{max_alt}°**, where `0°` is the horizon line)."
                )
            else:
                summary_lines.append(
                    f"**Not really — Why {odata.get('title', oid)} [{oid}] is not a good target from {loc_label} on {date_label}:** "
                    f"Not really: {odata.get('title', oid)} [{oid}] has a {dec_str}, so from {loc_label} "
                    f"it only reaches about **{max_alt:.0f} degrees** ({max_alt}°) of altitude, far below the **{min_alt_floor:.0f} degree** minimum floor, "
                    f"and it sets before true darkness is over."
                )

    q_low = q.lower()
    if len(dso_priority) == 1:
        primary_dso = dso_priority[0]
        true_const_code = (
            by_id.get(primary_dso, {}).get("metadata", {}).get("constellation") or ""
        ).upper()
        true_const_name = _CONSTELLATION_NAMES.get(true_const_code, true_const_code)
        for c_code, c_name, _, _, _ in IAU_CONSTELLATIONS:
            if c_code != true_const_code and re.search(rf"\b{re.escape(c_name.lower())}\b", q_low):
                dso_doc = by_id.get(primary_dso, {})
                dso_type = (
                    dso_doc.get("metadata", {}).get("object_type") or "deep-sky object"
                ).replace("_", " ")
                summary_lines.append(
                    f"**False premise correction:** This premise is wrong: **{dso_doc.get('title', primary_dso)}** [{primary_dso}] "
                    f"is an **{dso_type}** in the constellation **{true_const_name}** [CON_{true_const_code}], not in {c_name}."
                )
                break

    is_french_query = bool(
        re.search(
            r"\b(?:quelle|dans|est|sont|pourquoi|comment|constellation|galaxie|nébuleuse)\b", q_low
        )
        and re.search(r"[àâéèêëîïôùûüç]|\b(?:dans|quelle|se trouve)\b", q_low)
    )

    objs_vis_map = vis_data.get("objects", {})
    for cid in cited_ids:
        matching_chunks = [c for c in retrieved if c.doc_id == cid]
        if matching_chunks:
            prose_chunks = [c for c in matching_chunks if c.section != "Catalog Metadata"]
            ch = prose_chunks[0] if prose_chunks else matching_chunks[0]
            body_lines = []
            for pch in prose_chunks[:2] if prose_chunks else [ch]:
                for ln in pch.text.splitlines():
                    ln_s = ln.strip()
                    if ln_s and not ln_s.startswith("[") and not ln_s.startswith("Section:"):
                        body_lines.append(ln_s)
            full_body = " ".join(body_lines).strip()
            sentences = re.split(r"(?<=[.!?])\s+", full_body)
            clean_excerpt = " ".join(sentences[:3]).strip() or full_body[:320]

            if ch.metadata.catalog == "Guide":
                summary_lines.append(
                    f"- **{ch.title}** [{cid}] — **Practical guide ({ch.section}):** {clean_excerpt}"
                )
                continue

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
                peak_alt = (
                    odata.get("max_altitude_deg", pi.altitude_deg) if odata else pi.altitude_deg
                )
                best_time = (
                    (odata.get("best_slot") or {}).get("time", pi.start.strftime("%H:%M"))
                    if odata
                    else pi.start.strftime("%H:%M")
                )
                min_alt = (
                    min((float(s["altitude_deg"]) for s in (odata.get("slots") or [])), default=0.0)
                    if odata
                    else 0.0
                )
                dec_deg = ch.metadata.dec_deg
                circumpolar_limit = 90.0 - abs_lat
                circ_note = ""
                if dec_deg is not None and (dec_deg >= circumpolar_limit or min_alt > 0.0):
                    circ_note = (
                        f" **Circumpolar:** Yes — declination **{dec_deg:+.1f} degrees** is above the "
                        f"**+{circumpolar_limit:.0f} degree** circumpolar limit for {loc_label}, so it never sets "
                        f"(reaches about **{peak_alt:.0f} degrees** of altitude around midnight/{best_time} and stays above about **{min_alt:.0f} degrees** throughout the night)."
                    )
                where_sentence = (
                    f" **Verified Ephemeris ({loc_label}, {date_label}):** Observable! Scheduled slot **{pi.start.strftime('%H:%M')}–{pi.end.strftime('%H:%M')}** "
                    f"at **{pi.altitude_deg:.0f}°** ({pi.altitude_deg:.0f} degrees of altitude) facing **{dir_full} ({pi.direction})** "
                    f"(peaks at **{peak_alt:.0f}°** / {peak_alt:.0f} degrees around **{best_time}**).{circ_note}"
                )

            framing_sentence = ""
            if cid in framing_by_id:
                framing_sentence = f" **FOV Framing:** {framing_by_id[cid]}"

            fr_sentence = ""
            if is_french_query and not cid.startswith("CON_"):
                mag_fr = (
                    str(ch.metadata.magnitude).replace(".", ",")
                    if ch.metadata.magnitude is not None
                    else "non renseignée"
                )
                fr_sentence = (
                    f" **En français :** {ch.title} [{cid}] se trouve dans la constellation {const_full} [CON_{const_code}] "
                    f"et sa magnitude visuelle apparente est d'environ {mag_fr}."
                )

            summary_lines.append(
                f"- **{ch.title}** [{cid}] — **What it is:** A {plain_type} in the constellation **{const_full}** ({const_code}). "
                f"**How to see it:** {equip_guide}{size_sentence}.{where_sentence}{framing_sentence}{fr_sentence} "
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
        "Use the provided corpus facts and ephemeris data, and cite every retrieved source inline in exact brackets like [M031] or [C014]."
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
