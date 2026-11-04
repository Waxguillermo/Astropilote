"""Entity & catalog designation resolution with pre-compiled corpus alias regexes and hardware disambiguation."""

from __future__ import annotations

import re
from functools import lru_cache

from astropilote.agent.i18n import refusal_message
from astropilote.config import settings
from astropilote.ingest.catalog import IAU_CONSTELLATIONS
from astropilote.optics import angular_separation_deg
from astropilote.tools.catalog import load_documents_lookup, resolve_object_record

# Re-export spherical angular separation helper for backward compatibility
_angular_separation_deg = angular_separation_deg

CONSTELLATION_NAMES: dict[str, str] = {code: cname for code, cname, _, _, _ in IAU_CONSTELLATIONS}

# Pre-compiled catalog designation patterns
_MESSIER_MATCH_RE = re.compile(r"\b(Messier|M)\s*(\d{1,4})\b", flags=re.IGNORECASE)
_CALDWELL_MATCH_RE = re.compile(r"\b(Caldwell|C)\s*(\d{1,4})\b", flags=re.IGNORECASE)
_NGC_MATCH_RE = re.compile(r"\bNGC\s*0*(\d{1,4})\b", flags=re.IGNORECASE)
_IC_MATCH_RE = re.compile(r"\bIC\s*0*(\d{1,4})\b", flags=re.IGNORECASE)

_EXPLICIT_CATALOG_RES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pat, flags=re.IGNORECASE)
    for pat in (
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
    )
)

# Mechanical thread, optical adapter, camera body, and highway disambiguation patterns
_METRIC_PITCH_AFTER_RE = re.compile(r"^\s*(?:[x×*]\s*\d+(?:[.,]\d+)?|mm\b)", flags=re.IGNORECASE)
_THREAD_OR_ROAD_WINDOW_RE = re.compile(
    r"\b(?:threads?|threaded|filetages?|filet[ée]e?s?|pas\s+de\s+vis|bagues?|"
    r"adapters?|adaptateurs?|spacers?|allonges?|rings?|tilt|back-?focus|"
    r"focuser|porte-oculaire|cr[ée]maill[èe]re|flange|m[âa]le|femelle|"
    r"autoroutes?|motorways?|highways?)\b",
    flags=re.IGNORECASE,
)
_CAMERA_BRAND_BEFORE_RE = re.compile(
    r"\b(?:canon|eos|cinema|leica|lumix|nikon|sony)\s*(?:[a-z0-9_-]+\s*){0,2}$",
    flags=re.IGNORECASE,
)
_CAMERA_MODEL_WINDOW_RE = re.compile(
    r"\b(?:camera|cam[ée]ra|cmos|canon|eos|cinema)\b",
    flags=re.IGNORECASE,
)
_ADJACENT_CAMERA_RE = re.compile(
    r"(?:\b(?:camera|cam[ée]ra|cmos)\s+$|^\s+(?:camera|cam[ée]ra|cmos)\b)",
    flags=re.IGNORECASE,
)
_WORD_TOKENS_RE = re.compile(r"[a-z]{4,}")
_SHARED_STEM_IGNORE = frozenset(
    {"nebula", "galaxy", "cluster", "messier", "caldwell", "open", "globular", "planetary"}
)


def _is_non_astronomical_mc_code(question: str, match: re.Match[str]) -> bool:
    """Return True when shorthand `M<num>` or `C<num>` denotes mechanical threads, camera models, or roads."""
    prefix = match.group(1)
    # Full words 'Messier' or 'Caldwell' always refer to the astronomical catalogs
    if len(prefix) > 1:
        return False

    start, end = match.start(), match.end()
    after = question[end : end + 32]
    before = question[max(0, start - 32) : start]
    window = question[max(0, start - 80) : min(len(question), end + 80)]

    if _METRIC_PITCH_AFTER_RE.search(after):
        return True
    if _THREAD_OR_ROAD_WINDOW_RE.search(window):
        return True
    if _CAMERA_BRAND_BEFORE_RE.search(before):
        return True
    if _ADJACENT_CAMERA_RE.search(before) or _ADJACENT_CAMERA_RE.search(after):
        return True

    num = int(match.group(2))
    if prefix.upper() == "C" and num in (70, 100, 200, 300, 400, 500):
        near_window = question[max(0, start - 55) : min(len(question), end + 55)]
        if _CAMERA_MODEL_WINDOW_RE.search(near_window):
            return True

    return False


@lru_cache(maxsize=1)
def _compiled_corpus_aliases() -> tuple[tuple[str, re.Pattern[str], str], ...]:
    """Pre-compile corpus common-name alias regexes once at first lookup."""
    by_id, _ = load_documents_lookup()
    entries: list[tuple[str, re.Pattern[str], str]] = []
    for doc_id, doc in by_id.items():
        for alias in doc.get("metadata", {}).get("aliases", []):
            a_low = alias.strip().lower()
            if len(a_low) >= 5 and not a_low.startswith(
                ("m0", "c0", "ngc", "ic0", "con_", "guide_")
            ):
                pat = re.compile(rf"\b{re.escape(a_low)}s?\b")
                entries.append((a_low, pat, doc_id))
    return tuple(entries)


def _shares_complex_stem(doc_a: dict, doc_b: dict) -> bool:
    """Return True if two catalog entries share a multi-component nebula/cluster name stem (e.g. Veil Nebula)."""
    words_a = set(_WORD_TOKENS_RE.findall((doc_a.get("title") or "").lower()))
    words_b = set(_WORD_TOKENS_RE.findall((doc_b.get("title") or "").lower()))
    return bool((words_a & words_b) - _SHARED_STEM_IGNORE)


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
        sep = angular_separation_deg(float(ra1), float(dec1), float(ra2), float(dec2))
        if sep <= max_sep_deg:
            companions.append((sep, doc_id))

    companions.sort(key=lambda x: x[0])
    return [cid for _, cid in companions[:2]]


def _resolve_corpus_ids(
    question: str,
    mentioned_targets: list[str] | None = None,
    wants_companions: bool = False,
) -> list[str]:
    """Resolve LLM-extracted targets and explicit catalog designations against the 567-doc corpus."""
    by_id, _ = load_documents_lookup()
    dso_ids: list[str] = []
    con_ids: list[str] = []
    guide_ids: list[str] = []

    def _register_doc(doc_id: str) -> None:
        if doc_id.startswith("GUIDE_"):
            guide_ids.append(doc_id)
        elif doc_id.startswith("CON_"):
            con_ids.append(doc_id)
        else:
            dso_ids.append(doc_id)

    for raw_target in mentioned_targets or []:
        rec = resolve_object_record(raw_target)
        if rec:
            _register_doc(str(rec["doc_id"]))

    for m in _MESSIER_MATCH_RE.finditer(question):
        if _is_non_astronomical_mc_code(question, m):
            continue
        num = int(m.group(2))
        cand = f"M{num:03d}"
        if 1 <= num <= 110 and cand in by_id:
            dso_ids.append(cand)

    for m in _CALDWELL_MATCH_RE.finditer(question):
        if _is_non_astronomical_mc_code(question, m):
            continue
        num = int(m.group(2))
        cand = f"C{num:03d}"
        if 1 <= num <= 109 and cand in by_id:
            dso_ids.append(cand)

    for m in _NGC_MATCH_RE.finditer(question):
        rec = resolve_object_record(f"NGC{int(m.group(1)):04d}")
        if rec:
            _register_doc(str(rec["doc_id"]))

    for m in _IC_MATCH_RE.finditer(question):
        rec = resolve_object_record(f"IC{int(m.group(1)):04d}")
        if rec:
            _register_doc(str(rec["doc_id"]))

    q_low = question.lower()
    seen_alias_docs: set[str] = set()
    for a_low, pat, doc_id in _compiled_corpus_aliases():
        if doc_id not in seen_alias_docs and a_low in q_low and pat.search(q_low):
            seen_alias_docs.add(doc_id)
            _register_doc(doc_id)

    base_dsos = list(dict.fromkeys(dso_ids))
    for base_id in base_dsos:
        dso_ids.extend(
            _find_astrometric_companions(
                base_id,
                max_sep_deg=1.8 if len(base_dsos) <= 2 else 1.6,
                only_shared_complex=False if len(base_dsos) <= 2 else (not wants_companions),
            )
        )

    dso_ids = list(dict.fromkeys(dso_ids))
    host_con_ids: list[str] = []
    for did in dso_ids:
        const_code = (by_id.get(did, {}).get("metadata", {}).get("constellation") or "").upper()
        con_doc_id = f"CON_{const_code}"
        if const_code and con_doc_id in by_id:
            host_con_ids.append(con_doc_id)

    return list(dict.fromkeys(dso_ids + host_con_ids + con_ids + guide_ids))


def _find_unindexed_designations(question: str) -> list[str]:
    """Return any explicit astronomical catalog designations in `question` absent from the 567-doc corpus."""
    unindexed: list[str] = []
    for pat in _EXPLICIT_CATALOG_RES:
        for m in pat.finditer(question):
            raw_code = m.group(1).strip()
            if resolve_object_record(raw_code) is None:
                unindexed.append(raw_code)
    return list(dict.fromkeys(unindexed))


def _check_out_of_corpus(
    question: str, *, strict: bool | None = None, language: str = "en"
) -> str | None:
    """Check bounded catalog limits (`Messier 1..110`, `Caldwell 1..109`) and optional strict corpus membership."""
    for m in _MESSIER_MATCH_RE.finditer(question):
        if _is_non_astronomical_mc_code(question, m):
            continue
        num = int(m.group(2))
        if num < 1 or num > 110:
            return refusal_message("messier_out_of_range", language, num=num)

    for m in _CALDWELL_MATCH_RE.finditer(question):
        if _is_non_astronomical_mc_code(question, m):
            continue
        num = int(m.group(2))
        if num < 1 or num > 109:
            return refusal_message("caldwell_out_of_range", language, num=num)

    enforce_strict = settings.strict_corpus_guard if strict is None else strict
    if not enforce_strict:
        return None

    in_corpus_dsos = [
        oid for oid in _resolve_corpus_ids(question) if not oid.startswith(("CON_", "GUIDE_"))
    ]
    if in_corpus_dsos:
        return None

    unindexed = _find_unindexed_designations(question)
    if unindexed:
        return refusal_message("not_indexed", language, code=unindexed[0])
    return None
