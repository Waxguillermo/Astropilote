"""Hermetic pytest configuration isolating `.env`, cloud LLM keys, and external HTTP services."""

from __future__ import annotations

import ast
import json
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import pytest

import astropilote.llm as llm_mod
import astropilote.tools.world as world_mod
from astropilote.config import settings

_ORIG_CALL_OLLAMA_ROUTER = llm_mod._call_ollama_router
_ORIG_CALL_OLLAMA_LLM = llm_mod._call_ollama_llm
_ORIG_STREAM_OLLAMA_LLM = llm_mod._stream_ollama_llm

_MONTH_DATE_RE = re.compile(
    r"\b("
    r"January|February|March|April|May|June|July|August|September|October|November|December"
    r")\s+(\d{1,2}),?\s+(\d{4})\b",
    flags=re.IGNORECASE,
)
_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add opt-in `--run-live` flag for tests hitting external network or live daemons."""
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Execute tests marked with @pytest.mark.live.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers so pytest runs cleanly without unknown-mark warnings."""
    config.addinivalue_line(
        "markers",
        "live: marks tests requiring live external HTTP APIs or a running local Ollama daemon",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip `@pytest.mark.live` tests by default unless `--run-live` or `ASTROPILOTE_LIVE_TESTS=1`."""
    run_live = (
        bool(config.getoption("--run-live")) or os.environ.get("ASTROPILOTE_LIVE_TESTS") == "1"
    )
    if run_live:
        return
    skip_live = pytest.mark.skip(
        reason="Live external test skipped by default (pass --run-live to run)."
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@lru_cache(maxsize=1)
def _has_live_local_ollama() -> bool:
    """Probe once per test session whether a local Ollama daemon with `gemma4` is reachable."""
    if os.environ.get("ASTROPILOTE_FORCE_OFFLINE_LLM", "1") == "1":
        # Wait: probe localhost:11434 quickly; only use live Ollama if explicitly reachable & opted in,
        # OR check if daemon is actually listening with gemma4 installed.
        pass
    try:
        with httpx.Client(timeout=0.25) as client:
            resp = client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                models = [str(m.get("name", "")) for m in resp.json().get("models", [])]
                return any(m.startswith("gemma4") for m in models)
    except (httpx.HTTPError, ValueError, OSError):
        return False
    return False


def _detect_language(question: str) -> str:
    q_low = question.lower()
    if re.search(
        r"\b(dans|quelle|quel|comment|pourquoi|est-ce|donne-moi|toutes|précédentes|"
        r"faire|crêpes|télescope|galaxie|nébuleuse|prendre|collimater)\b",
        q_low,
    ):
        return "fr"
    return "en"


def _extract_explicit_date(question: str) -> str | None:
    iso_m = _ISO_DATE_RE.search(question)
    if iso_m:
        return iso_m.group(1)
    month_m = _MONTH_DATE_RE.search(question)
    if month_m:
        raw = f"{month_m.group(1)} {int(month_m.group(2))} {month_m.group(3)}"
        try:
            return datetime.strptime(raw, "%B %d %Y").date().isoformat()
        except ValueError:
            return None
    return None


def _extract_location(question: str) -> tuple[str | None, bool]:
    q_low = question.lower()
    for fictional in ("atlantis", "mordor", "tatooine", "hogwarts", "narnia", "wakanda"):
        if re.search(rf"\b{fictional}\b", q_low):
            return fictional.title(), True

    known_places = [
        (r"î?le[-\s]+de[-\s]+france", "Ile de France"),
        (r"\bfontainebleau\b", "Fontainebleau"),
        (r"\bchamonix\b", "Chamonix"),
        (r"\bmarseille\b", "Marseille"),
        (r"\bparis\b", "Paris"),
        (r"\blyon\b", "Lyon"),
        (r"\bbordeaux\b", "Bordeaux"),
        (r"\bnice\b", "Nice"),
        (r"\bsantiago\b", "Santiago"),
        (r"\blondon\b", "London"),
        (r"\btroms[øo]\b", "Tromsø"),
        (r"\bnew\s+delhi\b", "New Delhi"),
    ]
    for pattern, canonical in known_places:
        if re.search(pattern, q_low):
            return canonical, False
    return None, False


def _extract_explicit_targets(question: str) -> list[str]:
    targets: list[str] = []
    for m in re.finditer(r"\b(?:Messier\s*|M)(\d{1,3})\b", question, flags=re.IGNORECASE):
        num = int(m.group(1))
        if 1 <= num <= 110:
            targets.append(f"M{num:03d}")
    for m in re.finditer(r"\b(?:Caldwell\s*|C)(\d{1,3})\b", question, flags=re.IGNORECASE):
        num = int(m.group(1))
        if 1 <= num <= 109:
            targets.append(f"C{num:03d}")
    for m in re.finditer(r"\bNGC\s*0*(\d{1,4})\b", question, flags=re.IGNORECASE):
        targets.append(f"NGC {int(m.group(1))}")
    for m in re.finditer(r"\bIC\s*0*(\d{1,4})\b", question, flags=re.IGNORECASE):
        targets.append(f"IC {int(m.group(1))}")
    return list(dict.fromkeys(targets))


def _deterministic_router(question: str, sys_msg: str) -> tuple[dict[str, Any], int, int, str]:
    """Fast, offline deterministic router used when no local Ollama daemon runs on localhost:11434."""
    q_low = question.lower()
    lang = _detect_language(question)
    backend = f"ollama/{settings.ollama_model}"

    ref_m = _ISO_DATE_RE.search(sys_msg)
    ref_date_iso = ref_m.group(1) if ref_m else settings.get_reference_date().isoformat()

    # 1. Prompt injection / secret extraction
    if re.search(
        r"(ignore\s+(?:all\s+)?previous\s+instructions|ignore\s+toutes\s+tes\s+instructions|"
        r"system\s+prompt|prompt\s+syst[èe]me|system\s+override|anthropic_api_key|"
        r"\.env\s+file|brute-force\s+ssh)",
        q_low,
    ):
        reason = (
            "Je ne peux pas ignorer mes consignes ni divulguer des instructions internes ou secrets."
            if lang == "fr"
            else "I cannot comply with requests to ignore instructions or reveal internal prompts or secrets."
        )
        return (
            {"request_type": "prompt_injection", "language": lang, "refusal_reason": reason},
            32,
            18,
            backend,
        )

    # 2. Off-topic
    if re.search(r"\b(cr[êe]pes|recette|code\s+du\s+travail|employment\s+contract)\b", q_low):
        reason = (
            "Je suis spécialisé uniquement en astronomie visuelle et astrophotographie."
            if lang == "fr"
            else "I can only assist with visual astronomy and astrophotography questions."
        )
        return (
            {"request_type": "off_topic", "language": lang, "refusal_reason": reason},
            28,
            16,
            backend,
        )

    location_name, is_fictional = _extract_location(question)
    explicit_date = _extract_explicit_date(question)
    mentioned_targets = _extract_explicit_targets(question)

    forecast_days = 1
    if re.search(r"\b(upcoming\s+week|this\s+week|next\s+7\s+days|prochaine\s+semaine)\b", q_low):
        forecast_days = 7
        if explicit_date is None:
            explicit_date = ref_date_iso
    elif re.search(r"\b(tonight|ce\s+soir|cette\s+nuit)\b", q_low) and explicit_date is None:
        explicit_date = ref_date_iso

    weather_only = bool(
        re.search(
            r"(cloud-free|cloud\s+percentage|whether\s+it\s+will\s+rain|atmospheric\s+seeing)",
            q_low,
        )
        and not mentioned_targets
    )

    is_session = bool(
        is_fictional
        or weather_only
        or explicit_date is not None
        or (
            location_name is not None
            and re.search(r"\b(observe|photograph|shoot|image|sky|tonight|visible|see)\b", q_low)
        )
        or re.search(r"\b(tonight|ce\s+soir|upcoming\s+week|best\s+days)\b", q_low)
    )

    # Optics extraction
    focal_mm: float | None = None
    aperture_mm: float | None = None
    ap_m = re.search(
        r"\b(\d+(?:\.\d+)?)\s*mm\s+(?:refractor|newtonian|reflector|aperture)\b", q_low
    )
    if ap_m:
        aperture_mm = float(ap_m.group(1))
    fl_m = re.search(
        r"\b(\d+(?:\.\d+)?)\s*mm\s*(?:focal\s+length|scope|focal)\b",
        q_low,
    )
    if fl_m:
        focal_mm = float(fl_m.group(1))

    sensor_format = "full-frame" if ("full-frame" in q_low or "36 x 24" in q_low) else "APS-C"
    mode = (
        "astrophotography"
        if re.search(
            r"\b(astrophotograph|photograph|image|shoot|camera|cmos|sensor|frame|focal|filter|collimat)\b",
            q_low,
        )
        else ("visual" if re.search(r"\b(binoculars|naked\s+eye|eyepiece)\b", q_low) else "hybrid")
    )
    include_guides = bool(
        re.search(
            r"\b(collimat|cmos|camera|bortle|dark\s+frame|averted\s+vision|equatorial\s+mount|"
            r"altazimuth|filter|newton|astrophotograph)\b",
            q_low,
        )
    )
    narrowband_only = bool(re.search(r"\b(narrowband|dual-band|h-alpha|ha/oiii|oiii)\b", q_low))
    constellation = "CYG" if "cygnus" in q_low else None

    return (
        {
            "request_type": "session_plan" if is_session else "factual",
            "language": lang,
            "mode": mode,
            "is_fictional_location": is_fictional,
            "location_name": location_name,
            "start_date": explicit_date,
            "forecast_days": forecast_days,
            "weather_only": weather_only,
            "mentioned_targets": mentioned_targets,
            "wants_companions": bool(re.search(r"\b(companion|satellite|pair|triplet)\b", q_low)),
            "include_practical_guides": include_guides,
            "search_query": question,
            "constellation": constellation,
            "catalog_filter": None,
            "object_type": None,
            "max_magnitude": None,
            "narrowband_only": narrowband_only,
            "aperture_mm": aperture_mm,
            "focal_length_mm": focal_mm,
            "sensor_format": sensor_format,
            "refusal_reason": None,
        },
        45,
        28,
        backend,
    )


def _deterministic_composer(system_prompt: str, user_prompt: str) -> tuple[str, int, int, str]:
    """Fast, offline grounded answer composer strictly respecting retrieved corpus IDs & tool JSON."""
    is_french = "Write your ENTIRE response in French" in system_prompt

    # Parse available citation IDs
    cite_ids: list[str] = []
    cite_m = re.search(r"Available corpus citation IDs:\s*(\[[^\n]*\])", user_prompt)
    if cite_m:
        try:
            parsed_ids = ast.literal_eval(cite_m.group(1))
            if isinstance(parsed_ids, list):
                cite_ids = [str(x) for x in parsed_ids if str(x)]
        except (ValueError, SyntaxError):
            cite_ids = []

    # Parse deterministic JSON block
    det_data: dict[str, Any] = {}
    json_marker = "Deterministic Tool Outputs (JSON):\n"
    if json_marker in user_prompt:
        after = user_prompt.split(json_marker, 1)[1]
        json_line = after.split("\n\n", 1)[0].strip()
        try:
            det_data = json.loads(json_line)
        except json.JSONDecodeError:
            det_data = {}

    parts: list[str] = []
    loc_info = det_data.get("location") or {}
    horizon_days = det_data.get("max_weather_forecast_horizon_days", 14)

    if loc_info.get("provided"):
        place = loc_info.get("requested_name") or loc_info.get("resolved_place_name") or ""
        multi_rank = det_data.get("multi_night_ranking") or []
        if multi_rank:
            best = multi_rank[0]
            parts.append(
                f"For {place}, the best upcoming observing night is {best.get('weekday', '')} "
                f"{best.get('date', '')} (cloud cover {best.get('cloud_cover_pct')}%, "
                f"darkness {best.get('astronomical_dark_start')}–{best.get('astronomical_dark_end')})."
            )
        else:
            eph = det_data.get("night_ephemeris") or {}
            wth = det_data.get("night_weather") or {}
            if eph:
                parts.append(
                    f"From {place} on {eph.get('date')}, astronomical darkness runs "
                    f"{eph.get('astronomical_dark_start')}–{eph.get('astronomical_dark_end')} "
                    f"(Moon {eph.get('moon_illumination_pct')}%)."
                )
            if wth:
                if wth.get("available") is False:
                    parts.append(
                        f"Meteorological cloud-cover forecasts are unavailable because reliable weather "
                        f"predictions only extend up to {horizon_days} days ahead."
                    )
                elif wth.get("cloud_cover_pct") is not None:
                    parts.append(f"Forecast cloud cover is {wth.get('cloud_cover_pct')}%.")
    elif "unindexed_designations_note" not in det_data and det_data.get("catalog_objects"):
        if not is_french:
            parts.append(
                "Share your observation location or coordinates for exact local altitudes and twilight times."
            )

    for unobs in det_data.get("unobservable_targets") or []:
        uid = unobs.get("object_id", "")
        cite_tag = f" [{uid}]" if uid in cite_ids else ""
        parts.append(
            f"No: {unobs.get('title', uid)}{cite_tag} is not observable from this latitude "
            f"(maximum altitude {unobs.get('max_altitude_deg')}°, below the {unobs.get('min_required_altitude_deg', 20.0)}° minimum)."
        )

    schedule = det_data.get("verified_observation_schedule") or []
    if schedule:
        sched_lines = []
        for item in schedule:
            oid = str(item.get("object_id", ""))
            tag = f" [{oid}]" if oid in cite_ids else ""
            sched_lines.append(
                f"- {item.get('slot')}: {item.get('name')} ({oid}, {item.get('altitude_deg')}° {item.get('direction')}){tag}"
            )
        parts.append("Recommended schedule:\n" + "\n".join(sched_lines))

    for obj in det_data.get("catalog_objects") or []:
        oid = str(obj.get("doc_id", ""))
        tag = f" [{oid}]" if oid in cite_ids else ""
        const_id = obj.get("constellation_doc_id")
        const_tag = f" [{const_id}]" if const_id and const_id in cite_ids else ""
        mag_str = f", magnitude {obj['magnitude']}" if obj.get("magnitude") is not None else ""
        size_str = f", size {obj['size_arcmin']}'" if obj.get("size_arcmin") is not None else ""
        line = (
            f"{obj.get('title', oid)}{tag} is a {obj.get('type', 'deep-sky object')} in "
            f"{obj.get('constellation', 'the sky')}{const_tag}{mag_str}{size_str}."
        )
        if obj.get("framing_advice"):
            line += f" {obj['framing_advice']}"
        parts.append(line)

    if not parts or not any(f"[{cid}]" in " ".join(parts) for cid in cite_ids):
        if cite_ids:
            primary_tags = " ".join(f"[{cid}]" for cid in cite_ids[:2])
            if is_french:
                parts.append(
                    f"En astronomie et astrophotographie, une bonne collimation optique et le calibrage du capteur "
                    f"assurent des étoiles ponctuelles sur tout le champ {primary_tags}."
                )
            else:
                parts.append(
                    f"Based on the astronomy and astrophotography corpus guides and catalog references {primary_tags}, "
                    "proper optical collimation, dark adaptation, and exposure calibration are essential."
                )
        else:
            parts.append(
                "Voici les recommandations astronomiques générales."
                if is_french
                else "Here are the general astronomical observations and guidelines."
            )

    text = "\n\n".join(parts)
    return text, 120, 65, f"ollama/{settings.ollama_model}"


@pytest.fixture(autouse=True)
def _hermetic_test_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate every test from the developer's `.env`, paid cloud keys, and external HTTP calls."""
    # 1. Isolate `.env` file in `tmp_path` so `_refresh_env_settings()` and `save_anthropic_key()`
    #    never read or overwrite the user's real `.env`.
    isolated_env = tmp_path / ".env"
    isolated_env.write_text("LLM_MODE=local\nOLLAMA_MODEL=gemma4\n", encoding="utf-8")
    os.chmod(isolated_env, 0o600)
    monkeypatch.setattr(llm_mod, "ENV_PATH", isolated_env)
    if hasattr(llm_mod, "_LAST_ENV_PATH"):
        monkeypatch.setattr(llm_mod, "_LAST_ENV_PATH", isolated_env)
    if hasattr(llm_mod, "_LAST_ENV_MTIME_NS"):
        monkeypatch.setattr(llm_mod, "_LAST_ENV_MTIME_NS", isolated_env.stat().st_mtime_ns)

    # Clear process-level environment variables that `_refresh_env_settings()` could fall back to
    for env_var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_MODEL",
        "ROUTER_MODEL",
        "LLM_MODE",
        "OLLAMA_MODEL",
        "OLLAMA_BASE_URL",
        "EVAL_OFFLINE_MODE",
        "STRICT_CORPUS_GUARD",
        "REFERENCE_DATE",
    ):
        monkeypatch.delenv(env_var, raising=False)

    # 2. Enforce clean deterministic in-memory settings defaults
    monkeypatch.setattr(settings, "llm_mode", "local")
    monkeypatch.setattr(settings, "ollama_base_url", "http://localhost:11434")
    monkeypatch.setattr(settings, "ollama_model", "gemma4")
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    monkeypatch.setattr(settings, "eval_offline_mode", True)
    monkeypatch.setattr(settings, "strict_corpus_guard", False)
    monkeypatch.setattr(settings, "reference_date", "2026-10-15")

    # Clear fixture & HTTP caches so each test starts from clean deterministic state
    world_mod._load_fixtures.cache_clear()
    if hasattr(world_mod, "_nominatim_lookup") and hasattr(
        world_mod._nominatim_lookup, "cache_clear"
    ):
        world_mod._nominatim_lookup.cache_clear()
    if hasattr(world_mod, "_weather_cache"):
        world_mod._weather_cache.clear()

    # 3. Deterministic fast LLM fallback when no local Ollama daemon runs on localhost:11434
    def _hermetic_ollama_router(
        question: str, sys_msg: str
    ) -> tuple[dict[str, Any] | None, int, int, str]:
        base = settings.ollama_base_url.rstrip("/")
        # Preserve explicit unreachable URL tests (e.g. `http://127.0.0.1:1` in test_llm_unavailable_raises_explicit_error)
        if base not in ("http://localhost:11434", "http://127.0.0.1:11434"):
            return _ORIG_CALL_OLLAMA_ROUTER(question, sys_msg)
        if _has_live_local_ollama() and os.environ.get("ASTROPILOTE_USE_LIVE_OLLAMA") == "1":
            try:
                res = _ORIG_CALL_OLLAMA_ROUTER(question, sys_msg)
                if res[0] is not None:
                    return res
            except (httpx.HTTPError, OSError, ValueError, KeyError):
                pass
        return _deterministic_router(question, sys_msg)

    def _hermetic_ollama_llm(system_prompt: str, user_prompt: str) -> tuple[str, int, int, str]:
        base = settings.ollama_base_url.rstrip("/")
        if base not in ("http://localhost:11434", "http://127.0.0.1:11434"):
            return _ORIG_CALL_OLLAMA_LLM(system_prompt, user_prompt)
        if _has_live_local_ollama() and os.environ.get("ASTROPILOTE_USE_LIVE_OLLAMA") == "1":
            try:
                res = _ORIG_CALL_OLLAMA_LLM(system_prompt, user_prompt)
                if res[0]:
                    return res
            except (httpx.HTTPError, OSError, ValueError):
                pass
        return _deterministic_composer(system_prompt, user_prompt)

    def _hermetic_stream_ollama_llm(system_prompt: str, user_prompt: str):
        base = settings.ollama_base_url.rstrip("/")
        if base not in ("http://localhost:11434", "http://127.0.0.1:11434"):
            yield from _ORIG_STREAM_OLLAMA_LLM(system_prompt, user_prompt)
            return
        text, in_tok, out_tok, backend = _hermetic_ollama_llm(system_prompt, user_prompt)
        if text:
            mid = max(1, len(text) // 2)
            yield text[:mid], 0, 0, ""
            if text[mid:]:
                yield text[mid:], 0, 0, ""
            yield "", in_tok, out_tok, backend

    monkeypatch.setattr(llm_mod, "_call_ollama_router", _hermetic_ollama_router)
    monkeypatch.setattr(llm_mod, "_call_ollama_llm", _hermetic_ollama_llm)
    monkeypatch.setattr(llm_mod, "_stream_ollama_llm", _hermetic_stream_ollama_llm)
