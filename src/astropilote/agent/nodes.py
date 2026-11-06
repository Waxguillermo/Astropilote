"""AstroPilote workflow nodes (`guard`, `refuse`, `agent`, `tools`, `compose`, `validate`).

Natural-language understanding (intent, location, relative dates, target filters, optics, multi-turn
follow-ups, and off-topic/injection detection) is handled by `route_query_with_llm`.
All spherical astronomy, timezone/twilight math, White-Night / Midnight-Sun regimes, parallel tool
execution, target culmination ranking, and post-generation citation/altitude checks are deterministic Python.
"""

from __future__ import annotations

import inspect
import json
import re
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from typing import Any

from astropilote.agent.citations import extract_inline_citations
from astropilote.agent.entity_resolver import (
    CONSTELLATION_NAMES,
    _angular_separation_deg,
    _check_out_of_corpus,
    _find_astrometric_companions,
    _find_unindexed_designations,
    _resolve_corpus_ids,
)
from astropilote.agent.harness import (
    check_harness_budgets,
    execute_tool_with_timeout,
    gate_tool_calls,
    resolve_harness_config,
)
from astropilote.agent.i18n import normalize_language, refusal_message
from astropilote.agent.planner import (
    _build_plan_items,
    _select_dynamic_sky_targets,
    _select_seasonal_catalog_targets,
    classify_night_darkness,
)
from astropilote.agent.prompts import build_compose_prompts, format_retrieved_passages
from astropilote.agent.schemas import (
    CATALOG_NAMES,
    AstroAnswer,
    Chunk,
    ChunkMetadata,
    ConversationTurn,
    EquipmentProfile,
    Intent,
    PlanItem,
    SensorFormat,
    ToolCallRecord,
)
from astropilote.agent.state import AgentMessage, AstroState
from astropilote.agent.validator import verify_astro_answer
from astropilote.config import settings
from astropilote.llm import (
    clear_last_router_backend,
    generate_grounded_answer,
    get_last_router_backend,
    plan_react_step_with_llm,
    route_query_with_llm,
    stream_grounded_answer,
)
from astropilote.tools import TOOLS_BY_NAME
from astropilote.tools.catalog import (
    get_object,
    normalize_object_id,
    resolve_object_record,
)
from astropilote.tools.world import rank_observing_nights
from astropilote.tracking.tracker import compute_cost_usd

_DEFAULT_GENERATE_GROUNDED_ANSWER = generate_grounded_answer
_DEFAULT_STREAM_GROUNDED_ANSWER = stream_grounded_answer
_DEFAULT_ROUTE_QUERY_WITH_LLM = route_query_with_llm
_DEFAULT_PLAN_REACT_STEP = plan_react_step_with_llm

# Protects Qdrant local SQLite / MPS tensor inference when multiple `search_catalog` calls run in a step
_SEARCH_CATALOG_LOCK = threading.Lock()

_VALID_SENSOR_FORMATS: tuple[SensorFormat, ...] = (
    "APS-C",
    "full-frame",
    "micro-4/3",
    "1-inch",
    "1/2.8-inch",
    "smartphone",
)

_APSC_EXPLICIT_RE = re.compile(r"\baps[- ]?c\b", flags=re.IGNORECASE)
_JSON_SECTION_BRACKET_RE = re.compile(
    r"\s*\[(?:night_ephemeris|night_weather|optics|location|verified_observation_schedule|"
    r"catalog_objects|multi_night_ranking|white_night_warning|Deterministic[^\]]*)\]",
    flags=re.IGNORECASE,
)

_FOLLOWUP_PREFIX_RE = re.compile(
    r"^\s*(?:et\b|and\b|what\s+about\b|how\s+about\b|what\s+if\b|et\s+si\b|quid\b|"
    r"avec\s+un|avec\s+une|with\s+a\b|with\s+my\b|sinon\b|ou\s+demain\b|or\s+tomorrow\b|"
    r"même\b|same\b)",
    flags=re.IGNORECASE,
)
_FOLLOWUP_PRONOUN_RE = re.compile(
    r"\b(?:cette\s+cible|cet\s+objet|ces\s+cibles|ces\s+objets|this\s+target|that\s+target|"
    r"that\s+object|see\s+it|shoot\s+it|image\s+it|frame\s+it|l['’]observer|"
    r"la\s+photographier|le\s+photographier)\b",
    flags=re.IGNORECASE,
)
_TEMPORAL_FOLLOWUP_RE = re.compile(
    r"\b(?:demain|tomorrow|ce\s+soir|tonight|week[- ]?end|prochain|next\s+(?:night|week|weekend)|"
    r"samedi|dimanche|vendredi|saturday|sunday|friday|m[ée]t[ée]o|weather|nuages?|clouds?)\b",
    flags=re.IGNORECASE,
)


def _normalize_conversation_history(
    raw_history: list[ConversationTurn] | list[dict[str, Any]] | None,
) -> list[ConversationTurn]:
    """Coerce raw conversation history entries into validated `ConversationTurn` models."""
    if not raw_history:
        return []
    turns: list[ConversationTurn] = []
    for item in raw_history[-6:]:
        if isinstance(item, ConversationTurn):
            turns.append(item)
        elif isinstance(item, dict):
            try:
                turns.append(ConversationTurn(**item))
            except (TypeError, ValueError):
                continue
    return turns


def _is_implicit_followup(question: str) -> bool:
    """Return True when `question` reads as a conversational follow-up to the previous turn."""
    q_clean = question.strip()
    if _FOLLOWUP_PREFIX_RE.search(q_clean) or _FOLLOWUP_PRONOUN_RE.search(q_clean):
        return True
    if len(q_clean) <= 48 and _TEMPORAL_FOLLOWUP_RE.search(q_clean):
        return True
    return False


def _extract_equipment_profile(state: AstroState) -> EquipmentProfile | None:
    """Coerce `state.get('equipment_profile')` into an active `EquipmentProfile` or return None."""
    raw = state.get("equipment_profile")
    if raw is None:
        return None
    if isinstance(raw, EquipmentProfile):
        return raw if raw.enabled else None
    if isinstance(raw, dict):
        try:
            prof = EquipmentProfile(**raw)
            return prof if prof.enabled else None
        except (TypeError, ValueError):
            return None
    return None


def _format_rig_summary(prof: EquipmentProfile) -> str:
    """Format a concise human-readable summary of `prof`."""
    f_ratio = round(prof.focal_length_mm / prof.aperture_mm, 1) if prof.aperture_mm else None
    f_str = f" (f/{f_ratio})" if f_ratio else ""
    optic_lbl = (
        prof.telescope_or_lens
        or prof.rig_name
        or f"{prof.aperture_mm:.0f}/{prof.focal_length_mm:.0f}mm"
    )
    cam_lbl = prof.camera_model or prof.sensor_format
    mount_map = {
        "equatorial_goto": "Equatorial GoTo mount",
        "star_tracker": "Portable Star Tracker",
        "altaz_goto": "Alt-Az GoTo mount",
        "tripod": "Fixed Tripod (untracked)",
    }
    mount_lbl = mount_map.get(prof.mount_type, prof.mount_type)
    if prof.has_autoguiding and prof.mount_type != "tripod":
        mount_lbl += " + Autoguiding"
    filt_lbl = f" · Filters: {', '.join(prof.filters)}" if prof.filters else ""
    bortle_lbl = f" · Sky Bortle {prof.bortle_class}" if prof.bortle_class else ""
    return (
        f"{optic_lbl} · {prof.focal_length_mm:.0f}mm{f_str} · "
        f"{cam_lbl} ({prof.sensor_format}) · {mount_lbl}{filt_lbl}{bortle_lbl}"
    )


def _call_router_compat(
    question: str,
    ref_date_iso: str,
    ref_weekday: str,
    mode_override: Any,
    history: list[ConversationTurn],
) -> tuple[dict[str, Any], int, int]:
    """Invoke `route_query_with_llm` positionally for the 4 core args so any positional mock works."""
    if history:
        try:
            sig = inspect.signature(route_query_with_llm)
            accepts_history = "history" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )
        except (TypeError, ValueError):
            accepts_history = False
        if accepts_history:
            return route_query_with_llm(
                question,
                ref_date_iso,
                ref_weekday,
                mode_override,
                history=history,
            )
    return route_query_with_llm(
        question,
        ref_date_iso,
        ref_weekday,
        mode_override,
    )


def _compute_router_step_cost(state: AstroState, extra_in: int, extra_out: int) -> float:
    """Compute exact USD cost for the router call (Haiku rate in API mode, $0.00 on Ollama)."""
    if settings.eval_offline_mode or (extra_in <= 0 and extra_out <= 0):
        return 0.0
    backend = get_last_router_backend()
    if backend is not None:
        return compute_cost_usd(backend, extra_in, extra_out)
    eff_mode = state.get("llm_mode_override") or settings.llm_mode
    if eff_mode == "api" and settings.anthropic_api_key.strip():
        return compute_cost_usd(settings.router_model, extra_in, extra_out)
    return 0.0


# ---------------------------------------------------------------------------
# Graph Nodes (`guard`, `refuse`, `agent`, `tools`, `compose`, `validate`)
# ---------------------------------------------------------------------------


def guard_node(state: AstroState) -> dict[str, Any]:
    """Node 1 (`guard` / Comprendre): Route the query via the structured LLM Router and verify date/horizon bounds."""
    q = state["question"]
    history = _normalize_conversation_history(state.get("conversation_history"))

    ref_today = settings.get_reference_date()
    override_d: date | None = None
    if state.get("date_override"):
        try:
            override_d = date.fromisoformat(str(state["date_override"]).strip()[:10])
        except ValueError:
            override_d = None

    anchor_date = override_d or ref_today
    clear_last_router_backend()
    routed, extra_in, extra_out = _call_router_compat(
        question=q,
        ref_date_iso=anchor_date.isoformat(),
        ref_weekday=anchor_date.strftime("%A"),
        mode_override=state.get("llm_mode_override"),
        history=history,
    )
    router_cost = _compute_router_step_cost(state, extra_in, extra_out)
    accumulated_cost = round(float(state.get("cost_usd", 0.0)) + router_cost, 6)

    req_type = str(routed.get("request_type") or "factual")
    lang = normalize_language(routed.get("language"))
    if req_type in ("prompt_injection", "off_topic"):
        intent = Intent(
            request_type=req_type,  # type: ignore[arg-type]
            language=lang,
            refusal_reason=str(routed.get("refusal_reason") or refusal_message(req_type, lang)),
        )
        return {
            "intent": intent,
            "tokens_in": state.get("tokens_in", 0) + extra_in,
            "tokens_out": state.get("tokens_out", 0) + extra_out,
            "cost_usd": accumulated_cost,
        }

    # Parse extracted start_date
    parsed_date: date | None = override_d
    raw_start = routed.get("start_date")
    if isinstance(raw_start, str) and raw_start.strip():
        try:
            parsed_date = date.fromisoformat(raw_start.strip()[:10])
        except ValueError:
            parsed_date = override_d

    raw_days = int(routed.get("forecast_days") or 1)
    weather_only = bool(routed.get("weather_only"))
    mentioned_targets = [
        str(t).strip() for t in (routed.get("mentioned_targets") or []) if str(t).strip()
    ]
    raw_loc = routed.get("location_name")
    clean_loc = str(raw_loc).strip() if isinstance(raw_loc, str) and raw_loc.strip() else None

    # Multi-turn follow-up inheritance ("Et demain soir ?", "Et avec un 200mm ?")
    if history and not bool(routed.get("is_fictional_location")) and _is_implicit_followup(q):
        last_turn = history[-1]
        has_temporal_cue = bool(_TEMPORAL_FOLLOWUP_RE.search(q))
        if clean_loc is None and (req_type == "session_plan" or has_temporal_cue):
            for prev_turn in reversed(history):
                if prev_turn.location_name:
                    clean_loc = prev_turn.location_name.strip()
                    req_type = "session_plan"
                    break
        if not mentioned_targets and not _resolve_corpus_ids(q):
            inherited_dsos = [s for s in last_turn.sources if not s.startswith(("CON_", "GUIDE_"))][
                :4
            ]
            if inherited_dsos and not has_temporal_cue:
                mentioned_targets = inherited_dsos
                if clean_loc is None and last_turn.location_name and req_type == "session_plan":
                    clean_loc = last_turn.location_name.strip()
            elif inherited_dsos and len(inherited_dsos) <= 2 and last_turn.location_name:
                mentioned_targets = inherited_dsos
        if (
            parsed_date is None
            and not has_temporal_cue
            and last_turn.obs_date
            and req_type == "session_plan"
        ):
            try:
                parsed_date = date.fromisoformat(last_turn.obs_date.strip()[:10])
            except ValueError:
                pass

    resolved_dsos = [
        oid
        for oid in _resolve_corpus_ids(q, mentioned_targets)
        if not oid.startswith(("CON_", "GUIDE_"))
    ]

    # Deterministic meteorological horizon check (> MAX_FORECAST_DAYS or past date < -1d) for weather-only queries
    if parsed_date is not None:
        delta_days = (parsed_date - ref_today).days
        if (delta_days > settings.max_forecast_days or delta_days < -1) and (
            weather_only and not resolved_dsos
        ):
            reason_msg = (
                refusal_message(
                    "past_weather_unavailable",
                    lang,
                    date=parsed_date.isoformat(),
                    days_ago=abs(delta_days),
                    max_days=settings.max_forecast_days,
                )
                if delta_days < -1
                else refusal_message(
                    "impossible_forecast",
                    lang,
                    date=parsed_date.isoformat(),
                    delta_days=delta_days,
                    max_days=settings.max_forecast_days,
                )
            )
            intent = Intent(
                request_type="impossible_forecast",
                language=lang,
                target_date=parsed_date,
                location_name=clean_loc,
                weather_only=True,
                refusal_reason=reason_msg,
            )
            return {
                "intent": intent,
                "tokens_in": state.get("tokens_in", 0) + extra_in,
                "tokens_out": state.get("tokens_out", 0) + extra_out,
                "cost_usd": accumulated_cost,
            }

    cat_filter = routed.get("catalog_filter")
    valid_cat = cat_filter if cat_filter in CATALOG_NAMES else None

    raw_const = routed.get("constellation")
    clean_const = (
        str(raw_const).strip().upper()
        if isinstance(raw_const, str) and str(raw_const).strip().upper() in CONSTELLATION_NAMES
        else None
    )

    prof = _extract_equipment_profile(state)

    aperture = (
        float(routed["aperture_mm"])
        if routed.get("aperture_mm") is not None
        else (
            prof.aperture_mm
            if prof is not None
            else (
                state["aperture_override"] if state.get("aperture_override") is not None else None
            )
        )
    )
    focal = (
        float(routed["focal_length_mm"])
        if routed.get("focal_length_mm") is not None
        else (
            prof.focal_length_mm
            if prof is not None
            else (
                state["focal_length_override"]
                if state.get("focal_length_override") is not None
                else None
            )
        )
    )

    raw_sensor = routed.get("sensor_format")
    if raw_sensor in _VALID_SENSOR_FORMATS:
        if prof is not None and raw_sensor == "APS-C" and not _APSC_EXPLICIT_RE.search(q):
            sensor_fmt: SensorFormat = prof.sensor_format
        else:
            sensor_fmt = raw_sensor  # type: ignore[assignment]
    elif prof is not None:
        sensor_fmt = prof.sensor_format
    else:
        sensor_fmt = "APS-C"

    mount_val = prof.mount_type if prof is not None else "equatorial_goto"
    autoguide_val = bool(prof.has_autoguiding) if prof is not None else False
    filters_val = list(prof.filters) if prof is not None else []
    rig_summary_val = _format_rig_summary(prof) if prof is not None else None

    mode_val = (
        routed.get("mode")
        if routed.get("mode") in ("visual", "astrophotography", "hybrid")
        else ("astrophotography" if prof is not None else "hybrid")
    )

    if req_type == "session_plan" and parsed_date is None:
        parsed_date = anchor_date

    resolved_with_companions = _resolve_corpus_ids(
        q,
        mentioned_targets,
        wants_companions=bool(routed.get("wants_companions")),
    )
    merged_targets = list(dict.fromkeys(mentioned_targets + resolved_with_companions))

    intent = Intent(
        request_type="session_plan" if req_type == "session_plan" else "factual",
        language=lang,
        mode=mode_val,  # type: ignore[arg-type]
        is_fictional_location=bool(routed.get("is_fictional_location")),
        location_name=clean_loc,
        target_date=parsed_date,
        forecast_days=max(1, min(raw_days, settings.max_forecast_days)),
        weather_only=weather_only,
        mentioned_targets=merged_targets,
        include_practical_guides=bool(routed.get("include_practical_guides")),
        search_query=str(routed["search_query"]).strip() if routed.get("search_query") else None,
        constellation=clean_const,
        catalog_filter=valid_cat,  # type: ignore[arg-type]
        object_type=str(routed["object_type"]) if routed.get("object_type") else None,
        max_magnitude=float(routed["max_magnitude"])
        if routed.get("max_magnitude") is not None
        else None,
        narrowband_only=bool(routed.get("narrowband_only")),
        aperture_mm=aperture,
        focal_length_mm=focal,
        sensor_format=sensor_fmt,
        mount_type=mount_val,
        has_autoguiding=autoguide_val,
        filters=filters_val,
        rig_summary=rig_summary_val,
    )
    return {
        "intent": intent,
        "tokens_in": state.get("tokens_in", 0) + extra_in,
        "tokens_out": state.get("tokens_out", 0) + extra_out,
        "cost_usd": accumulated_cost,
    }


def refuse_node(state: AstroState) -> dict[str, Any]:
    """Node (`refuse` / Refuser): Return a polite, source-free refusal without calling tools."""
    intent = state.get("intent")
    reason = (
        intent.refusal_reason
        if intent and intent.refusal_reason
        else refusal_message("generic_refusal", intent.language if intent else "en")
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


def _extract_geocode_coords(messages: list[AgentMessage]) -> tuple[float, float, bool]:
    """Read `(lat, lon, geocode_found)` from prior `geocode` tool messages."""
    for msg in messages:
        if msg.role == "tool" and msg.name == "geocode":
            try:
                data = json.loads(msg.content)
                if data.get("found"):
                    return float(data["lat"]), float(data["lon"]), True
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return 0.0, 0.0, False


def _extract_all_found_geocodes(messages: list[AgentMessage]) -> list[dict[str, Any]]:
    """Return all successful `geocode` tool payloads in execution order."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool" and msg.name == "geocode":
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict) and data.get("found"):
                    out.append(data)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return out


def _enrich_llm_react_call(
    raw_call: dict[str, Any],
    intent: Intent | None,
    get_object_args_fn: Any,
    step_idx: int,
    call_idx: int,
) -> dict[str, Any] | None:
    """Normalize LLM-proposed ReAct tool call arguments and attach user equipment profile defaults."""
    t_name = str(raw_call.get("name") or "").strip()
    if not t_name:
        return None
    raw_args = raw_call.get("arguments")
    if not isinstance(raw_args, dict):
        raw_args = raw_call.get("args") if isinstance(raw_call.get("args"), dict) else {}

    args = dict(raw_args)
    if t_name == "geocode":
        if "place_name" not in args and "place" in args:
            args["place_name"] = args.pop("place")
        if "place_name" not in args and "location" in args:
            args["place_name"] = args.pop("location")
        if not str(args.get("place_name") or "").strip():
            return None

    elif t_name in ("night", "weather"):
        if "lat" not in args and "latitude" in args:
            args["lat"] = args.pop("latitude")
        if "lon" not in args and "longitude" in args:
            args["lon"] = args.pop("longitude")
        if "obs_date" not in args and "date" in args:
            args["obs_date"] = args.pop("date")
        if "obs_date" not in args:
            args["obs_date"] = (
                intent.target_date or settings.get_reference_date()
                if intent
                else settings.get_reference_date()
            ).isoformat()
        if "days" not in args and intent is not None:
            args["days"] = max(1, min(intent.forecast_days, settings.max_forecast_days))
        if args.get("lat") is None or args.get("lon") is None:
            return None

    elif t_name == "get_object":
        oid = str(args.get("object_id") or args.get("id") or "").strip()
        if not oid:
            return None
        norm_oid = normalize_object_id(oid)
        merged = get_object_args_fn(norm_oid)
        for k, v in args.items():
            if k not in ("id", "object_id") and v is not None:
                merged[k] = v
        args = merged

    elif t_name == "visibility":
        if "object_ids" not in args and "ids" in args:
            args["object_ids"] = args.pop("ids")
        if "lat" not in args and "latitude" in args:
            args["lat"] = args.pop("latitude")
        if "lon" not in args and "longitude" in args:
            args["lon"] = args.pop("longitude")
        if "obs_date" not in args and "date" in args:
            args["obs_date"] = args.pop("date")
        if "obs_date" not in args:
            args["obs_date"] = (
                intent.target_date or settings.get_reference_date()
                if intent
                else settings.get_reference_date()
            ).isoformat()
        raw_oids = args.get("object_ids")
        if isinstance(raw_oids, str):
            raw_oids = [raw_oids]
        if not isinstance(raw_oids, list) or not raw_oids:
            return None
        args["object_ids"] = [normalize_object_id(str(x)) for x in raw_oids if str(x).strip()][:8]
        if not args["object_ids"] or args.get("lat") is None or args.get("lon") is None:
            return None
        if args.get("focal_length_mm") is None and intent and intent.focal_length_mm is not None:
            args["focal_length_mm"] = intent.focal_length_mm

    elif t_name == "search_catalog":
        q_str = str(args.get("query") or "").strip()
        if not q_str:
            return None
        args["query"] = q_str
        args.setdefault("top_k", 5)

    return {
        "name": t_name,
        "args": args,
        "id": f"call_react_{step_idx}_{call_idx}_{t_name}",
    }


def _build_react_observation_ledger(
    state: AstroState,
    suggested_sky_candidates: list[str],
) -> dict[str, Any]:
    """Assemble a compact structured Observation Ledger for `plan_react_step_with_llm`."""
    intent = state.get("intent")
    geocoded_sites: list[dict[str, Any]] = []
    night_obs: list[dict[str, Any]] = []
    weather_obs: list[dict[str, Any]] = []
    vis_obs: dict[str, Any] = {}
    tool_errors: list[str] = []

    for msg in state.get("messages", []):
        if msg.role != "tool":
            continue
        try:
            payload = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("ok") is False:
            tool_errors.append(str(payload.get("error") or msg.name))
            continue
        if msg.name == "geocode" and isinstance(payload, dict):
            geocoded_sites.append(
                {
                    "place_name": payload.get("place_name"),
                    "found": payload.get("found"),
                    "lat": payload.get("lat"),
                    "lon": payload.get("lon"),
                }
            )
        elif msg.name == "night" and isinstance(payload, dict):
            night_obs.append(
                {
                    "date": payload.get("date"),
                    "lat": payload.get("lat"),
                    "lon": payload.get("lon"),
                    "has_astronomical_darkness": payload.get("has_astronomical_darkness"),
                    "min_sun_altitude_deg": payload.get("min_sun_altitude_deg"),
                    "astronomical_dark_start": payload.get("astronomical_dark_start"),
                    "astronomical_dark_end": payload.get("astronomical_dark_end"),
                    "moon_illumination_pct": payload.get("moon_illumination_pct"),
                }
            )
        elif msg.name == "weather" and isinstance(payload, dict):
            weather_obs.append(
                {
                    "date": payload.get("date"),
                    "lat": payload.get("lat"),
                    "lon": payload.get("lon"),
                    "available": payload.get("available"),
                    "cloud_cover_pct": payload.get("cloud_cover_pct"),
                    "clear_hours": payload.get("clear_hours"),
                }
            )
        elif msg.name == "visibility" and isinstance(payload, dict):
            for oid, odata in (payload.get("objects") or {}).items():
                vis_obs[oid] = {
                    "observable": odata.get("observable"),
                    "max_altitude_deg": odata.get("max_altitude_deg"),
                }

    retrieved_doc_ids = list(dict.fromkeys(c.doc_id for c in state.get("retrieved", [])))
    return {
        "step": state.get("steps", 0),
        "question": state.get("question", ""),
        "intent": (
            intent.model_dump(mode="json", exclude_none=True) if intent is not None else {}
        ),
        "executed_tool_signatures": list(state.get("seen_tool_signatures", [])),
        "geocoded_sites": geocoded_sites,
        "night_observations": night_obs,
        "weather_observations": weather_obs,
        "visibility_observations": vis_obs,
        "retrieved_doc_ids": retrieved_doc_ids,
        "suggested_sky_candidates": suggested_sky_candidates,
        "tool_errors": tool_errors,
    }


def agent_node(state: AstroState) -> dict[str, Any]:
    """Node 2 (`agent` / ReAct + Harness): Hybrid LLM `Reason -> Act` loop gated by Harness budgets & dedup."""
    steps = int(state.get("steps", 0))
    q = state["question"]
    intent = state.get("intent")
    exec_mode = state.get("execution_mode") or "agent"
    harness_cfg = resolve_harness_config(state)
    harness_events = list(state.get("harness_events", []))
    react_thoughts = list(state.get("react_thoughts", []))

    enforce_strict = (
        settings.strict_corpus_guard
        if state.get("strict_corpus_guard") is None
        else bool(state.get("strict_corpus_guard"))
    )

    # 1. Pre-ReAct guardrail short-circuit on non-existent catalog numbers (M111) or fictional places (Atlantis)
    if _check_out_of_corpus(q, strict=enforce_strict) is not None or (
        intent is not None and intent.is_fictional_location
    ):
        ai_msg = AgentMessage(role="assistant", content="", tool_calls=[])
        return {"messages": state.get("messages", []) + [ai_msg], "steps": steps + 1}

    # 2. Pre-step Harness circuit-breaker check (abort immediately on consecutive tool errors, token, or cost caps)
    within_budget, tripped_kind, tripped_msg = check_harness_budgets(state, harness_cfg)
    if not within_budget and tripped_kind in (
        "consecutive_tool_errors",
        "max_tokens",
        "max_cost_usd",
    ):
        if tripped_msg:
            harness_events.append(tripped_msg)
        ai_msg = AgentMessage(role="assistant", content="", tool_calls=[])
        return {
            "messages": state.get("messages", []) + [ai_msg],
            "steps": steps,
            "harness_events": harness_events,
            "harness_budget_tripped": tripped_kind,
        }

    baseline_calls: list[dict[str, Any]] = []
    baseline_thought = ""
    suggested_sky_candidates: list[str] = []

    retrieval_query = (intent.search_query if intent and intent.search_query else q).strip()
    focal_mm = intent.focal_length_mm if intent else None
    aperture_mm = intent.aperture_mm if intent else None
    sensor_fmt = intent.sensor_format if intent else "APS-C"
    mount_type = intent.mount_type if intent else "equatorial_goto"
    has_autoguiding = bool(intent.has_autoguiding) if intent else False
    filters_list = list(intent.filters) if intent else []

    def _get_object_args(oid: str) -> dict[str, Any]:
        args: dict[str, Any] = {
            "object_id": oid,
            "focal_length_mm": focal_mm,
            "sensor_format": sensor_fmt,
        }
        if aperture_mm is not None:
            args["aperture_mm"] = aperture_mm
        if mount_type != "equatorial_goto" or has_autoguiding or filters_list:
            args["mount_type"] = mount_type
            args["has_autoguiding"] = has_autoguiding
            args["filters"] = filters_list
        return args

    is_located_session = bool(
        intent is not None
        and not intent.is_fictional_location
        and intent.location_name
        and (intent.request_type == "session_plan" or exec_mode != "rag")
    )

    messages = state.get("messages", [])
    ran_tools = {m.name for m in messages if m.role == "tool" and m.name}

    if exec_mode == "rag" and not is_located_session:
        if steps == 0:
            baseline_thought = "RAG-only mode: retrieving top hybrid passages from corpus."
            baseline_calls.append(
                {
                    "name": "search_catalog",
                    "args": {"query": retrieval_query, "top_k": settings.final_top_k},
                    "id": "call_rag_search",
                }
            )
    elif steps == 0:
        resolved_ids = _resolve_corpus_ids(
            q,
            intent.mentioned_targets if intent else None,
        )
        if is_located_session and intent and intent.location_name:
            baseline_thought = (
                f"Geocoding observation site '{intent.location_name}' and retrieving initial catalog context."
            )
            baseline_calls.append(
                {
                    "name": "geocode",
                    "args": {"place_name": intent.location_name},
                    "id": "call_geocode",
                }
            )
        else:
            baseline_thought = "Retrieving catalog astrometry and hybrid corpus passages."

        non_guide_ids = [oid for oid in resolved_ids if not oid.startswith("GUIDE_")]
        dso_only_ids = [oid for oid in non_guide_ids if not oid.startswith("CON_")]
        if (
            not is_located_session
            and len(dso_only_ids) < 4
            and intent
            and (
                intent.request_type == "session_plan"
                or intent.object_type
                or intent.max_magnitude
                or intent.constellation
                or intent.catalog_filter
                or intent.narrowband_only
            )
        ):
            obs_d = intent.target_date or settings.get_reference_date()
            seasonal_ids = _select_seasonal_catalog_targets(obs_d, intent, limit=5)
            non_guide_ids = list(dict.fromkeys(non_guide_ids + seasonal_ids))

        for oid in non_guide_ids[:6]:
            baseline_calls.append(
                {
                    "name": "get_object",
                    "args": _get_object_args(oid),
                    "id": f"call_get_{oid}",
                }
            )

        search_args: dict[str, Any] = {
            "query": retrieval_query,
            "top_k": 6,
        }
        if intent:
            if intent.catalog_filter:
                search_args["catalog"] = intent.catalog_filter
            if intent.object_type:
                search_args["object_type"] = intent.object_type
            if intent.constellation:
                search_args["constellation"] = intent.constellation
            if intent.max_magnitude is not None:
                search_args["max_magnitude"] = intent.max_magnitude
            if intent.narrowband_only:
                search_args["narrowband_only"] = True
            if non_guide_ids and not intent.include_practical_guides:
                search_args["exclude_guides"] = True

        baseline_calls.append(
            {
                "name": "search_catalog",
                "args": search_args,
                "id": "call_search",
            }
        )

        if intent and (intent.include_practical_guides or is_located_session):
            baseline_calls.append(
                {
                    "name": "search_catalog",
                    "args": {"query": retrieval_query, "catalog": "Guide", "top_k": 3},
                    "id": "call_search_guides",
                }
            )

    elif is_located_session and intent is not None:
        found_geocodes = _extract_all_found_geocodes(messages)
        lat, lon, geocode_ok = _extract_geocode_coords(messages)
        if geocode_ok and ("night" not in ran_tools or "weather" not in ran_tools):
            obs_date_str = (intent.target_date or settings.get_reference_date()).isoformat()
            span = max(1, min(intent.forecast_days, settings.max_forecast_days))
            baseline_thought = (
                f"Computing twilight darkness ephemeris and hourly cloud cover for {len(found_geocodes)} geocoded site(s)."
            )
            for g_idx, g_info in enumerate(found_geocodes[:3]):
                g_lat = float(g_info["lat"])
                g_lon = float(g_info["lon"])
                g_place = str(g_info.get("short_name") or intent.location_name or "")
                baseline_calls.extend(
                    [
                        {
                            "name": "night",
                            "args": {
                                "lat": g_lat,
                                "lon": g_lon,
                                "obs_date": obs_date_str,
                                "days": span,
                            },
                            "id": f"call_night_{g_idx}",
                        },
                        {
                            "name": "weather",
                            "args": {
                                "lat": g_lat,
                                "lon": g_lon,
                                "obs_date": obs_date_str,
                                "days": span,
                                "place_name": g_place,
                            },
                            "id": f"call_weather_{g_idx}",
                        },
                    ]
                )
        elif geocode_ok and "night" in ran_tools and "visibility" not in ran_tools:
            daily_nights: list[dict[str, Any]] = []
            daily_forecasts: list[dict[str, Any]] = []
            for msg in messages:
                if msg.role == "tool":
                    try:
                        payload = json.loads(msg.content)
                        if msg.name == "night" and isinstance(payload, dict) and not daily_nights:
                            daily_nights = list(payload.get("daily_nights") or [payload])
                        elif (
                            msg.name == "weather"
                            and isinstance(payload, dict)
                            and not daily_forecasts
                        ):
                            daily_forecasts = list(payload.get("daily_forecasts") or [payload])
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass

            best_date = (intent.target_date or settings.get_reference_date()).isoformat()
            active_night_rec: dict[str, Any] = daily_nights[0] if daily_nights else {}
            if len(daily_nights) > 1:
                ranked = rank_observing_nights(daily_nights, daily_forecasts)
                if ranked:
                    best_date = str(ranked[0]["date"])
                    active_night_rec = next(
                        (n for n in daily_nights if n.get("date") == best_date),
                        daily_nights[0],
                    )
            elif daily_nights:
                best_date = str(daily_nights[0].get("date") or best_date)

            regime, eff_dark_s, eff_dark_e = classify_night_darkness(active_night_rec)
            moon_pct = int(active_night_rec.get("moon_illumination_pct") or 50)

            explicit_dsos = [
                oid
                for oid in _resolve_corpus_ids(q, intent.mentioned_targets)
                if not oid.startswith(("CON_", "GUIDE_"))
            ]

            if regime == "bright_twilight_or_daylight":
                candidate_ids = explicit_dsos[:6]
            elif len(explicit_dsos) >= 5:
                candidate_ids = explicit_dsos[:6]
            else:
                dyn_ids = _select_dynamic_sky_targets(
                    lat=lat,
                    lon=lon,
                    obs_date_str=best_date,
                    dark_start=eff_dark_s or "22:30",
                    dark_end=eff_dark_e or "03:30",
                    moon_pct=moon_pct,
                    intent=intent,
                    limit=6,
                )
                candidate_ids = list(dict.fromkeys(explicit_dsos + dyn_ids))[:6]

            suggested_sky_candidates = list(candidate_ids)
            baseline_thought = (
                f"Evaluating hourly altitude curves and framing for candidate targets {candidate_ids}."
            )
            existing_doc_ids = {c.doc_id for c in state.get("retrieved", [])}
            for oid in candidate_ids:
                if oid not in existing_doc_ids:
                    baseline_calls.append(
                        {
                            "name": "get_object",
                            "args": _get_object_args(oid),
                            "id": f"call_get_dyn_{oid}",
                        }
                    )

            if candidate_ids:
                baseline_calls.append(
                    {
                        "name": "visibility",
                        "args": {
                            "object_ids": candidate_ids,
                            "lat": lat,
                            "lon": lon,
                            "obs_date": best_date,
                            "focal_length_mm": focal_mm,
                        },
                        "id": "call_visibility",
                    }
                )

    # 3. Hybrid LLM ReAct step (`Reason -> Act`): active in live mode or when explicitly patched in tests
    extra_in = 0
    extra_out = 0
    react_cost = 0.0
    proposed_calls = list(baseline_calls)
    chosen_thought = baseline_thought

    can_plan_more_tools = (
        steps < harness_cfg.max_steps
        and len(state.get("tool_trace", [])) < harness_cfg.max_tool_calls
    ) or plan_react_step_with_llm is not _DEFAULT_PLAN_REACT_STEP

    use_llm_react = (
        exec_mode != "rag"
        and can_plan_more_tools
        and (
            plan_react_step_with_llm is not _DEFAULT_PLAN_REACT_STEP
            or (
                not settings.eval_offline_mode
                and route_query_with_llm is _DEFAULT_ROUTE_QUERY_WITH_LLM
            )
        )
    )

    if use_llm_react:
        ledger = _build_react_observation_ledger(state, suggested_sky_candidates)
        try:
            react_plan, extra_in, extra_out, react_backend = plan_react_step_with_llm(
                ledger,
                mode_override=state.get("llm_mode_override"),
            )
            if not settings.eval_offline_mode and react_backend:
                react_cost = compute_cost_usd(react_backend, extra_in, extra_out)
            llm_thought = str(react_plan.get("thought") or "").strip()
            if llm_thought:
                chosen_thought = llm_thought

            raw_llm_calls = react_plan.get("tool_calls") or []
            enriched_llm_calls: list[dict[str, Any]] = []
            if isinstance(raw_llm_calls, list) and not bool(react_plan.get("is_complete")):
                for c_idx, rc in enumerate(raw_llm_calls[:6]):
                    if isinstance(rc, dict):
                        norm_call = _enrich_llm_react_call(
                            rc, intent, _get_object_args, steps, c_idx
                        )
                        if norm_call is not None:
                            enriched_llm_calls.append(norm_call)

            if enriched_llm_calls:
                # Merge LLM-driven adaptive calls first, supplemented by any missing essential ephemeris calls
                proposed_calls = enriched_llm_calls + baseline_calls
            elif bool(react_plan.get("is_complete")) and not baseline_calls:
                proposed_calls = []
        except Exception as exc:
            harness_events.append(f"react_planner_fallback:{type(exc).__name__}")

    updated_tokens_in = int(state.get("tokens_in", 0)) + extra_in
    updated_tokens_out = int(state.get("tokens_out", 0)) + extra_out
    updated_cost = round(float(state.get("cost_usd", 0.0)) + react_cost, 6)

    state_for_gate: AstroState = {
        **state,
        "tokens_in": updated_tokens_in,
        "tokens_out": updated_tokens_out,
        "cost_usd": updated_cost,
    }

    # 4. Agent Harness pre-execution gate (anti-duplicate signature filter + hard budgets)
    gate_res = gate_tool_calls(
        proposed_calls=proposed_calls,
        state=state_for_gate,
        config=harness_cfg,
        allowed_tools=set(TOOLS_BY_NAME.keys()),
    )
    harness_events.extend(gate_res.new_events)
    if chosen_thought and (gate_res.approved_calls or steps == 0):
        react_thoughts.append(chosen_thought)

    ai_msg = AgentMessage(
        role="assistant",
        content=chosen_thought,
        tool_calls=gate_res.approved_calls,
    )
    return {
        "messages": messages + [ai_msg],
        "steps": steps + 1,
        "tokens_in": updated_tokens_in,
        "tokens_out": updated_tokens_out,
        "cost_usd": updated_cost,
        "seen_tool_signatures": gate_res.seen_signatures,
        "harness_events": harness_events,
        "harness_budget_tripped": gate_res.budget_tripped or state.get("harness_budget_tripped"),
        "react_thoughts": react_thoughts,
    }


def tools_node(state: AstroState) -> dict[str, Any]:
    """Node 3 (`tools` / Harness Executor): Execute approved tool calls in parallel under per-tool timeouts."""
    messages = state.get("messages", [])
    if not messages:
        return {}
    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []

    valid_calls: list[tuple[str, dict[str, Any], str, Any]] = []
    for tc in tool_calls:
        t_name = tc["name"]
        tool_fn = TOOLS_BY_NAME.get(t_name)
        if tool_fn is not None:
            valid_calls.append((t_name, tc.get("args", {}), tc.get("id", t_name), tool_fn))

    if not valid_calls:
        return {}

    harness_cfg = resolve_harness_config(state)
    timeout_s = harness_cfg.tool_timeout_s

    # Detect whether a prior `night` tool call already reported absence of true astronomical darkness
    had_white_night = False
    for msg in messages:
        if msg.role == "tool" and msg.name == "night":
            try:
                prev_night = json.loads(msg.content)
                if (
                    isinstance(prev_night, dict)
                    and prev_night.get("has_astronomical_darkness") is False
                ):
                    had_white_night = True
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

    # Parallel execution across independent tool calls with strict per-tool wall-clock timeouts
    ordered_outputs: list[tuple[str, dict[str, Any], str, Any, float, str]]
    if len(valid_calls) == 1:
        t_name, t_args, t_id, tool_fn = valid_calls[0]
        outcome = execute_tool_with_timeout(
            t_name, t_args, tool_fn, timeout_s=timeout_s, search_lock=_SEARCH_CATALOG_LOCK
        )
        ordered_outputs = [
            (t_name, t_args, t_id, outcome.output, outcome.duration_s, outcome.status)
        ]
    else:
        max_workers = min(4, len(valid_calls))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(
                    execute_tool_with_timeout,
                    t_name,
                    t_args,
                    tool_fn,
                    timeout_s,
                    _SEARCH_CATALOG_LOCK,
                )
                for t_name, t_args, _, tool_fn in valid_calls
            ]
            ordered_outputs = []
            for (t_name, t_args, t_id, _), fut in zip(valid_calls, futures, strict=True):
                outcome = fut.result()
                ordered_outputs.append(
                    (t_name, t_args, t_id, outcome.output, outcome.duration_s, outcome.status)
                )

    new_tool_messages: list[AgentMessage] = []
    trace: list[ToolCallRecord] = list(state.get("tool_trace", []))
    retrieved: list[Chunk] = list(state.get("retrieved", []))
    existing_chunk_ids = {c.chunk_id for c in retrieved}
    existing_doc_ids = {c.doc_id for c in retrieved}
    calc_alts: dict[str, list[float]] = dict(state.get("calculated_altitudes", {}))
    night_win = state.get("night_window")
    consec_errors = int(state.get("consecutive_tool_errors", 0))
    harness_events = list(state.get("harness_events", []))

    for t_name, t_args, t_id, result, dur, status in ordered_outputs:
        if status == "ok":
            consec_errors = 0
        else:
            consec_errors += 1
            harness_events.append(f"tool_{status}:{t_name} ({dur:.3f}s)")

        trace.append(
            ToolCallRecord(name=t_name, arguments=t_args, duration_s=dur, status=status)  # type: ignore[arg-type]
        )
        new_tool_messages.append(
            AgentMessage(
                role="tool",
                content=json.dumps(result, ensure_ascii=False),
                name=t_name,
                tool_call_id=t_id,
            )
        )
        if status != "ok":
            continue

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
            regime, eff_s, eff_e = classify_night_darkness(result)
            if regime == "bright_twilight_or_daylight":
                had_white_night = True
                night_win = None
            elif regime == "nautical_twilight":
                had_white_night = True
                night_win = (eff_s, eff_e) if (eff_s and eff_e) else None
            else:
                night_win = (
                    str(result["astronomical_dark_start"]),
                    str(result["astronomical_dark_end"]),
                )

        elif t_name == "visibility" and isinstance(result, dict):
            nw = result.get("night_window") or {}
            if not had_white_night and nw.get("dark_start") and nw.get("dark_end"):
                night_win = (str(nw["dark_start"]), str(nw["dark_end"]))
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
        "consecutive_tool_errors": consec_errors,
        "harness_events": harness_events,
    }


@dataclass
class _ComposePrepared:
    """Prepared deterministic payload and prompts shared between `compose_node` and `stream_compose_node`."""

    sys_prompt: str
    user_prompt: str
    plan_items: list[PlanItem]
    conditions_str: str | None
    retrieved_doc_ids: list[str]
    effective_night_window: tuple[str, str] | None


def _prepare_compose_context(
    state: AstroState,
) -> AstroAnswer | _ComposePrepared:
    """Build deterministic JSON, schedule, conditions, and full-chunk prompts (or return an early abstention)."""
    q = state["question"]
    retrieved = state.get("retrieved", [])
    intent = state.get("intent")
    lang = intent.language if intent else "en"

    enforce_strict = (
        settings.strict_corpus_guard
        if state.get("strict_corpus_guard") is None
        else bool(state.get("strict_corpus_guard"))
    )
    ooc_reason = _check_out_of_corpus(q, strict=enforce_strict, language=lang)
    if ooc_reason is not None or (enforce_strict and not retrieved):
        return AstroAnswer(
            answer=ooc_reason or refusal_message("no_documents", lang),
            plan=[],
            conditions=None,
            sources=[],
            abstained=True,
            verified=True,
        )

    if intent and intent.is_fictional_location:
        bad_place = intent.location_name or refusal_message("unknown_place", lang)
        return AstroAnswer(
            answer=refusal_message("fictional_location", lang, place=bad_place),
            plan=[],
            conditions=None,
            sources=[],
            abstained=True,
            verified=True,
        )

    night_data: dict[str, Any] = {}
    vis_data: dict[str, Any] = {}
    weather_data: dict[str, Any] = {}
    geocode_data: dict[str, Any] = {}
    failed_geocode_data: dict[str, Any] = {}
    found_geocodes: list[dict[str, Any]] = []
    seen_geo_coords: set[tuple[float, float]] = set()
    all_nights: list[dict[str, Any]] = []
    all_weathers: list[dict[str, Any]] = []
    object_records: dict[str, dict[str, Any]] = {}

    for msg in state.get("messages", []):
        if msg.role == "tool":
            try:
                payload = json.loads(msg.content)
                if isinstance(payload, dict) and payload.get("ok") is False:
                    continue
                if msg.name == "night" and isinstance(payload, dict):
                    if not night_data:
                        night_data = payload
                    all_nights.append(payload)
                elif msg.name == "visibility" and isinstance(payload, dict):
                    if not vis_data:
                        vis_data = payload
                elif msg.name == "weather" and isinstance(payload, dict):
                    if not weather_data:
                        weather_data = payload
                    all_weathers.append(payload)
                elif msg.name == "geocode" and isinstance(payload, dict):
                    if payload.get("found"):
                        if not geocode_data:
                            geocode_data = payload
                        gk = (
                            round(float(payload.get("lat") or 0.0), 3),
                            round(float(payload.get("lon") or 0.0), 3),
                        )
                        if gk not in seen_geo_coords:
                            seen_geo_coords.add(gk)
                            found_geocodes.append(payload)
                    else:
                        failed_geocode_data = payload
                elif (
                    msg.name == "get_object" and isinstance(payload, dict) and payload.get("found")
                ):
                    object_records[str(payload["doc_id"])] = payload
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

    if not geocode_data and failed_geocode_data:
        geocode_data = failed_geocode_data

    if geocode_data and geocode_data.get("found") is False:
        bad_loc = (
            (intent.location_name if intent else None)
            or geocode_data.get("place_name")
            or refusal_message("unknown_place", lang)
        )
        return AstroAnswer(
            answer=refusal_message("geocode_failed", lang, place=bad_loc),
            plan=[],
            conditions=None,
            sources=[],
            abstained=True,
            verified=True,
        )

    is_session_plan = bool(intent and intent.request_type == "session_plan")
    has_real_location = bool(geocode_data.get("found"))

    daily_nights = list(night_data.get("daily_nights") or ([night_data] if night_data else []))
    daily_forecasts = list(
        weather_data.get("daily_forecasts") or ([weather_data] if weather_data else [])
    )
    is_multi_night = len(daily_nights) > 1
    ranked_nights = rank_observing_nights(daily_nights, daily_forecasts) if is_multi_night else []
    best_night = ranked_nights[0] if ranked_nights else None

    active_night = (
        next((n for n in daily_nights if n.get("date") == best_night["date"]), daily_nights[0])
        if (best_night and daily_nights)
        else night_data
    )
    active_weather = (
        next(
            (w for w in daily_forecasts if w.get("date") == best_night["date"]),
            daily_forecasts[0] if daily_forecasts else {},
        )
        if best_night
        else weather_data
    )

    retrieved_doc_ids = list(dict.fromkeys(c.doc_id for c in retrieved))
    resolved_mentions = _resolve_corpus_ids(q, intent.mentioned_targets if intent else None)
    priority_dsos = [
        d
        for d in resolved_mentions
        if d in retrieved_doc_ids and not d.startswith(("CON_", "GUIDE_"))
    ]

    plan_items: list[PlanItem] = []
    conditions_str: str | None = None
    effective_night_window: tuple[str, str] | None = state.get("night_window")
    white_night_warning: str | None = None
    no_darkness_impossible = False

    if is_session_plan and has_real_location and active_night:
        site_label = str(
            intent.location_name
            if intent and intent.location_name
            else geocode_data.get("short_name", "")
        )
        tz_lbl = str(active_night.get("tz_label") or "local")
        moon_pct = int(active_night.get("moon_illumination_pct") or 0)
        moon_phase = str(active_night.get("moon_phase_name") or "")
        w_avail = bool(active_weather.get("available")) if active_weather else False
        cloud_note = (
            f"Cloud cover {active_weather.get('cloud_cover_pct')}%"
            if w_avail
            else f"Weather unavailable (> {settings.max_forecast_days} days forecast horizon)"
        )
        moon_note = f"{moon_pct}% {moon_phase}".strip()
        date_lbl = str(active_night.get("date") or "")
        min_sun_alt = float(active_night.get("min_sun_altitude_deg", -25.0))

        regime, eff_dark_start, eff_dark_end = classify_night_darkness(active_night)

        if regime == "bright_twilight_or_daylight":
            no_darkness_impossible = True
            plan_items = []
            effective_night_window = None
            phenomenon = (
                f"Midnight Sun / continuous daylight (min Sun altitude {min_sun_alt:+.1f}°)"
                if min_sun_alt >= -0.8
                else f"Continuous civil/bright twilight — no astronomical darkness (min Sun altitude {min_sun_alt:.1f}°)"
            )
            if is_multi_night and best_night:
                conditions_str = (
                    f"{site_label} ({daily_nights[0]['date']} to {daily_nights[-1]['date']}): "
                    f"{phenomenon} | {cloud_note} | Moon {moon_note}"
                )
            else:
                conditions_str = (
                    f"{site_label} on {date_lbl}: {phenomenon} | Moon {moon_note} | {cloud_note}"
                )
            white_night_warning = (
                f"Deep-sky observation is physically impossible at {site_label} on {date_lbl}: "
                f"the Sun only reaches a minimum altitude of {min_sun_alt:+.1f}° (never dropping below -12° "
                "for dark nautical or -18° for astronomical night), so solar skyscatter washes out stars, "
                "nebulae, and galaxies all night."
            )

        elif regime == "nautical_twilight":
            tw_s = eff_dark_start or "23:30"
            tw_e = eff_dark_end or "02:30"
            effective_night_window = (tw_s, tw_e)
            if is_multi_night and best_night:
                conditions_str = (
                    f"{site_label} ({daily_nights[0]['date']} to {daily_nights[-1]['date']}): "
                    f"Best night {best_night['weekday']} {best_night['date']} — "
                    f"Nautical twilight only (no full astronomical darkness, min Sun {min_sun_alt:.1f}°, "
                    f"twilight {tw_s}–{tw_e} {tz_lbl}, {cloud_note}, Moon {moon_note})"
                )
            else:
                conditions_str = (
                    f"{site_label} on {date_lbl}: Nautical twilight only (no full astronomical darkness, "
                    f"min Sun {min_sun_alt:.1f}°, darkest window {tw_s}–{tw_e} {tz_lbl}) | "
                    f"Moon {moon_note} | {cloud_note}"
                )
            white_night_warning = (
                f"Summer white night at {site_label} on {date_lbl}: the Sun reaches {min_sun_alt:.1f}° "
                f"(above -18°), so true astronomical darkness never occurs. Only nautical twilight ({tw_s}–{tw_e} {tz_lbl}) "
                "is available around solar midnight — restrict observing/imaging to bright star clusters, double stars, "
                "planetary nebulae, or narrowband Hα/OIII targets."
            )
            plan_items = _build_plan_items(
                vis_data=vis_data,
                dark_start=tw_s,
                dark_end=tw_e,
                retrieved_doc_ids=retrieved_doc_ids,
                priority_ids=priority_dsos + list(object_records.keys()),
                object_records=object_records,
                intent=intent,
                twilight_only=True,
            )

        else:
            dark_start = eff_dark_start or "21:00"
            dark_end = eff_dark_end or "05:30"
            effective_night_window = (dark_start, dark_end)
            if is_multi_night and best_night:
                conditions_str = (
                    f"{site_label} ({daily_nights[0]['date']} to {daily_nights[-1]['date']}): "
                    f"Best night {best_night['weekday']} {best_night['date']} "
                    f"({cloud_note}, dark {dark_start}–{dark_end} {tz_lbl}, Moon {moon_note})"
                )
            else:
                conditions_str = (
                    f"{site_label} on {date_lbl}: Astronomical darkness {dark_start} to {dark_end} ({tz_lbl}) | "
                    f"Moon {moon_note} | {cloud_note}"
                )
            plan_items = _build_plan_items(
                vis_data=vis_data,
                dark_start=dark_start,
                dark_end=dark_end,
                retrieved_doc_ids=retrieved_doc_ids,
                priority_ids=priority_dsos + list(object_records.keys()),
                object_records=object_records,
                intent=intent,
                twilight_only=False,
            )

    scheduled_ids = [pi.object_id for pi in plan_items]
    con_ids = [d for d in resolved_mentions if d in retrieved_doc_ids and d.startswith("CON_")]
    guide_ids = [d for d in retrieved_doc_ids if d.startswith("GUIDE_")][:2]
    candidate_cite_ids = list(
        dict.fromkeys(
            priority_dsos
            + scheduled_ids
            + list(object_records.keys())
            + con_ids
            + guide_ids
            + retrieved_doc_ids
        )
    )[:8]

    location_payload: dict[str, Any]
    if has_real_location:
        location_payload = {
            "provided": True,
            "requested_name": intent.location_name if intent else None,
            "resolved_place_name": geocode_data.get("place_name"),
            "lat": geocode_data.get("lat"),
            "lon": geocode_data.get("lon"),
        }
    else:
        location_payload = {
            "provided": False,
            "note": (
                "No geographic location was specified by the user. Do NOT assume or invent any default city. "
                "If local sky/weather/timing matters for this question, present seasonal/catalog target recommendations "
                "and invite the user to share their city, region, or GPS coordinates for exact local cloud cover, "
                "darkness hours, and target altitudes."
            ),
        }

    unobservable_targets: list[dict[str, Any]] = []
    for oid, odata in (vis_data.get("objects") or {}).items():
        if not odata.get("observable"):
            unobservable_targets.append(
                {
                    "object_id": oid,
                    "title": odata.get("title", oid),
                    "observable": False,
                    "max_altitude_deg": odata.get("max_altitude_deg"),
                    "min_required_altitude_deg": settings.min_observable_altitude_deg,
                }
            )

    prof = _extract_equipment_profile(state)
    eff_focal = intent.focal_length_mm if intent else None
    eff_aperture = intent.aperture_mm if intent else None
    eff_sensor = intent.sensor_format if intent else "APS-C"
    eff_mount = intent.mount_type if intent else "equatorial_goto"
    eff_autoguide = bool(intent.has_autoguiding) if intent else False
    eff_filters = list(intent.filters) if intent else []

    for cid in candidate_cite_ids:
        if len(object_records) >= 6:
            break
        if cid not in object_records and not cid.startswith(("CON_", "GUIDE_")):
            enriched = get_object(
                object_id=cid,
                focal_length_mm=eff_focal,
                sensor_format=eff_sensor,
                aperture_mm=eff_aperture,
                mount_type=eff_mount,
                has_autoguiding=eff_autoguide,
                filters=eff_filters,
            )
            if enriched.get("found"):
                object_records[cid] = enriched

    has_rig = bool(
        prof is not None
        or eff_focal is not None
        or eff_aperture is not None
        or eff_sensor != "APS-C"
        or (intent and intent.mode == "astrophotography")
    )

    catalog_summary: list[dict[str, Any]] = []
    shared_cam_defaults: dict[str, Any] = {}
    for cid in candidate_cite_ids:
        if len(catalog_summary) >= 5 or no_darkness_impossible and cid not in priority_dsos:
            continue
        if cid in object_records:
            rec = object_records[cid]
            const_code = str(rec.get("constellation") or "").upper()
            sg = rec.get("shooting_guide") or {}
            sg_ori = sg.get("orientation") or {}
            sg_app = sg.get("appearance") or {}
            sg_cam = sg.get("camera_settings") or {}
            if has_rig and not shared_cam_defaults and sg_cam:
                shared_cam_defaults = {
                    "fov_arcmin": sg_cam.get("fov_arcmin"),
                    "recommended_iso_or_gain": sg_cam.get("recommended_iso_or_gain"),
                }
            raw_look = str(sg_app.get("visual_and_photo_look") or "")
            short_look = raw_look.split(".")[0].strip()
            if "HDR" in raw_look:
                short_look += "; HDR bright core (blend short+long subs)"
            raw_sub = str(sg_cam.get("recommended_sub_exposure") or "")
            short_sub = raw_sub.split("(")[0].strip() if "500-rule" not in raw_sub else raw_sub
            nb_ok = bool(rec.get("narrowband_friendly"))
            compact_sg = {
                k: v
                for k, v in {
                    "coords": f"{sg_ori.get('ra_hms') or ''} {sg_ori.get('dec_dms') or ''}".strip()
                    or None,
                    "size_vs_moon": sg_ori.get("size_vs_full_moon"),
                    "how_to_find": str(sg_ori.get("how_to_find") or "").split(".")[0].strip()
                    or None,
                    "visual_and_photo_look": short_look or None,
                    "target_fill_pct": sg_cam.get("target_fill_pct") if has_rig else None,
                    "recommended_sub_exposure": short_sub or None,
                    "filter": "Dual-band Hα/OIII or RGB"
                    if nb_ok
                    else "Broadband UV/IR-cut (no narrowband)",
                }.items()
                if v is not None
            }
            obj_entry: dict[str, Any] = {
                "doc_id": cid,
                "title": rec.get("title"),
                "type": rec.get("object_type"),
                "constellation": f"{CONSTELLATION_NAMES.get(const_code, const_code)} ({const_code})",
                "magnitude": rec.get("magnitude"),
                "size_arcmin": rec.get("size_major_arcmin"),
                "narrowband": nb_ok,
                "shooting_guide": compact_sg,
            }
            if f"CON_{const_code}" in retrieved_doc_ids:
                obj_entry["constellation_doc_id"] = f"CON_{const_code}"
            if has_rig and rec.get("framing_advice"):
                obj_entry["framing_advice"] = rec["framing_advice"]
            catalog_summary.append(obj_entry)

    optics_payload: dict[str, Any] = {
        k: v
        for k, v in {
            "mode": intent.mode if intent else "hybrid",
            "aperture_mm": eff_aperture,
            "focal_length_mm": eff_focal,
            "sensor_format": eff_sensor if has_rig else None,
            "mount_type": eff_mount if has_rig else None,
            "has_autoguiding": eff_autoguide if eff_autoguide else None,
            "filters": eff_filters if eff_filters else None,
            **shared_cam_defaults,
        }.items()
        if v is not None
    }
    if prof is not None:
        optics_payload["user_equipment_profile"] = {
            "rig_summary": _format_rig_summary(prof),
            "telescope_or_lens": prof.telescope_or_lens,
            "focal_length_mm": prof.focal_length_mm,
            "aperture_mm": prof.aperture_mm,
            "f_ratio": f"f/{round(prof.focal_length_mm / prof.aperture_mm, 1)}"
            if prof.aperture_mm
            else None,
            "camera_model": prof.camera_model,
            "sensor_format": prof.sensor_format,
            "mount_type": prof.mount_type,
            "has_autoguiding": prof.has_autoguiding,
            "filters": prof.filters,
            "bortle_class": prof.bortle_class,
        }

    deterministic_json: dict[str, Any] = {
        "reference_date": settings.get_reference_date().isoformat(),
        "max_weather_forecast_horizon_days": settings.max_forecast_days,
        "location": location_payload,
        "optics": optics_payload,
    }
    if len(found_geocodes) > 1:
        compared_sites: list[dict[str, Any]] = []
        for g_entry in found_geocodes:
            g_lat = float(g_entry.get("lat") or 0.0)
            g_lon = float(g_entry.get("lon") or 0.0)
            matched_n = next(
                (
                    n
                    for n in all_nights
                    if abs(float(n.get("lat") or 0.0) - g_lat) < 0.05
                    and abs(float(n.get("lon") or 0.0) - g_lon) < 0.05
                ),
                None,
            )
            matched_w = next(
                (
                    w
                    for w in all_weathers
                    if abs(float(w.get("lat") or 0.0) - g_lat) < 0.05
                    and abs(float(w.get("lon") or 0.0) - g_lon) < 0.05
                ),
                None,
            )
            site_entry: dict[str, Any] = {
                "place_name": g_entry.get("place_name"),
                "short_name": g_entry.get("short_name"),
                "lat": g_lat,
                "lon": g_lon,
            }
            if matched_n:
                site_entry["night_ephemeris"] = {
                    k: matched_n.get(k)
                    for k in (
                        "date",
                        "tz_label",
                        "has_astronomical_darkness",
                        "astronomical_dark_start",
                        "astronomical_dark_end",
                        "moonless_dark_hours",
                        "moon_illumination_pct",
                        "moon_phase_name",
                    )
                    if matched_n.get(k) is not None
                }
            if matched_w:
                site_entry["night_weather"] = {
                    k: matched_w.get(k)
                    for k in (
                        "available",
                        "date",
                        "cloud_cover_pct",
                        "clear_hours",
                        "best_clear_window",
                        "humidity_pct",
                        "wind_kmh",
                        "transparency",
                    )
                    if matched_w.get(k) is not None
                }
            compared_sites.append(site_entry)
        deterministic_json["compared_sites"] = compared_sites
    unindexed = _find_unindexed_designations(q)
    if unindexed:
        deterministic_json["unindexed_designations_note"] = (
            f"Designation(s) {unindexed} are not standalone corpus records; "
            "answer astronomically and cite only retrieved IDs below."
        )
    if white_night_warning:
        deterministic_json["white_night_warning"] = white_night_warning

    if is_multi_night and ranked_nights and not no_darkness_impossible:
        deterministic_json["multi_night_ranking"] = ranked_nights
    elif active_night or active_weather:
        if active_night:
            has_true_dark = active_night.get("has_astronomical_darkness", True) is not False
            ephem_payload: dict[str, Any] = {
                "date": active_night.get("date"),
                "weekday": active_night.get("weekday"),
                "tz_label": active_night.get("tz_label"),
                "has_astronomical_darkness": has_true_dark,
                "min_sun_altitude_deg": active_night.get("min_sun_altitude_deg"),
                "astronomical_dark_start": active_night.get("astronomical_dark_start")
                if has_true_dark
                else None,
                "astronomical_dark_end": active_night.get("astronomical_dark_end")
                if has_true_dark
                else None,
                "nautical_twilight_window": (
                    f"{effective_night_window[0]}–{effective_night_window[1]}"
                    if (not has_true_dark and effective_night_window)
                    else None
                ),
                "moonless_dark_hours": active_night.get("moonless_dark_hours")
                if has_true_dark
                else 0.0,
                "moon_illumination_pct": active_night.get("moon_illumination_pct"),
                "moon_phase_name": active_night.get("moon_phase_name"),
                "moon_set_local": active_night.get("moon_set_local"),
                "moon_rise_local": active_night.get("moon_rise_local"),
                "summary": active_night.get("summary"),
            }
            deterministic_json["night_ephemeris"] = {
                k: v for k, v in ephem_payload.items() if v is not None
            }
        if active_weather:
            deterministic_json["night_weather"] = {
                k: active_weather[k]
                for k in (
                    "available",
                    "date",
                    "cloud_cover_pct",
                    "evening_cloud_cover_pct",
                    "late_night_cloud_cover_pct",
                    "clear_hours",
                    "best_clear_window",
                    "max_precip_prob_pct",
                    "wind_kmh",
                    "humidity_pct",
                    "transparency",
                )
                if k in active_weather and active_weather[k] is not None
            }

    if plan_items:
        deterministic_json["verified_observation_schedule"] = [
            {
                "object_id": pi.object_id,
                "name": pi.name,
                "slot": f"{pi.start.strftime('%H:%M')}–{pi.end.strftime('%H:%M')}",
                "altitude_deg": pi.altitude_deg,
                "direction": pi.direction,
                "difficulty": pi.difficulty,
            }
            for pi in plan_items
        ]
    if unobservable_targets:
        deterministic_json["unobservable_targets"] = unobservable_targets
    if catalog_summary:
        deterministic_json["catalog_objects"] = catalog_summary

    catalog_doc_ids = {item["doc_id"] for item in catalog_summary}
    snippet_chunks = [
        c
        for c in retrieved
        if c.doc_id in candidate_cite_ids
        and not (c.chunk_id.endswith("#catalog-0") and c.doc_id in catalog_doc_ids)
    ]
    if not snippet_chunks:
        snippet_chunks = [c for c in retrieved if c.doc_id in candidate_cite_ids]

    # Full cleaned Wikipedia/Guide passages up to ~1,400 chars across up to 6 chunks (no [:220] truncation!)
    context_snippets = format_retrieved_passages(snippet_chunks, max_chunks=6, max_chars=1400)

    prev_ans = state.get("answer")
    retry_errors = (
        prev_ans.verification_errors
        if (state.get("retries", 0) > 0 and prev_ans and prev_ans.verification_errors)
        else None
    )
    history = _normalize_conversation_history(state.get("conversation_history"))

    sys_prompt, user_prompt = build_compose_prompts(
        question=q,
        lang=lang,
        deterministic_json=deterministic_json,
        context_snippets=context_snippets,
        candidate_cite_ids=candidate_cite_ids,
        plan_items=plan_items,
        unindexed=unindexed,
        priority_dsos=priority_dsos,
        catalog_summary=catalog_summary,
        retry_errors=retry_errors,
        conversation_history=history,
        no_darkness_impossible=no_darkness_impossible,
    )

    return _ComposePrepared(
        sys_prompt=sys_prompt,
        user_prompt=user_prompt,
        plan_items=plan_items,
        conditions_str=conditions_str,
        retrieved_doc_ids=retrieved_doc_ids,
        effective_night_window=effective_night_window,
    )


def _finalize_compose_output(
    state: AstroState,
    prepared: _ComposePrepared,
    raw_answer_text: str,
    extra_in: int,
    extra_out: int,
    backend_model: str,
) -> dict[str, Any]:
    """Post-process synthesized text, extract retrieved inline citations (without fake injection), and track cost."""
    answer_text = _JSON_SECTION_BRACKET_RE.sub("", raw_answer_text).strip()

    inline_cited: list[str] = []
    for raw_cid in extract_inline_citations(answer_text):
        if raw_cid in prepared.retrieved_doc_ids:
            inline_cited.append(raw_cid)
        else:
            norm_cid = normalize_object_id(raw_cid)
            if norm_cid in prepared.retrieved_doc_ids:
                inline_cited.append(norm_cid)
            else:
                rec = resolve_object_record(raw_cid)
                if rec and rec.get("doc_id") in prepared.retrieved_doc_ids:
                    inline_cited.append(str(rec["doc_id"]))

    plan_ids = [
        pi.object_id for pi in prepared.plan_items if pi.object_id in prepared.retrieved_doc_ids
    ]
    # Never inject `candidate_cite_ids[:3]` when the LLM failed to cite sources — let `validator.py` trigger a retry!
    merged_sources = list(dict.fromkeys(inline_cited + plan_ids))

    ans = AstroAnswer(
        answer=answer_text,
        plan=prepared.plan_items,
        conditions=prepared.conditions_str,
        sources=merged_sources,
        abstained=False,
        verified=True,
    )

    compose_cost = (
        0.0 if settings.eval_offline_mode else compute_cost_usd(backend_model, extra_in, extra_out)
    )
    return {
        "answer": ans,
        "night_window": prepared.effective_night_window,
        "tokens_in": state.get("tokens_in", 0) + extra_in,
        "tokens_out": state.get("tokens_out", 0) + extra_out,
        "cost_usd": round(float(state.get("cost_usd", 0.0)) + compose_cost, 6),
    }


def compose_node(state: AstroState) -> dict[str, Any]:
    """Node 4 (`compose` / Rédiger): Pass clean structured tool JSON + full RAG passages to the LLM."""
    prepared_or_early = _prepare_compose_context(state)
    if isinstance(prepared_or_early, AstroAnswer):
        return {"answer": prepared_or_early}

    answer_text, extra_in, extra_out, backend_model = generate_grounded_answer(
        prepared_or_early.sys_prompt,
        prepared_or_early.user_prompt,
        mode_override=state.get("llm_mode_override"),
    )
    return _finalize_compose_output(
        state=state,
        prepared=prepared_or_early,
        raw_answer_text=answer_text,
        extra_in=extra_in,
        extra_out=extra_out,
        backend_model=backend_model,
    )


def stream_compose_node(
    state: AstroState,
) -> Iterator[tuple[str | None, dict[str, Any] | None]]:
    """Streaming variant of `compose_node` yielding `(delta_text, None)` chunks then `(None, state_update)`."""
    prepared_or_early = _prepare_compose_context(state)
    if isinstance(prepared_or_early, AstroAnswer):
        if prepared_or_early.answer:
            yield prepared_or_early.answer, None
        yield None, {"answer": prepared_or_early}
        return

    # Support tests that monkeypatch `nodes.generate_grounded_answer` without patching `stream_grounded_answer`
    if (
        generate_grounded_answer is not _DEFAULT_GENERATE_GROUNDED_ANSWER
        and stream_grounded_answer is _DEFAULT_STREAM_GROUNDED_ANSWER
    ):
        full_text, extra_in, extra_out, backend_model = generate_grounded_answer(
            prepared_or_early.sys_prompt,
            prepared_or_early.user_prompt,
            mode_override=state.get("llm_mode_override"),
        )
        if full_text:
            yield full_text, None
        yield (
            None,
            _finalize_compose_output(
                state=state,
                prepared=prepared_or_early,
                raw_answer_text=full_text,
                extra_in=extra_in,
                extra_out=extra_out,
                backend_model=backend_model,
            ),
        )
        return

    chunks: list[str] = []
    extra_in = 0
    extra_out = 0
    backend_model = ""
    for delta, in_tok, out_tok, backend in stream_grounded_answer(
        prepared_or_early.sys_prompt,
        prepared_or_early.user_prompt,
        mode_override=state.get("llm_mode_override"),
    ):
        if delta:
            chunks.append(delta)
            yield delta, None
        if backend or in_tok or out_tok:
            extra_in = in_tok
            extra_out = out_tok
            backend_model = backend

    yield (
        None,
        _finalize_compose_output(
            state=state,
            prepared=prepared_or_early,
            raw_answer_text="".join(chunks),
            extra_in=extra_in,
            extra_out=extra_out,
            backend_model=backend_model,
        ),
    )


def validate_node(state: AstroState) -> dict[str, Any]:
    """Node 5 (`validate` / Vérifier): Run pure-Python verification on `state['answer']`."""
    ans = state.get("answer")
    if ans is None:
        return {}
    checked = verify_astro_answer(ans, state)
    retries = state.get("retries", 0)
    if not checked.verified:
        retries += 1
    return {"answer": checked, "retries": retries}


__all__ = [
    "_angular_separation_deg",
    "_build_plan_items",
    "_check_out_of_corpus",
    "_find_astrometric_companions",
    "_find_unindexed_designations",
    "_resolve_corpus_ids",
    "_select_dynamic_sky_targets",
    "_select_seasonal_catalog_targets",
    "agent_node",
    "compose_node",
    "generate_grounded_answer",
    "guard_node",
    "plan_react_step_with_llm",
    "refuse_node",
    "route_query_with_llm",
    "stream_compose_node",
    "stream_grounded_answer",
    "tools_node",
    "validate_node",
]
