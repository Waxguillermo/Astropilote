"""Google Agent Development Kit (`google-adk`) orchestration for AstroPilote with Gemma 4 (`LiteLlm`)."""

from __future__ import annotations

import re
import time
from typing import Any

import httpx
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from astropilote.agent.schemas import ToolCallRecord
from astropilote.config import settings
from astropilote.llm import get_adk_model, get_litellm_model_string
from astropilote.tools import ALL_TOOLS
from astropilote.tracking.tracker import estimate_text_tokens

ASTRO_ADK_INSTRUCTION = """You are AstroPilote, an expert visual astronomy and astrophotography copilot.
Follow these strict rules on every turn:
1. Ground every factual claim and recommendation in the AstroPilote 506-document corpus using `search_catalog` and `get_object`.
2. For observation or session-planning queries, call `geocode` for the location, `night` for the astronomical darkness window (-18° twilight) and Moon illumination, `visibility` for 30-minute target altitudes and airmass, and `weather` for cloud cover.
3. Never schedule an observation outside astronomical darkness or below 20° altitude.
4. Cite every corpus source inline in brackets, e.g. [M031], [C014], [CON_HER].
5. Abstain politely if the requested target is outside the corpus (e.g. exoplanets, quasars, CMOS sensor datasheets, collimation manuals) or if a weather forecast is requested more than 14 days into the future.
"""


def is_live_llm_available() -> bool:
    """Return True if `EVAL_OFFLINE_MODE` is False and the configured LLM endpoint is reachable."""
    if settings.eval_offline_mode:
        return False

    provider = settings.llm_provider
    if provider == "gemini":
        return bool(settings.gemini_api_key)
    if provider == "openai":
        return bool(settings.openai_api_key)
    if provider == "anthropic":
        return bool(settings.anthropic_api_key)
    if provider == "openai_compatible":
        try:
            with httpx.Client(timeout=0.8) as client:
                resp = client.get(f"{settings.ollama_base_url.rstrip('/')}/models")
                return resp.status_code == 200
        except Exception:
            return False

    # Default: local Ollama server (`gemma4`)
    try:
        with httpx.Client(timeout=0.8) as client:
            resp = client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            return resp.status_code == 200
    except Exception:
        return False


def build_adk_agent(
    tool_trace_sink: list[ToolCallRecord] | None = None,
    step_counter: dict[str, int] | None = None,
) -> LlmAgent:
    """Construct a Google ADK `LlmAgent` wired with `get_adk_model()` (`gemma4` default) and all 6 tools."""
    trace = tool_trace_sink if tool_trace_sink is not None else []
    counters = step_counter if step_counter is not None else {"steps": 0}
    timings: dict[str, float] = {}

    def _before_model_callback(callback_context: Any, llm_request: Any) -> Any:
        counters["steps"] = counters.get("steps", 0) + 1
        if counters["steps"] > settings.max_agent_steps:
            return genai_types.Content(
                role="model",
                parts=[
                    genai_types.Part.from_text(
                        text="Maximum agent steps (6) reached; synthesizing final grounded answer."
                    )
                ],
            )
        return None

    def _before_tool_callback(tool: Any, args: dict[str, Any], tool_context: Any) -> Any:
        t_name = getattr(tool, "name", str(tool))
        timings[t_name] = time.perf_counter()
        return None

    def _after_tool_callback(
        tool: Any, args: dict[str, Any], tool_context: Any, tool_response: Any
    ) -> Any:
        t_name = getattr(tool, "name", str(tool))
        t0 = timings.pop(t_name, time.perf_counter())
        dur = round(time.perf_counter() - t0, 4)
        trace.append(ToolCallRecord(name=t_name, arguments=dict(args), duration_s=dur))
        return None

    return LlmAgent(
        name="astropilote_adk_copilot",
        model=get_adk_model(),
        description=(
            "Hybrid RAG and spherical astronomy copilot for visual astronomy and astrophotography."
        ),
        instruction=ASTRO_ADK_INSTRUCTION,
        tools=ALL_TOOLS,
        before_model_callback=_before_model_callback,
        before_tool_callback=_before_tool_callback,
        after_tool_callback=_after_tool_callback,
    )


def create_adk_runner() -> InMemoryRunner:
    """Create a Google ADK `InMemoryRunner` wrapping the AstroPilote `LlmAgent`."""
    agent = build_adk_agent()
    return InMemoryRunner(agent=agent, app_name="astropilote")


def synthesize_with_adk_llm(
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, int, int]:
    """Generate a grounded response via LiteLLM / Gemma 4 (`google-adk` backend) when online."""
    if not is_live_llm_available():
        return "", 0, 0

    try:
        import litellm

        model_str = get_litellm_model_string()
        kwargs: dict[str, Any] = {
            "model": model_str,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": settings.llm_temperature,
            "timeout": 25.0,
        }
        if settings.llm_provider == "ollama":
            kwargs["api_base"] = settings.ollama_base_url
        elif settings.llm_provider == "openai_compatible":
            kwargs["api_base"] = settings.llm_base_url
            kwargs["api_key"] = settings.openai_api_key or "local-no-key"

        resp = litellm.completion(**kwargs)
        content = str(resp.choices[0].message.content or "").strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or estimate_text_tokens(user_prompt))
        out_tok = int(getattr(usage, "completion_tokens", 0) or estimate_text_tokens(content))
        return content, in_tok, out_tok
    except Exception:
        return "", 0, 0
