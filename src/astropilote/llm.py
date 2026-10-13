"""Unified Google ADK model (`LiteLlm` / native Gemini) and hardware device factory for AstroPilote."""

from __future__ import annotations

import os

import httpx
import torch
from google.adk.models.lite_llm import LiteLlm

from astropilote.config import settings


def get_torch_device() -> str:
    """Detect the fastest available compute device ('mps' on Apple Silicon, 'cuda', or 'cpu')."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def check_ollama_status() -> dict[str, object]:
    """Check whether the local Ollama daemon is running and list installed models."""
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            payload = resp.json()
            models = [m.get("name", "") for m in payload.get("models", [])]
            return {
                "running": True,
                "base_url": settings.ollama_base_url,
                "installed_models": models,
                "target_model_ready": any(
                    m == settings.llm_model or m.startswith(settings.llm_model.split(":")[0])
                    for m in models
                ),
            }
    except Exception as exc:
        return {
            "running": False,
            "base_url": settings.ollama_base_url,
            "installed_models": [],
            "target_model_ready": False,
            "error": str(exc),
        }


def get_adk_model(
    provider: str | None = None,
    model_name: str | None = None,
) -> LiteLlm | str:
    """Return a Google ADK model instance (`LiteLlm` for Ollama/OpenAI/Anthropic, or native Gemini ID).

    In Google ADK (`LlmAgent`), `model` accepts either:
    - A `LiteLlm(model="ollama_chat/qwen3:8b")` wrapper for local Ollama or third-party APIs
    - A native Gemini model string (e.g. `"gemini-2.5-flash"`) when `GEMINI_API_KEY` is configured.
    """
    prov = (provider or settings.llm_provider).lower()
    mod = model_name or settings.llm_model

    if prov == "ollama":
        os.environ["OLLAMA_API_BASE"] = settings.ollama_base_url
        litellm_id = mod if mod.startswith("ollama") else f"ollama_chat/{mod}"
        return LiteLlm(model=litellm_id, api_base=settings.ollama_base_url)

    if prov == "openai_compatible":
        litellm_id = mod if "/" in mod else f"openai/{mod}"
        return LiteLlm(
            model=litellm_id,
            api_base=settings.ollama_base_url,
            api_key=settings.openai_api_key or "local-gemma4",
        )

    if prov == "gemini":
        if settings.gemini_api_key:
            os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key
            os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
        return mod if mod.startswith("gemini") else "gemini-2.5-flash"

    if prov == "openai":
        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        litellm_id = mod if "/" in mod else f"openai/{mod}"
        return LiteLlm(model=litellm_id)

    if prov == "anthropic":
        if settings.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
        litellm_id = mod if "/" in mod else f"anthropic/{mod}"
        return LiteLlm(model=litellm_id)

    raise ValueError(f"Unsupported ADK LLM provider: {prov}")


def get_litellm_model_string(
    provider: str | None = None,
    model_name: str | None = None,
) -> str:
    """Return the normalized LiteLLM model identifier string (e.g. 'ollama_chat/gemma4')."""
    prov = (provider or settings.llm_provider).lower()
    mod = model_name or settings.llm_model
    if prov == "ollama":
        return mod if mod.startswith("ollama") else f"ollama_chat/{mod}"
    if prov in ("openai", "openai_compatible"):
        return mod if "/" in mod else f"openai/{mod}"
    if prov == "gemini":
        return mod if mod.startswith("gemini/") else f"gemini/{mod}"
    if prov == "anthropic":
        return mod if "/" in mod else f"anthropic/{mod}"
    return mod
