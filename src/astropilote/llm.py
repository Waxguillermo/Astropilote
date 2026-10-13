"""Open-source LLM client supporting Anthropic API mode (`anthropic`) and Local mode (`Ollama`)."""

from __future__ import annotations

import re
from typing import Literal

import httpx
import torch

from astropilote.config import settings
from astropilote.tracking.tracker import estimate_text_tokens


def get_torch_device() -> str:
    """Detect the fastest available compute device ('mps' on Apple Silicon, 'cuda', or 'cpu')."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def check_llm_status() -> dict[str, object]:
    """Return availability of both Anthropic API key and local Ollama daemon."""
    has_anthropic_key = bool(settings.anthropic_api_key and settings.anthropic_api_key.strip())

    ollama_running = False
    ollama_models: list[str] = []
    try:
        with httpx.Client(timeout=1.2) as client:
            resp = client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            if resp.status_code == 200:
                ollama_running = True
                ollama_models = [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        ollama_running = False

    return {
        "configured_mode": settings.llm_mode,
        "anthropic_ready": has_anthropic_key,
        "anthropic_model": settings.anthropic_model,
        "ollama_running": ollama_running,
        "ollama_model": settings.ollama_model,
        "ollama_installed_models": ollama_models,
    }


def generate_grounded_answer(
    system_prompt: str,
    user_prompt: str,
    mode_override: Literal["api", "local"] | None = None,
) -> tuple[str, int, int, str]:
    """Generate a grounded answer using Anthropic API (`api`) or Local Ollama (`local`).

    Returns `(content, tokens_in, tokens_out, backend_used)`.
    Falls back gracefully if offline so tests and evaluations never crash.
    """
    if settings.eval_offline_mode:
        return "", 0, 0, "offline"

    mode = mode_override or settings.llm_mode

    # 1. Try Anthropic API if mode == "api" (or if selected and key is present)
    if mode == "api" and settings.anthropic_api_key.strip():
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=settings.anthropic_api_key.strip(), timeout=25.0)
            msg = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=900,
                temperature=settings.llm_temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            blocks = [getattr(b, "text", "") for b in msg.content if hasattr(b, "text")]
            text = "\n".join(blocks).strip()
            in_tok = int(getattr(msg.usage, "input_tokens", 0) or estimate_text_tokens(user_prompt))
            out_tok = int(getattr(msg.usage, "output_tokens", 0) or estimate_text_tokens(text))
            if text:
                return text, in_tok, out_tok, f"anthropic/{settings.anthropic_model}"
        except Exception:
            pass

    # 2. Try Local Ollama (`gemma4` or configured local model)
    try:
        with httpx.Client(timeout=25.0) as client:
            resp = client.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/chat",
                json={
                    "model": settings.ollama_model,
                    "stream": False,
                    "options": {"temperature": settings.llm_temperature},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                content = str(data.get("message", {}).get("content", "")).strip()
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                if content:
                    in_tok = int(data.get("prompt_eval_count") or estimate_text_tokens(user_prompt))
                    out_tok = int(data.get("eval_count") or estimate_text_tokens(content))
                    return content, in_tok, out_tok, f"ollama/{settings.ollama_model}"
    except Exception:
        pass

    return "", 0, 0, "deterministic_fallback"
