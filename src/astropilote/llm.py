"""Open-source LLM client supporting API mode (Anthropic / OpenRouter) and Local mode (Ollama / MPS Transformers)."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import httpx
from dotenv import dotenv_values

from astropilote.config import settings
from astropilote.tracking.tracker import estimate_text_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
LOCAL_HF_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def _refresh_env_settings() -> None:
    """Reload `.env` dynamically so changes to `ANTHROPIC_API_KEY` or `LLM_MODE` apply immediately."""
    if ENV_PATH.exists():
        vals = dotenv_values(ENV_PATH)
        if vals.get("ANTHROPIC_API_KEY"):
            settings.anthropic_api_key = str(vals["ANTHROPIC_API_KEY"]).strip()
        elif os.environ.get("ANTHROPIC_API_KEY"):
            settings.anthropic_api_key = os.environ["ANTHROPIC_API_KEY"].strip()
        if vals.get("ANTHROPIC_MODEL"):
            settings.anthropic_model = str(vals["ANTHROPIC_MODEL"]).strip()
        if vals.get("OLLAMA_MODEL"):
            settings.ollama_model = str(vals["OLLAMA_MODEL"]).strip()
        if vals.get("LLM_MODE") in ("api", "local"):
            settings.llm_mode = vals["LLM_MODE"]  # type: ignore[assignment]


def save_anthropic_key(api_key: str) -> dict[str, object]:
    """Save `ANTHROPIC_API_KEY` into `.env` and update runtime settings immediately."""
    clean_key = api_key.strip()
    settings.anthropic_api_key = clean_key
    settings.llm_mode = "api"

    lines: list[str] = []
    found_key = False
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY="):
                lines.append(f"ANTHROPIC_API_KEY={clean_key}")
                found_key = True
            else:
                lines.append(line)
    if not found_key:
        lines.append(f"ANTHROPIC_API_KEY={clean_key}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return check_llm_status()


def get_torch_device() -> str:
    """Detect the fastest available compute device ('mps' on Apple Silicon, 'cuda', or 'cpu')."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _resolve_openrouter_model(raw_model: str) -> str:
    """Map Anthropic model names to valid OpenRouter Anthropic model IDs."""
    m = raw_model.strip()
    if m.startswith("anthropic/"):
        return m
    if "sonnet" in m.lower():
        return "anthropic/claude-sonnet-4"
    if "haiku" in m.lower():
        return "anthropic/claude-haiku-4.5"
    return "anthropic/claude-haiku-4.5"


@lru_cache(maxsize=1)
def _load_local_hf_pipeline():
    """Lazy-load local open-weight instruct model (`SmolLM2-360M-Instruct`) on Apple Silicon MPS / CPU."""
    from astropilote.retrieval.hybrid import _configure_quiet_hf_hub

    _configure_quiet_hf_hub()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = get_torch_device()
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_HF_MODEL, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_HF_MODEL, dtype=dtype, local_files_only=True
        ).to(device)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_HF_MODEL)
        model = AutoModelForCausalLM.from_pretrained(LOCAL_HF_MODEL, dtype=dtype).to(device)
    model.eval()
    return tokenizer, model, device


def check_llm_status() -> dict[str, object]:
    """Return availability of both Anthropic/OpenRouter API key and local Ollama / MPS Transformers."""
    _refresh_env_settings()
    key = (settings.anthropic_api_key or "").strip()
    has_anthropic_key = bool(key)
    api_model_display = (
        _resolve_openrouter_model(settings.anthropic_model)
        if key.startswith("sk-or-")
        else settings.anthropic_model
    )

    ollama_running = False
    ollama_models: list[str] = []
    try:
        with httpx.Client(timeout=0.8) as client:
            resp = client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            if resp.status_code == 200:
                ollama_running = True
                ollama_models = [m.get("name", "") for m in resp.json().get("models", [])]
    except Exception:
        ollama_running = False

    local_model_label = (
        settings.ollama_model
        if ollama_running
        else f"SmolLM2-360M ({get_torch_device().upper()})"
    )

    return {
        "configured_mode": settings.llm_mode,
        "anthropic_ready": has_anthropic_key,
        "anthropic_model": api_model_display,
        "ollama_running": True,
        "ollama_daemon": ollama_running,
        "ollama_model": local_model_label,
        "ollama_installed_models": ollama_models,
    }


def _call_api_llm(system_prompt: str, user_prompt: str) -> tuple[str, int, int, str]:
    """Call Anthropic via OpenRouter (`sk-or-...`) or direct Anthropic SDK (`sk-ant-...`)."""
    key = settings.anthropic_api_key.strip()
    if not key:
        return "", 0, 0, ""

    # A. OpenRouter key (`sk-or-v1-...`) routing to Anthropic Claude
    if key.startswith("sk-or-"):
        or_model = _resolve_openrouter_model(settings.anthropic_model)
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": or_model,
                    "max_tokens": 1024,
                    "temperature": settings.llm_temperature,
                    "reasoning": {"enabled": False},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
            )
            if resp.status_code == 200:
                payload = resp.json()
                text = (
                    payload.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                usage = payload.get("usage", {})
                in_tok = int(usage.get("prompt_tokens") or estimate_text_tokens(user_prompt))
                out_tok = int(usage.get("completion_tokens") or estimate_text_tokens(text))
                if text:
                    return text, in_tok, out_tok, or_model
        return "", 0, 0, ""

    # B. Direct Anthropic key (`sk-ant-...`)
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=key, timeout=30.0)
        msg = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1024,
            temperature=settings.llm_temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        blocks = [getattr(b, "text", "") for b in msg.content if hasattr(b, "text")]
        text = "\n".join(blocks).strip()
        in_tok = int(getattr(msg.usage, "input_tokens", 0) or estimate_text_tokens(user_prompt))
        out_tok = int(getattr(msg.usage, "output_tokens", 0) or estimate_text_tokens(text))
        return text, in_tok, out_tok, f"anthropic/{settings.anthropic_model}"
    except ImportError:
        with httpx.Client(timeout=30.0) as http_client:
            resp = http_client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.anthropic_model,
                    "max_tokens": 1024,
                    "temperature": settings.llm_temperature,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )
            if resp.status_code == 200:
                payload = resp.json()
                blocks = [
                    b.get("text", "")
                    for b in payload.get("content", [])
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                text = "\n".join(blocks).strip()
                usage = payload.get("usage", {})
                in_tok = int(usage.get("input_tokens") or estimate_text_tokens(user_prompt))
                out_tok = int(usage.get("output_tokens") or estimate_text_tokens(text))
                if text:
                    return text, in_tok, out_tok, f"anthropic/{settings.anthropic_model}"
        return "", 0, 0, ""


def _compact_prompt_for_small_local_model(user_prompt: str) -> str:
    """Keep the prompt within ~600 tokens for 360M local models so chat template tags are never truncated."""
    if "\n\nRetrieved Corpus Passages:\n" in user_prompt and "\n\nRequired inline citations:" in user_prompt:
        before_passages, rest = user_prompt.split("\n\nRetrieved Corpus Passages:\n", 1)
        passages, tail = rest.split("\n\nRequired inline citations:", 1)
        short_before = (
            before_passages[:1400].rsplit("\n", 1)[0]
            if len(before_passages) > 1400
            else before_passages
        )
        short_passages = passages[:450].rsplit("\n", 1)[0] if len(passages) > 450 else passages
        return (
            f"{short_before}\n\nRetrieved Corpus Passages:\n{short_passages}\n\n"
            f"Required inline citations:{tail}"
        )
    return user_prompt[:1800]


def _call_local_llm(system_prompt: str, user_prompt: str) -> tuple[str, int, int, str]:
    """Call local Ollama if running, or local open-weight Transformers model on Apple Silicon MPS."""
    # 1. Try Ollama daemon first if running
    try:
        with httpx.Client(timeout=1.5) as client:
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
                    in_tok = int(
                        data.get("prompt_eval_count") or estimate_text_tokens(user_prompt)
                    )
                    out_tok = int(data.get("eval_count") or estimate_text_tokens(content))
                    return content, in_tok, out_tok, f"ollama/{settings.ollama_model}"
    except Exception:
        pass

    # 2. Run local open-weight model via HuggingFace Transformers on MPS / CPU
    import torch

    tokenizer, model, device = _load_local_hf_pipeline()
    compact_user = _compact_prompt_for_small_local_model(user_prompt)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": compact_user},
    ]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=2048).to(
        device
    )
    in_tok = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=260,
            do_sample=False,
            repetition_penalty=1.18,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = outputs[0][in_tok:]
    out_tok = int(gen_ids.shape[0])
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text, in_tok, out_tok, LOCAL_HF_MODEL


def generate_grounded_answer(
    system_prompt: str,
    user_prompt: str,
    mode_override: Literal["api", "local"] | None = None,
) -> tuple[str, int, int, str]:
    """Generate a grounded answer using API (`api`) or Local (`local`) LLM.

    Returns `(content, tokens_in, tokens_out, backend_used)`.
    Falls back gracefully if offline so tests and evaluations never crash.
    """
    if settings.eval_offline_mode:
        return "", 0, 0, "offline"

    _refresh_env_settings()
    mode = mode_override or settings.llm_mode

    if mode == "api":
        try:
            text, in_tok, out_tok, backend = _call_api_llm(system_prompt, user_prompt)
            if text:
                return text, in_tok, out_tok, backend
        except Exception:
            pass

    if mode == "local":
        try:
            text, in_tok, out_tok, backend = _call_local_llm(system_prompt, user_prompt)
            if text:
                return text, in_tok, out_tok, backend
        except Exception:
            pass

    return "", 0, 0, "deterministic_fallback"
