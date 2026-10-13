"""Centralized configuration loaded from environment variables and `.env`."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for AstroPilote across ingestion, retrieval, agent, and evaluation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- 1. Local LLM Configuration (Gemma 4 via Ollama or OpenAI-compatible local server) ---
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    llm_provider: Literal["ollama", "openai_compatible", "gemini", "openai", "anthropic"] = Field(
        default="ollama", alias="ASTROPILOTE_LLM_PROVIDER"
    )
    llm_model: str = Field(default="gemma4", alias="ASTROPILOTE_LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="ASTROPILOTE_LLM_TEMPERATURE")

    # --- 2. Optional Cloud API Keys ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    judge_model: str = Field(default="gemma4", alias="ASTROPILOTE_JUDGE_MODEL")

    # --- 3. Retrieval & Indexing Models ---
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="ASTROPILOTE_EMBEDDING_MODEL"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="ASTROPILOTE_RERANKER_MODEL"
    )
    use_reranker: bool = Field(default=True, alias="ASTROPILOTE_USE_RERANKER")

    # --- 4. Retrieval Hyperparameters ---
    bm25_top_k: int = Field(default=20, alias="ASTROPILOTE_BM25_TOP_K")
    dense_top_k: int = Field(default=20, alias="ASTROPILOTE_DENSE_TOP_K")
    rrf_k: int = Field(default=60, alias="ASTROPILOTE_RRF_K")
    final_top_k: int = Field(default=5, alias="ASTROPILOTE_FINAL_TOP_K")
    max_chunk_tokens: int = Field(default=400, alias="ASTROPILOTE_MAX_CHUNK_TOKENS")

    # --- 5. Agent & Tool Execution ---
    max_agent_steps: int = Field(default=6, alias="ASTROPILOTE_MAX_AGENT_STEPS")
    max_compose_retries: int = Field(default=1, alias="ASTROPILOTE_MAX_COMPOSE_RETRIES")
    eval_offline_mode: bool = Field(default=False, alias="ASTROPILOTE_EVAL_OFFLINE_MODE")

    # --- 6. Storage Paths ---
    data_dir: Path = Field(default=Path("data"), alias="ASTROPILOTE_DATA_DIR")
    qdrant_path: Path = Field(default=Path("data/index/qdrant"), alias="ASTROPILOTE_QDRANT_PATH")
    bm25_path: Path = Field(default=Path("data/index/bm25"), alias="ASTROPILOTE_BM25_PATH")


settings = Settings()
