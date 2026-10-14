"""Centralized configuration for AstroPilote (Anthropic API mode + Local Ollama mode)."""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from `.env` or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Dual LLM Mode: "api" (Anthropic) or "local" (Ollama) ---
    llm_mode: Literal["api", "local"] = Field(default="api", alias="LLM_MODE")

    # Anthropic API settings
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-sonnet-4-20250514", alias="ANTHROPIC_MODEL")

    # Local Ollama settings
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1:8b", alias="OLLAMA_MODEL")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")

    # --- Embeddings & Reranker ---
    embedding_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL")
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )
    use_reranker: bool = Field(default=True, alias="USE_RERANKER")

    # --- Retrieval & Agent Hyperparameters ---
    max_chunk_tokens: int = Field(default=320, alias="MAX_CHUNK_TOKENS")
    bm25_top_k: int = Field(default=20, alias="BM25_TOP_K")
    dense_top_k: int = Field(default=20, alias="DENSE_TOP_K")
    rrf_k: int = Field(default=60, alias="RRF_K")
    final_top_k: int = Field(default=5, alias="FINAL_TOP_K")
    max_agent_steps: int = Field(default=6, alias="MAX_AGENT_STEPS")
    max_compose_retries: int = Field(default=1, alias="MAX_COMPOSE_RETRIES")

    # --- Storage Paths ---
    data_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data",
        alias="DATA_DIR",
    )
    qdrant_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "index" / "qdrant",
        alias="QDRANT_PATH",
    )
    bm25_path: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "index" / "bm25",
        alias="BM25_PATH",
    )
    eval_offline_mode: bool = Field(default=False, alias="EVAL_OFFLINE_MODE")

    @property
    def active_model_name(self) -> str:
        """Return the model identifier for the currently selected mode."""
        if self.llm_mode == "api" and self.anthropic_api_key:
            return self.anthropic_model
        return self.ollama_model


settings = Settings()
