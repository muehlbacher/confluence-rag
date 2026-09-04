"""Central configuration, loaded from environment / .env via pydantic-settings.

Secrets never live in source. Everything here comes from the environment; see
`.env.example` for the full list. Fields the later milestones need carry
plan-default values so that importing this module during M1 does not require the
whole stack to be configured, but the Confluence credentials (the only true
secrets M1 uses) have no defaults and must be supplied.
"""

from functools import lru_cache
from typing import List

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from typing_extensions import Annotated


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Confluence Data Center ---
    confluence_base_url: str = Field(..., validation_alias="CONFLUENCE_BASE_URL")
    # Accept the plan's CONFLUENCE_PAT, but also honour an existing CONFLUENCE_TOKEN.
    confluence_pat: str = Field(
        ...,
        validation_alias=AliasChoices("CONFLUENCE_PAT", "CONFLUENCE_TOKEN"),
    )
    # NoDecode: disable pydantic-settings' default JSON decoding so the
    # comma-splitting validator below handles the raw "ENG,OPS,DOCS" string.
    confluence_spaces: Annotated[List[str], NoDecode] = Field(
        ..., validation_alias="CONFLUENCE_SPACES"
    )

    # --- LLM (OpenAI-compatible) ---
    llm_base_url: str = "http://llm:8000/v1"
    llm_api_key: str = "not-needed"
    llm_model: str = "qwen2.5-32b-instruct"

    # --- Embeddings (OpenAI-compatible, BGE-M3) ---
    embed_base_url: str = "http://embed:8080/v1"
    embed_api_key: str = "not-needed"
    embed_model: str = "bge-m3"
    embed_dim: int = 1024

    # --- Reranker ---
    rerank_url: str = "http://rerank:8080/rerank"
    rerank_model: str = "bge-reranker-v2-m3"
    rerank_api_key: str = ""  # Bearer token if the reranker is behind an API gateway

    # --- Qdrant ---
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "confluence"
    # If set, use qdrant-client local (embedded) mode at this path instead of a
    # server URL — lets M2 run without a Qdrant daemon / docker.
    qdrant_path: str = ""

    # --- Chunking ---
    chunk_max_tokens: int = 512
    # Hard ceiling for whole tables/code blocks (embedder context limit).
    chunk_hard_max_tokens: int = 5000

    # --- Retrieval tuning ---
    retrieve_top_k: int = 20
    rerank_top_n: int = 5
    rerank_score_min: float = 0.3

    # --- Webhook ---
    webhook_secret: str = "changeme"

    # --- Local page-state store (M1) ---
    page_state_db: str = "page_state.db"

    @field_validator("confluence_spaces", mode="before")
    @classmethod
    def _split_spaces(cls, v):
        """Accept a comma-separated string or a list; trim blanks."""
        if isinstance(v, str):
            v = v.split(",")
        return [s.strip() for s in v if s and s.strip()]

    @field_validator("confluence_base_url", mode="before")
    @classmethod
    def _strip_trailing_slash(cls, v):
        if isinstance(v, str):
            return v.rstrip("/")
        return v


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton. Call this instead of instantiating Settings."""
    return Settings()
