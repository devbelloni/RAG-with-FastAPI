"""Settings, overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = Path("docs")
    persist_dir: Path = Path("chroma_db")
    chunk_size: int = 1500
    chunk_overlap: int = 200
    top_k: int = 6
    llm_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "all-MiniLM-L6-v2"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Settings:
        env = dict(os.environ) if env is None else env
        d = cls()
        return cls(
            docs_dir=Path(env.get("RAG_DOCS_DIR", d.docs_dir)),
            persist_dir=Path(env.get("RAG_PERSIST_DIR", d.persist_dir)),
            chunk_size=int(env.get("RAG_CHUNK_SIZE", d.chunk_size)),
            chunk_overlap=int(env.get("RAG_CHUNK_OVERLAP", d.chunk_overlap)),
            top_k=int(env.get("RAG_TOP_K", d.top_k)),
            llm_model=env.get("RAG_LLM_MODEL", d.llm_model),
            embedding_model=env.get("RAG_EMBEDDING_MODEL", d.embedding_model),
        )
