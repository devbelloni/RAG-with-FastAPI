"""Prompt and context formatting. Pure functions, so they are unit-tested."""

from __future__ import annotations

from typing import Any

PROMPT_TEMPLATE = """Você é um assistente especializado em analisar documentos técnicos e acadêmicos.
Responda a pergunta baseado APENAS no contexto abaixo.
Se a informação estiver no contexto, mesmo que parcialmente, use-a para responder.
Se realmente não houver nenhuma informação relevante, diga que não encontrou.

Contexto:
{context}

Pergunta: {question}

Resposta detalhada:
"""


def format_context(chunks: list[Any]) -> str:
    """Join retrieved chunks (objects with ``page_content``) into the prompt context."""
    return "\n\n---\n\n".join(chunk.page_content.strip() for chunk in chunks)


def extract_sources(chunks: list[Any]) -> list[str]:
    """Unique ``file (page N)`` references, in retrieval order."""
    seen: dict[str, None] = {}
    for chunk in chunks:
        meta = getattr(chunk, "metadata", {}) or {}
        name = str(meta.get("source", "unknown")).replace("\\", "/").rsplit("/", 1)[-1]
        page = meta.get("page")
        seen[f"{name} (page {int(page) + 1})" if page is not None else name] = None
    return list(seen)


def build_prompt(question: str, chunks: list[Any]) -> str:
    return PROMPT_TEMPLATE.format(context=format_context(chunks), question=question)
