"""REST API."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .service import Answerer, RagService


class Pergunta(BaseModel):
    pergunta: str = Field(..., max_length=2000, examples=["Qual é a conclusão do estudo?"])


class Resposta(BaseModel):
    pergunta: str
    resposta: str
    fontes: list[str] = Field(default_factory=list, description="Source files and pages used")


def create_app(service: Answerer | None = None) -> FastAPI:
    """Build the app; ``service`` can be injected (tests use a fake)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The vector store and the LLM client are created lazily on the first question.
        app.state.service = service or RagService()
        yield

    app = FastAPI(
        title="RAG API",
        description="Question answering grounded in your PDF documents.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/")
    def root() -> dict[str, str]:
        return {"status": "online", "mensagem": "RAG API funcionando!"}

    @app.post("/perguntar", response_model=Resposta)
    def perguntar(body: Pergunta) -> Resposta:
        question = body.pergunta.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Pergunta não pode ser vazia")
        try:
            answer = app.state.service.ask(question)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return Resposta(pergunta=question, resposta=answer.text, fontes=answer.sources)

    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn rag_api.api:app_factory --factory``."""
    return create_app()
