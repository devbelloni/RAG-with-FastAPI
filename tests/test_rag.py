from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rag_api.api import create_app
from rag_api.config import Settings
from rag_api.prompting import PROMPT_TEMPLATE, build_prompt, extract_sources, format_context
from rag_api.service import Answer


def chunk(text, source="docs/relatorio.pdf", page=0):
    return SimpleNamespace(page_content=text, metadata={"source": source, "page": page})


class FakeService:
    def __init__(self, error=None):
        self.questions, self.error = [], error

    def ask(self, question):
        if self.error:
            raise self.error
        self.questions.append(question)
        return Answer(text="42", sources=["relatorio.pdf (page 1)"])


def client(service):
    return TestClient(create_app(service))


def test_prompt_contains_question_context_and_grounding_rule():
    prompt = build_prompt("O que é RAG?", [chunk("RAG combina busca e geração.")])
    assert "O que é RAG?" in prompt and "RAG combina busca e geração." in prompt
    assert "APENAS no contexto" in PROMPT_TEMPLATE


def test_format_context_separates_chunks():
    assert format_context([chunk("a"), chunk("b")]) == "a\n\n---\n\nb"


def test_sources_are_unique_ordered_and_one_based():
    chunks = [chunk("x", "docs/a.pdf", 0), chunk("y", "C:\\docs\\b.pdf", 4), chunk("z", "docs/a.pdf", 0)]
    assert extract_sources(chunks) == ["a.pdf (page 1)", "b.pdf (page 5)"]
    assert extract_sources([SimpleNamespace(page_content="t", metadata={})]) == ["unknown"]


def test_root_and_successful_question():
    service = FakeService()
    with client(service) as c:
        assert c.get("/").json()["status"] == "online"
        r = c.post("/perguntar", json={"pergunta": "  Qual a conclusão?  "})
    assert r.status_code == 200
    assert r.json() == {"pergunta": "Qual a conclusão?", "resposta": "42", "fontes": ["relatorio.pdf (page 1)"]}
    assert service.questions == ["Qual a conclusão?"]


@pytest.mark.parametrize("payload,status", [({"pergunta": "   "}, 400), ({}, 422), ({"pergunta": "x" * 2001}, 422)])
def test_input_validation(payload, status):
    with client(FakeService()) as c:
        assert c.post("/perguntar", json=payload).status_code == status


def test_missing_documents_returns_503():
    with client(FakeService(error=FileNotFoundError("No PDF files found in docs/"))) as c:
        r = c.post("/perguntar", json={"pergunta": "oi"})
    assert r.status_code == 503 and "No PDF files" in r.json()["detail"]


def test_settings_from_env():
    s = Settings.from_env({"RAG_TOP_K": "3", "RAG_CHUNK_SIZE": "500", "RAG_DOCS_DIR": "pdfs"})
    assert (s.top_k, s.chunk_size, str(s.docs_dir)) == (3, 500, "pdfs")
    assert Settings.from_env({}).top_k == 6
