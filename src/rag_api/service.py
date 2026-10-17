"""The RAG engine. Heavy dependencies (LangChain, Chroma, Groq) are imported lazily."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Protocol

from .config import Settings
from .prompting import build_prompt, extract_sources


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[str]


class Answerer(Protocol):
    def ask(self, question: str) -> Answer: ...


class RagService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._retriever = None
        self._llm = None

    def _embeddings(self):
        from langchain_community.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=self.settings.embedding_model)

    def ingest(self) -> int:
        """(Re)build the vector index from every PDF in ``docs_dir``. Returns the chunk count."""
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.vectorstores import Chroma
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        pdfs = sorted(self.settings.docs_dir.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDF files found in {self.settings.docs_dir}/")
        pages = [page for pdf in pdfs for page in PyPDFLoader(str(pdf)).load()]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size, chunk_overlap=self.settings.chunk_overlap
        )
        chunks = splitter.split_documents(pages)
        if self.settings.persist_dir.exists():
            shutil.rmtree(self.settings.persist_dir)
        Chroma.from_documents(chunks, self._embeddings(), persist_directory=str(self.settings.persist_dir))
        self._retriever = None
        return len(chunks)

    def _load(self) -> None:
        from langchain_community.vectorstores import Chroma
        from langchain_groq import ChatGroq

        if not self.settings.persist_dir.exists():
            self.ingest()
        store = Chroma(persist_directory=str(self.settings.persist_dir), embedding_function=self._embeddings())
        self._retriever = store.as_retriever(search_kwargs={"k": self.settings.top_k})
        self._llm = ChatGroq(model=self.settings.llm_model, temperature=0)

    def ask(self, question: str) -> Answer:
        if self._retriever is None:
            self._load()
        chunks = self._retriever.invoke(question)
        reply = self._llm.invoke(build_prompt(question, chunks))
        return Answer(text=str(reply.content).strip(), sources=extract_sources(chunks))
