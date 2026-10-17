# RAG API: Question Answering over PDFs

![CI](https://github.com/devbelloni/RAG-with-FastAPI/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![License](https://img.shields.io/badge/license-MIT-green)

A Retrieval-Augmented Generation (RAG) service that indexes your PDF documents and answers
natural-language questions **grounded only in their content**, citing the file and page of
each source. Use it from the terminal or as a REST API with automatic Swagger docs.

> Portuguese version: [README.pt-BR.md](README.pt-BR.md)

## How it works

```
PDFs ─► chunking (1500 chars, 200 overlap)
     ─► local embeddings (all-MiniLM-L6-v2, HuggingFace)
     ─► ChromaDB (persistent local vector store)

question ─► top-k semantic search ─► grounded prompt ─► Llama 3.3 70B (Groq) ─► answer + sources
```

Design choices:

- **Local embeddings and vector store**: only the generation step calls an external API.
- **Grounded answers**: the prompt tells the model to answer only from the retrieved context
  and to say when nothing relevant was found, which limits hallucination.
- **Source citations**: every answer lists the documents and pages that were retrieved.
- **Lazy loading**: heavy libraries and the index load on the first question, so the API
  starts fast and its HTTP layer is testable without them.
- **Injected service**: the API depends on a small `Answerer` interface, so tests run with a fake.
- **Configurable** through environment variables (chunk size, overlap, top-k, models, paths).

## Quick start

```bash
git clone https://github.com/devbelloni/RAG-with-FastAPI.git && cd RAG-with-FastAPI
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

cp .env.example .env          # add your GROQ_API_KEY (free at console.groq.com)
cp your_documents/*.pdf docs/ # put PDFs in ./docs
```

```bash
rag-api ingest                # build the vector index
rag-api chat                  # ask questions in the terminal
rag-api serve                 # REST API at http://127.0.0.1:8000/docs
```

Or with Docker:

```bash
docker build -t rag-api . && docker run -p 8000:8000 --env-file .env -v "$PWD/docs:/app/docs" rag-api
```

## API

`GET /` returns the service status.

`POST /perguntar` asks a question:

```json
{ "pergunta": "What is the conclusion of the study?" }
```

```json
{
  "pergunta": "What is the conclusion of the study?",
  "resposta": "According to the documents, ...",
  "fontes": ["relatorio.pdf (page 12)", "relatorio.pdf (page 14)"]
}
```

Errors: `400` for an empty question, `422` for invalid input, `503` when there are no PDFs to index.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | required | LLM access |
| `RAG_DOCS_DIR` | `docs` | Folder with PDFs |
| `RAG_PERSIST_DIR` | `chroma_db` | Vector store location |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `1500` / `200` | Chunking |
| `RAG_TOP_K` | `6` | Chunks retrieved per question |
| `RAG_LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model |

## Tests

```bash
pytest -q && ruff check .
```

Nine tests cover prompt construction, source extraction, input validation, error mapping
and configuration. They use a fake service, so they need no model, index or API key.
The ingestion and LLM layer (`service.py`) needs the real dependencies and a Groq key, and
is exercised by running the application.

## Layout

```
src/rag_api/
├── api.py        # FastAPI app factory
├── service.py    # ingestion, retrieval and generation
├── prompting.py  # prompt and source formatting (pure functions)
├── config.py     # settings from environment
└── cli.py        # ingest / chat / serve
tests/
```

## Roadmap

- [ ] More formats (DOCX, TXT, CSV)
- [ ] Retrieval evaluation (precision@k) with a labelled question set
- [ ] API authentication and rate limiting
- [ ] Streamlit front end

## Author

**Márcio Belloni**, AI/ML Engineer.
[LinkedIn](https://www.linkedin.com/in/marcio-belloni/) · [GitHub](https://github.com/devbelloni)

## License

MIT, see [LICENSE](LICENSE).
