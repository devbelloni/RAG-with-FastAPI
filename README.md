# RAG API: Question Answering over PDFs with LangChain, ChromaDB and Groq

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C)
![License](https://img.shields.io/badge/license-MIT-green)

A Retrieval-Augmented Generation (RAG) service that indexes PDF documents and answers
natural-language questions grounded **only** in their content. It runs as a REST API
(FastAPI, with automatic Swagger docs) or as an interactive terminal app.

> Portuguese version: [README.pt-BR.md](README.pt-BR.md)

## How it works

```
PDF files ──► chunking (1500 chars, 200 overlap)
          ──► embeddings: all-MiniLM-L6-v2 (HuggingFace, local, no API cost)
          ──► ChromaDB (local vector store)
          ──► top-6 semantic search
          ──► prompt enriched with retrieved context
          ──► Llama 3.3 70B via Groq
          ──► grounded answer
```

Design choices:

- **Local embeddings and vector store**: only the generation step calls an external API.
- **Grounded answers**: the prompt instructs the model to answer only from the retrieved
  context and to say so when nothing relevant is found, which reduces hallucination.
- **Tuned retrieval**: chunk size and overlap were increased (500/50 to 1500/200) and `k`
  raised from 3 to 6 to keep more context at chunk boundaries.

## Stack

| Tool | Role |
|---|---|
| LangChain | RAG pipeline orchestration |
| Groq (Llama 3.3 70B) | LLM inference |
| ChromaDB | Vector database |
| HuggingFace sentence-transformers | Embeddings |
| FastAPI + Uvicorn | REST API |

## Quick start

```bash
git clone https://github.com/devbelloni/RAG-with-FastAPI.git
cd RAG-with-FastAPI
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # add your GROQ_API_KEY (free at console.groq.com)
# put your PDFs in ./docs
```

Terminal mode (also builds the index):

```bash
python main.py
```

API mode:

```bash
uvicorn api:app --reload
# Swagger UI: http://localhost:8000/docs
```

## API

`GET /` returns the service status.

`POST /perguntar` asks a question about the indexed documents:

```json
{ "pergunta": "What is the conclusion of the study?" }
```

```json
{ "pergunta": "What is the conclusion of the study?", "resposta": "According to the documents, ..." }
```

## Configuration

| Parameter | Value | Description |
|---|---|---|
| `chunk_size` | 1500 | Characters per chunk |
| `chunk_overlap` | 200 | Overlap between chunks |
| `k` | 6 | Chunks retrieved per query |
| LLM | `llama-3.3-70b-versatile` | Generation model |
| Embeddings | `all-MiniLM-L6-v2` | Local embedding model |

## Roadmap

- [ ] More formats (DOCX, TXT, CSV)
- [ ] Streamlit front end
- [ ] Multiple document collections
- [ ] API authentication
- [ ] Deployment (Railway or Render)
- [ ] Retrieval evaluation (precision@k) and automated tests

## Security

`GROQ_API_KEY` is read from `.env`, which is git-ignored. The `chroma_db/` index is
generated locally and is not versioned.

## Author

**Márcio Belloni**, AI/ML Engineer.
[LinkedIn](https://www.linkedin.com/in/marcio-belloni/) · [GitHub](https://github.com/devbelloni)

## License

MIT, see [LICENSE](LICENSE).
