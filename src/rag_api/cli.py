"""Command line: ingest PDFs, chat in the terminal, or serve the API."""

from __future__ import annotations

import argparse

from .config import Settings
from .service import RagService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Build the vector index from docs/*.pdf")
    sub.add_parser("chat", help="Ask questions in the terminal")
    serve = sub.add_parser("serve", help="Run the REST API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run("rag_api.api:app_factory", factory=True, host=args.host, port=args.port)
        return 0

    service = RagService(Settings.from_env())
    if args.command == "ingest":
        print(f"Indexed {service.ingest()} chunks.")
        return 0
    print("RAG ready. Type a question, or 'sair' to quit.")
    while (question := input("\nVocê: ").strip()).lower() != "sair":
        if question:
            answer = service.ask(question)
            print(f"\nRAG: {answer.text}\nFontes: {', '.join(answer.sources) or 'n/a'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
