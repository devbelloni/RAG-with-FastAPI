FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
ENV PYTHONPATH=/app/src
RUN useradd --system --create-home rag && mkdir /app/chroma_db && chown rag /app/chroma_db
USER rag

EXPOSE 8000
CMD ["uvicorn", "rag_api.api:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
