# Production RAG System

A Retrieval-Augmented Generation (RAG) system built with FastAPI, Qdrant, Ollama, and Docker.

## Features

* PDF document ingestion
* Automatic text chunking
* Embedding generation using Sentence Transformers
* Vector search with Qdrant
* Local LLM inference with Ollama (Llama 3)
* Retrieval-Augmented Generation (RAG)
* FastAPI backend
* Streaming responses using Server-Sent Events (SSE)
* Docker Compose deployment
* Source attribution for generated answers

## Architecture

PDF Documents
→ Chunking
→ Embeddings
→ Qdrant Vector Store
→ Retrieval
→ Prompt Construction
→ Ollama (Llama 3)
→ Response Generation

## Project Structure

src/
├── api/
├── core/
├── rag/
│ ├── embeddings/
│ ├── generation/
│ └── retrieval/
├── services/
└── scripts/

frontend/
docker-compose.yml
Dockerfile

## Prerequisites

* Docker
* Docker Compose

## Quick Start

### Build and Start Services

```bash
docker-compose up --build
```

### Pull Llama 3

```bash
docker exec -it ollama bash
ollama pull llama3
```

### API Documentation

```text
http://localhost:8000/docs
```

### Frontend

Open:

```text
frontend/index.html
```

## API Endpoints

### Query

POST /query

Request:

```json
{
  "query": "What is federated learning?"
}
```

### Streaming

POST /stream

Returns a streaming SSE response.

## Current Limitations

* Basic conversation memory
* Vector retrieval only
* No reranking
* No query rewriting
* Minimal frontend UI

## Planned Improvements

* Conversation-aware retrieval
* Query rewriting
* Hybrid search (BM25 + Vector)
* Reranking
* Enhanced frontend UI
* Production monitoring
* Cloud deployment
