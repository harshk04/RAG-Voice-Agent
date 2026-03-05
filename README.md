# RAG Voice Agent

A LiveKit-based voice assistant with Deepgram STT/TTS and optional RAG (Qdrant + OpenAI embeddings).

## Requirements

- Python 3.10+
- Docker (for local Qdrant)
- API keys:
  - `DEEPGRAM_API_KEY`
  - `OPENAI_API_KEY`
  - `GROQ_API_KEY`
  - LiveKit credentials if using LiveKit cloud

## Setup

1. Create a virtual environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Create `.env` (see `.env.example`) and set required variables:

```
DEEPGRAM_API_KEY=...
OPENAI_API_KEY=...
GROQ_API_KEY=...
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=voice_agent_docs
```

## Start Qdrant (Local)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

If you use Qdrant Cloud, set `QDRANT_URL` to the HTTPS endpoint and `QDRANT_API_KEY`.

## Ingest Documents

Put your documents under `docs/` (txt, md, pdf, json supported), then run:

```bash
python -m rag.ingest --path docs/
```

Notes:
- Embeddings use OpenAI (`EMBEDDING_MODEL` in `.env.example`).
- Chunking defaults: 500 tokens, 50 overlap.

## Test Retrieval

```bash
python -m rag.test_retrieval "What is the refund policy?"
```

Expected: prints retrieved chunks (or "No chunks retrieved" if none found).

## Run the Voice Agent

```bash
python agent.py
```

When running, the agent:
- Uses Deepgram for STT and TTS (English only).
- Calls RAG only for information‑seeking queries (heuristic gate).
- Injects retrieved context before the LLM generates a response.

## RAG Flow (Quick)

1. User speaks → Deepgram STT → text
2. `_should_use_rag()` decides if retrieval is needed
3. If yes: embed query → Qdrant search → retrieve chunks
4. Inject context into a temporary system message
5. LLM answers → Deepgram TTS speaks

## Troubleshooting

- **RAG disabled**: check `OPENAI_API_KEY` and `QDRANT_URL`.
- **Qdrant warnings**: if using `QDRANT_API_KEY` with `http://`, you’ll see a warning. Use HTTPS or remove the key for local.
- **PDF warnings**: `pypdf` may warn about malformed PDFs; try another PDF or use txt/md.
- **No chunks retrieved**: ensure ingestion succeeded and docs contain the query text.

## Useful Commands

```bash
# Ingest
python -m rag.ingest --path docs/

# Test retrieval
python -m rag.test_retrieval "Your question here"

# Run agent
python agent.py
```

## Project Structure

- `agent.py` — LiveKit agent entrypoint + RAG hook
- `prompts.py` — System instructions and greeting
- `rag/` — Embeddings, retrieval, ingestion
- `docs/` — Your knowledge base files
