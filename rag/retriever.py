"""RAG retrieval utilities."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import List

from rag.embeddings import get_embedding
from rag.qdrant_service import get_collection_name, get_qdrant_client

logger = logging.getLogger(__name__)


def _rag_debug_enabled() -> bool:
    """Return True if RAG debug logging is enabled."""
    return os.getenv("RAG_DEBUG", "false").strip().lower() in {"1", "true", "yes", "y"}


def _get_top_k(default: int = 5) -> int:
    """Return the configured top-k value."""
    value = os.getenv("RAG_TOP_K")
    if value is None:
        return default
    try:
        return max(1, int(value))
    except ValueError:
        logger.warning("Invalid RAG_TOP_K=%s, using default %d", value, default)
        return default


def _get_max_context_chars(default: int = 4000) -> int:
    """Return the configured maximum context length."""
    value = os.getenv("MAX_CONTEXT_CHARS")
    if value is None:
        return default
    try:
        return max(200, int(value))
    except ValueError:
        logger.warning("Invalid MAX_CONTEXT_CHARS=%s, using default %d", value, default)
        return default


def format_context(chunks: list[str]) -> str:
    """Format retrieved chunks into a single context block."""
    context = "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    max_chars = _get_max_context_chars()
    if len(context) <= max_chars:
        return context
    truncated = context[:max_chars]
    cut = truncated.rfind(" ")
    if cut > max_chars - 200:
        truncated = truncated[:cut]
    return truncated.rstrip()


async def retrieve_context(query: str, top_k: int | None = None) -> List[str]:
    """Retrieve relevant context chunks for a query.

    Args:
        query: The user query text.
        top_k: Number of results to retrieve.

    Returns:
        A list of retrieved text chunks.
    """
    if not query:
        return []

    client = get_qdrant_client()
    collection = get_collection_name()
    rag_debug = _rag_debug_enabled()
    limit = top_k if top_k is not None else _get_top_k()

    try:
        embed_start = time.perf_counter()
        query_embedding = await get_embedding(query)
        embed_duration = time.perf_counter() - embed_start
        logger.debug("Query embedding time: %.4fs", embed_duration)
        if rag_debug:
            logger.info("Query embedding vector size: %d", len(query_embedding))

        search_start = time.perf_counter()
        if hasattr(client, "search"):
            results = await asyncio.to_thread(
                client.search,
                collection_name=collection,
                query_vector=query_embedding,
                limit=limit,
                with_payload=True,
            )
            points = results
        else:
            response = await asyncio.to_thread(
                client.query_points,
                collection_name=collection,
                query=query_embedding,
                limit=limit,
                with_payload=True,
            )
            points = response.points
        search_duration = time.perf_counter() - search_start
        logger.debug("Qdrant latency: %.4fs", search_duration)
        if rag_debug:
            logger.info("Qdrant latency: %.4fs", search_duration)

        chunks: list[str] = []
        for item in points:
            payload = item.payload or {}
            text = payload.get("text") if isinstance(payload, dict) else None
            if isinstance(text, str) and text.strip():
                chunks.append(text)

        logger.debug("Retrieved documents count: %d", len(chunks))
        if rag_debug:
            logger.info("Retrieved chunks: %s", chunks)
        return chunks
    except Exception as exc:
        logger.warning("RAG retrieval failed, continuing without context: %s", exc)
        return []
