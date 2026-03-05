"""Embedding helpers for the RAG pipeline."""
from __future__ import annotations

import asyncio
import os
from functools import lru_cache
from typing import List

import aiohttp

from livekit.plugins import openai as openai_plugin


def _get_embedding_model() -> str:
    """Return the embedding model name from the environment or default."""
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


def _require_openai_key() -> None:
    """Ensure the OpenAI API key is available."""
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be set to create embeddings")


async def _compute_embedding(text: str) -> List[float]:
    """Compute a single embedding without caching."""
    async with aiohttp.ClientSession() as session:
        results = await openai_plugin.create_embeddings(
            input=[text],
            model=_get_embedding_model(),
            http_session=session,
        )
        return results[0].embedding


@lru_cache(maxsize=512)
def _embedding_task(text: str) -> asyncio.Task[List[float]]:
    """Create (or reuse) an embedding task for the given text."""
    loop = asyncio.get_running_loop()
    return loop.create_task(_compute_embedding(text))


async def get_embedding(text: str) -> List[float]:
    """Generate an embedding for a single text input.

    Args:
        text: The text to embed.

    Returns:
        The embedding vector.
    """
    _require_openai_key()
    task = _embedding_task(text)
    try:
        return await task
    except Exception:
        _embedding_task.cache_clear()
        raise


async def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text inputs.

    Args:
        texts: The texts to embed.

    Returns:
        A list of embedding vectors in the same order.
    """
    _require_openai_key()
    async with aiohttp.ClientSession() as session:
        results = await openai_plugin.create_embeddings(
            input=texts,
            model=_get_embedding_model(),
            http_session=session,
        )
        return [item.embedding for item in results]
