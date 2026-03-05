"""Qdrant client helpers for the RAG pipeline."""
from __future__ import annotations

import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

logger = logging.getLogger(__name__)


def get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client using environment configuration.

    Returns:
        An initialized QdrantClient.
    """
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    return QdrantClient(url=url, api_key=api_key)


def get_collection_name() -> str:
    """Return the Qdrant collection name from the environment or default."""
    return os.getenv("QDRANT_COLLECTION", "voice_agent_docs")


def create_collection_if_not_exists(
    client: QdrantClient,
    collection: str,
    vector_size: int,
) -> None:
    """Create a Qdrant collection if it does not already exist.

    Args:
        client: The Qdrant client.
        collection: The collection name.
        vector_size: Embedding vector size.
    """
    collections = client.get_collections().collections
    existing = {c.name for c in collections}
    if collection in existing:
        return
    client.create_collection(
        collection_name=collection,
        vectors_config=qdrant_models.VectorParams(
            size=vector_size,
            distance=qdrant_models.Distance.COSINE,
        ),
    )


def check_qdrant_connection() -> None:
    """Check whether Qdrant is reachable."""
    client = get_qdrant_client()
    client.get_collections()
    logger.info("Qdrant connection successful")
