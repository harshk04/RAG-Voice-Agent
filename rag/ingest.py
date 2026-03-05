"""Ingest documents into Qdrant for RAG retrieval."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from dotenv import load_dotenv
from langchain_text_splitters import TokenTextSplitter
from pypdf import PdfReader
from qdrant_client.http import models as qdrant_models

from rag.embeddings import get_embeddings
from rag.qdrant_service import (
    create_collection_if_not_exists,
    get_collection_name,
    get_qdrant_client,
)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".json"}


@dataclass
class Document:
    """Simple document representation for ingestion."""
    text: str
    source: str


@dataclass
class Chunk:
    """Chunked document text with source metadata."""
    text: str
    source: str
    chunk_index: int


def _read_text_file(path: Path) -> str:
    """Read a UTF-8 text file safely."""
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _extract_json_texts(data: object) -> list[str]:
    """Extract text blobs from JSON data."""
    if isinstance(data, str):
        return [data]
    if isinstance(data, list):
        texts: list[str] = []
        for item in data:
            texts.extend(_extract_json_texts(item))
        return texts
    if isinstance(data, dict):
        for key in ("text", "content", "body"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return [value]
        return [json.dumps(data, ensure_ascii=True)]
    return [str(data)]


def _read_json_file(path: Path) -> list[str]:
    """Read a JSON file and extract text entries."""
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    return [text for text in _extract_json_texts(data) if text.strip()]


def _load_documents(path: Path) -> list[Document]:
    """Load documents from a file or directory path."""
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    else:
        files = [path]

    documents: list[Document] = []
    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue
        source = str(file_path)
        if suffix in {".txt", ".md"}:
            text = _read_text_file(file_path)
            if text.strip():
                documents.append(Document(text=text, source=source))
        elif suffix == ".pdf":
            text = _read_pdf_file(file_path)
            if text.strip():
                documents.append(Document(text=text, source=source))
        elif suffix == ".json":
            for item in _read_json_file(file_path):
                documents.append(Document(text=item, source=source))

    return documents


def _chunk_documents(documents: Iterable[Document]) -> list[Chunk]:
    """Split documents into chunks using a token-based splitter."""
    splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks: list[Chunk] = []
    for doc in documents:
        pieces = splitter.split_text(doc.text)
        for idx, piece in enumerate(pieces):
            if piece.strip():
                chunks.append(Chunk(text=piece, source=doc.source, chunk_index=idx))
    return chunks


async def _embed_chunks(chunks: list[Chunk], batch_size: int = 64) -> list[list[float]]:
    """Embed chunks in batches."""
    embeddings: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_texts = [chunk.text for chunk in batch]
        batch_embeddings = await get_embeddings(batch_texts)
        embeddings.extend(batch_embeddings)
        logger.info("Embedded %d/%d chunks", len(embeddings), len(chunks))
    return embeddings


async def ingest(path: Path, batch_size: int = 64) -> None:
    """Ingest documents from a path into Qdrant.

    Args:
        path: File or directory path containing documents.
        batch_size: Embedding batch size.
    """
    documents = _load_documents(path)
    if not documents:
        raise ValueError(f"No supported documents found at {path}")

    chunks = _chunk_documents(documents)
    if not chunks:
        raise ValueError("No chunks created from documents")

    embeddings = await _embed_chunks(chunks, batch_size=batch_size)
    if not embeddings:
        raise ValueError("No embeddings generated")

    client = get_qdrant_client()
    collection = get_collection_name()

    create_collection_if_not_exists(client, collection, vector_size=len(embeddings[0]))

    points: list[qdrant_models.PointStruct] = []
    for chunk, vector in zip(chunks, embeddings):
        points.append(
            qdrant_models.PointStruct(
                id=uuid.uuid4().hex,
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "created_at": time.time(),
                },
            )
        )

    client.upsert(collection_name=collection, points=points)
    logger.info("Ingested %d chunks into collection '%s'", len(points), collection)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ingestion."""
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant.")
    parser.add_argument("--path", required=True, help="Path to docs file or directory")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
        help="Embedding batch size (default: 64)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    asyncio.run(ingest(Path(args.path), batch_size=args.batch_size))


if __name__ == "__main__":
    main()
