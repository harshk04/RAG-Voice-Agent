"""Quick retrieval test for the RAG pipeline."""
from __future__ import annotations

import asyncio
import sys
from typing import List

from dotenv import load_dotenv

from rag.retriever import retrieve_context


def _print_chunks(chunks: List[str]) -> None:
    """Print retrieved chunks to stdout."""
    if not chunks:
        print("No chunks retrieved")
        return
    for idx, chunk in enumerate(chunks, start=1):
        print(f"--- Chunk {idx} ---")
        print(chunk)
        print()


async def _run(query: str) -> None:
    """Execute retrieval for the provided query."""
    chunks = await retrieve_context(query)
    _print_chunks(chunks)


def main() -> None:
    """CLI entrypoint."""
    load_dotenv(override=True)
    if len(sys.argv) < 2:
        print('Usage: python rag/test_retrieval.py "Your query here"')
        sys.exit(1)
    query = " ".join(sys.argv[1:]).strip()
    asyncio.run(_run(query))


if __name__ == "__main__":
    main()
