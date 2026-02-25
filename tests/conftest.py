"""
Shared pytest fixtures for cache-for-clankers tests.

Uses ChromaDB in ephemeral (in-memory) mode and a deterministic
fake embedding function so that tests run fast without downloading
any ML models.
"""

from __future__ import annotations

import hashlib
import uuid

import chromadb
import pytest

from cache_for_clankers.memory import MemoryManager
from cache_for_clankers.store import VectorStore


class FakeEmbeddingFunction:
    """
    Deterministic embedding function that maps text to a unit vector
    derived from its MD5 hash.  Fast and reproducible – no model download.
    Implements both the legacy ``__call__`` interface and the newer
    ``embed_documents`` / ``embed_query`` interface used by ChromaDB ≥ 0.5.
    """

    def name(self) -> str:  # required by ChromaDB >= 0.5
        return "fake-md5-embedding"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            digest = hashlib.md5(text.encode()).digest()
            # 16-byte digest → 16-dim float vector in [-1, 1]
            vec = [(b - 128) / 128.0 for b in digest]
            norm = sum(x * x for x in vec) ** 0.5 or 1.0
            embeddings.append([x / norm for x in vec])
        return embeddings

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self._embed(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self._embed(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self._embed(input)


# A single shared EphemeralClient instance for the test session.
# Each fixture call creates a uniquely named collection so tests are isolated.
_EPHEMERAL_CLIENT = chromadb.EphemeralClient()


@pytest.fixture()
def ephemeral_store() -> VectorStore:
    """In-memory VectorStore with the fake embedding function.

    A unique collection name is used per fixture invocation so that tests
    cannot interfere with each other despite sharing the same EphemeralClient.
    """
    collection_name = f"test_{uuid.uuid4().hex}"
    return VectorStore(
        _client=_EPHEMERAL_CLIENT,
        collection_name=collection_name,
        _embedding_function=FakeEmbeddingFunction(),
    )


@pytest.fixture()
def memory_manager(ephemeral_store: VectorStore) -> MemoryManager:
    """MemoryManager wired to the ephemeral in-memory store."""
    return MemoryManager(_store=ephemeral_store)
