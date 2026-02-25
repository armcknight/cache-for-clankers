"""
Vector store wrapper around ChromaDB for persistent semantic memory.
"""

from __future__ import annotations

from typing import Any

import chromadb
from chromadb.utils import embedding_functions


def get_embedding_function(
    model_name: str = "all-MiniLM-L6-v2",
) -> embedding_functions.SentenceTransformerEmbeddingFunction:
    """Return a sentence-transformer embedding function for ChromaDB."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )


class VectorStore:
    """
    Persistent vector store backed by ChromaDB.

    Uses cosine similarity so that distance values returned by queries
    are in the range [0, 2]:
        distance = 1 - cosine_similarity
        cosine_similarity ∈ [-1, 1]  →  distance ∈ [0, 2]
    """

    def __init__(
        self,
        path: str = "./chroma_db",
        collection_name: str = "memories",
        embedding_model: str = "all-MiniLM-L6-v2",
        _client: chromadb.ClientAPI | None = None,
        _embedding_function: Any | None = None,
    ) -> None:
        self.client = _client or chromadb.PersistentClient(path=path)
        ef = _embedding_function or get_embedding_function(embedding_model)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, id: str, document: str, metadata: dict | None = None) -> None:
        """Add a new document."""
        self.collection.add(
            ids=[id],
            documents=[document],
            metadatas=[metadata] if metadata else None,
        )

    def update(self, id: str, document: str, metadata: dict | None = None) -> None:
        """Update an existing document."""
        self.collection.update(
            ids=[id],
            documents=[document],
            metadatas=[metadata] if metadata else None,
        )

    def delete(self, id: str) -> None:
        """Delete a document by ID."""
        self.collection.delete(ids=[id])

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(self, query_text: str, n_results: int = 5) -> dict:
        """
        Query the collection by semantic similarity.

        Returns a ChromaDB result dict with keys:
            ids, documents, metadatas, distances
        """
        n = min(n_results, self.count())
        if n == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        return self.collection.query(
            query_texts=[query_text],
            n_results=n,
        )

    def get(self, id: str) -> dict:
        """Fetch a single document by ID."""
        return self.collection.get(ids=[id])

    def get_all(self) -> dict:
        """Return all documents in the collection."""
        return self.collection.get()

    def count(self) -> int:
        """Return the total number of stored documents."""
        return self.collection.count()
