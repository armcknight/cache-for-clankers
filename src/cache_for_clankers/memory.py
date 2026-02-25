"""
MemoryManager: high-level API for storing and retrieving semantic memories.

This is the main entry-point for applications that want to persist
context across LLM sessions or compactions.

Usage example::

    from cache_for_clankers import MemoryManager

    memory = MemoryManager(db_path="./my_memory")

    # Store something important from a session
    ids = memory.store("The user's name is Alice and she prefers Python.")

    # Later, retrieve relevant context for a new prompt
    results = memory.retrieve("What programming language does the user prefer?")
    for r in results:
        print(r["content"], r["similarity"])
"""

from __future__ import annotations

import time
from typing import Any

from .intelligence import (
    SIMILARITY_THRESHOLD,
    chunk_text,
    compute_importance,
    deduplicate_content,
    generate_id,
)
from .store import VectorStore


class MemoryManager:
    """
    Intelligent memory manager backed by a local ChromaDB vector store.

    Responsibilities
    ----------------
    * **Store** – Accepts raw text, chunks it if it is long, scores each
      chunk for importance, and deduplicates against the existing store
      before writing.  Near-duplicate content is merged rather than stored
      twice, keeping the database lean.
    * **Retrieve** – Runs a semantic similarity search and re-ranks
      results by a combined score of similarity *and* importance so that
      the most useful memories bubble to the top.
    * **Manage** – Offers helpers to delete specific memories or list
      everything stored.

    Parameters
    ----------
    db_path:
        Filesystem path for the ChromaDB persistent store.
    collection_name:
        Name of the ChromaDB collection to use.
    similarity_threshold:
        Cosine-similarity threshold (0–1) above which two documents are
        treated as duplicates.  Defaults to 0.92.
    embedding_model:
        HuggingFace sentence-transformers model identifier used to embed
        text.  Defaults to ``"all-MiniLM-L6-v2"``.
    """

    def __init__(
        self,
        db_path: str = "./chroma_db",
        collection_name: str = "memories",
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        embedding_model: str = "all-MiniLM-L6-v2",
        _store: VectorStore | None = None,
    ) -> None:
        self._store = _store or VectorStore(
            path=db_path,
            collection_name=collection_name,
            embedding_model=embedding_model,
        )
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        auto_chunk: bool = True,
    ) -> list[str]:
        """
        Store *content* in memory with automatic chunking and deduplication.

        Long texts are split into semantic chunks before storage so that
        retrieval can pinpoint the most relevant fragment rather than
        returning a wall of text.

        Parameters
        ----------
        content:
            The text to remember.
        metadata:
            Optional extra key/value pairs attached to every stored chunk.
        session_id:
            Identifier for the originating LLM session.  Stored in metadata.
        auto_chunk:
            When ``True`` (default) and *content* is longer than the default
            chunk size, the text is split into multiple chunks.

        Returns
        -------
        list[str]
            IDs of the stored (or updated) memory entries.
        """
        base_meta: dict[str, Any] = {
            "timestamp": time.time(),
            "session_id": session_id or "unknown",
            "importance": compute_importance(content),
        }
        if metadata:
            base_meta.update(metadata)

        chunks = chunk_text(content) if auto_chunk and len(content) > 500 else [content]

        stored_ids: list[str] = []
        for chunk in chunks:
            chunk_meta = dict(base_meta)
            chunk_meta["importance"] = compute_importance(chunk)
            mem_id = self._store_with_dedup(chunk, chunk_meta)
            stored_ids.append(mem_id)

        return stored_ids

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        min_importance: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant memories for *query*.

        Results are re-ranked by a weighted combination of semantic
        similarity (70 %) and stored importance score (30 %) so that
        information-dense memories are preferred when relevance is close.

        Parameters
        ----------
        query:
            Natural-language question or statement to search against.
        n_results:
            Maximum number of memories to return.
        min_importance:
            Discard results whose stored importance score is below this
            value.  Useful for filtering out low-quality fragments.

        Returns
        -------
        list[dict]
            Each dict has keys: ``id``, ``content``, ``metadata``,
            ``similarity``.
        """
        if self._store.count() == 0:
            return []

        # Over-fetch so we have room to filter by importance.
        fetch_n = min(n_results * 3, self._store.count())
        results = self._store.query(query, n_results=fetch_n)

        memories: list[dict[str, Any]] = []
        docs = results["documents"][0]
        ids = results["ids"][0]
        metadatas = results.get("metadatas") or [[{}] * len(docs)]
        distances = results["distances"][0]

        for i, doc in enumerate(docs):
            meta = metadatas[0][i] if isinstance(metadatas[0], list) else {}
            importance = float(meta.get("importance", 0.0))
            if importance < min_importance:
                continue
            similarity = 1.0 - distances[i]
            memories.append(
                {
                    "id": ids[i],
                    "content": doc,
                    "metadata": meta,
                    "similarity": similarity,
                }
            )

        # Re-rank: 70 % similarity, 30 % importance.
        memories.sort(
            key=lambda m: m["similarity"] * 0.7
            + float(m["metadata"].get("importance", 0.0)) * 0.3,
            reverse=True,
        )
        return memories[:n_results]

    def delete(self, memory_id: str) -> None:
        """Delete a memory by its ID."""
        self._store.delete(memory_id)

    def count(self) -> int:
        """Return the total number of stored memories."""
        return self._store.count()

    def list_all(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Return up to *limit* stored memories (no ranking applied).

        Each dict has keys: ``id``, ``content``, ``metadata``.
        """
        result = self._store.get_all()
        ids = result.get("ids") or []
        docs = result.get("documents") or []
        metas = result.get("metadatas") or [{}] * len(docs)

        memories = [
            {"id": ids[i], "content": docs[i], "metadata": metas[i]}
            for i in range(len(docs))
        ]
        return memories[:limit]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _store_with_dedup(self, content: str, metadata: dict[str, Any]) -> str:
        """
        Write *content* to the store, merging with a near-duplicate if one
        exists.

        Near-duplicate detection uses ``deduplicate_content`` with the
        configured ``similarity_threshold``.  When a duplicate is found
        the existing entry is updated with the longer of the two texts
        (on the assumption that more text carries more information).

        Returns the ID of the stored or updated memory.
        """
        if self._store.count() == 0:
            new_id = generate_id()
            self._store.add(new_id, content, metadata)
            return new_id

        results = self._store.query(content, n_results=1)
        distances = results["distances"][0] if results["distances"] else []

        is_dup, best_idx = deduplicate_content(
            content,
            results["documents"][0] if results["documents"] else [],
            distances,
            self.similarity_threshold,
        )

        if is_dup and best_idx is not None:
            existing_id = results["ids"][0][best_idx]
            existing_doc = results["documents"][0][best_idx]
            # Keep the more complete version.
            merged = content if len(content) >= len(existing_doc) else existing_doc
            self._store.update(existing_id, merged, metadata)
            return existing_id

        new_id = generate_id()
        self._store.add(new_id, content, metadata)
        return new_id
