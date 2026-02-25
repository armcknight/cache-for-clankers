"""Tests for the VectorStore ChromaDB wrapper."""

from __future__ import annotations

import pytest

from cache_for_clankers.store import VectorStore


class TestVectorStore:
    def test_initial_count_is_zero(self, ephemeral_store: VectorStore):
        assert ephemeral_store.count() == 0

    def test_add_increases_count(self, ephemeral_store: VectorStore):
        ephemeral_store.add("id1", "Hello world")
        assert ephemeral_store.count() == 1

    def test_add_and_get(self, ephemeral_store: VectorStore):
        ephemeral_store.add("id1", "Hello world", metadata={"key": "val"})
        result = ephemeral_store.get("id1")
        assert result["ids"] == ["id1"]
        assert result["documents"] == ["Hello world"]
        assert result["metadatas"][0]["key"] == "val"

    def test_update_changes_document(self, ephemeral_store: VectorStore):
        ephemeral_store.add("id1", "Original text")
        ephemeral_store.update("id1", "Updated text", metadata={"v": 2})
        result = ephemeral_store.get("id1")
        assert result["documents"] == ["Updated text"]
        assert result["metadatas"][0]["v"] == 2

    def test_delete_removes_document(self, ephemeral_store: VectorStore):
        ephemeral_store.add("id1", "To be deleted")
        ephemeral_store.delete("id1")
        assert ephemeral_store.count() == 0

    def test_query_on_empty_store_returns_empty(self, ephemeral_store: VectorStore):
        result = ephemeral_store.query("anything", n_results=5)
        assert result["ids"] == [[]]
        assert result["documents"] == [[]]

    def test_query_returns_documents(self, ephemeral_store: VectorStore):
        ephemeral_store.add("a", "Python programming language")
        ephemeral_store.add("b", "JavaScript web development")
        ephemeral_store.add("c", "Machine learning with neural networks")
        result = ephemeral_store.query("programming", n_results=2)
        assert len(result["ids"][0]) == 2
        assert len(result["documents"][0]) == 2

    def test_get_all_returns_all_documents(self, ephemeral_store: VectorStore):
        ephemeral_store.add("x", "First")
        ephemeral_store.add("y", "Second")
        result = ephemeral_store.get_all()
        assert len(result["ids"]) == 2

    def test_query_n_results_capped_at_store_size(self, ephemeral_store: VectorStore):
        ephemeral_store.add("only", "Single document")
        # Asking for more than exist should not error and returns what's available.
        result = ephemeral_store.query("document", n_results=10)
        assert len(result["ids"][0]) == 1
