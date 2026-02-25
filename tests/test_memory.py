"""Tests for MemoryManager – the main high-level API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cache_for_clankers.memory import MemoryManager
from cache_for_clankers.store import VectorStore


class TestMemoryManagerStore:
    def test_store_returns_list_of_ids(self, memory_manager: MemoryManager):
        ids = memory_manager.store("Remember that the user likes coffee.")
        assert isinstance(ids, list)
        assert len(ids) >= 1

    def test_store_increments_count(self, memory_manager: MemoryManager):
        memory_manager.store("First memory.")
        assert memory_manager.count() == 1

    def test_store_with_session_id(self, memory_manager: MemoryManager):
        ids = memory_manager.store("Session test.", session_id="sess-123")
        memories = memory_manager.list_all()
        assert any(m["metadata"].get("session_id") == "sess-123" for m in memories)

    def test_store_with_extra_metadata(self, memory_manager: MemoryManager):
        memory_manager.store("Meta test.", metadata={"source": "unit_test"})
        memories = memory_manager.list_all()
        assert any(m["metadata"].get("source") == "unit_test" for m in memories)

    def test_store_long_text_creates_multiple_chunks(self, memory_manager: MemoryManager):
        # Build text that is clearly longer than 500 chars with clear paragraph splits.
        para = "word " * 120  # ~600 chars per paragraph
        long_text = f"{para}\n\n{para}\n\n{para}"
        ids = memory_manager.store(long_text, auto_chunk=True)
        assert len(ids) > 1

    def test_store_no_chunk_keeps_single_entry(self, memory_manager: MemoryManager):
        para = "word " * 120
        long_text = f"{para}\n\n{para}"
        ids = memory_manager.store(long_text, auto_chunk=False)
        assert len(ids) == 1

    def test_near_duplicate_merges_instead_of_adding(self, memory_manager: MemoryManager):
        text = "The capital of France is Paris."
        memory_manager.store(text)
        count_before = memory_manager.count()
        # Store an identical text; the deduplication should update rather than add.
        memory_manager.store(text)
        # Count must not have grown (duplicate was merged).
        assert memory_manager.count() == count_before


class TestMemoryManagerRetrieve:
    def test_retrieve_empty_store_returns_empty_list(self, memory_manager: MemoryManager):
        result = memory_manager.retrieve("anything")
        assert result == []

    def test_retrieve_returns_list_of_dicts(self, memory_manager: MemoryManager):
        memory_manager.store("Python is a programming language.")
        results = memory_manager.retrieve("programming")
        assert isinstance(results, list)
        for r in results:
            assert "id" in r
            assert "content" in r
            assert "metadata" in r
            assert "similarity" in r

    def test_retrieve_n_results_limit(self, memory_manager: MemoryManager):
        for i in range(10):
            memory_manager.store(f"Unique fact number {i} about topic {i}.")
        results = memory_manager.retrieve("fact", n_results=3)
        assert len(results) <= 3

    def test_retrieve_similarity_in_valid_range(self, memory_manager: MemoryManager):
        memory_manager.store("The sky is blue.")
        results = memory_manager.retrieve("What color is the sky?")
        for r in results:
            assert -1.0 <= r["similarity"] <= 1.0

    def test_retrieve_min_importance_filter(self, memory_manager: MemoryManager):
        # Store something and then filter with a very high importance floor.
        memory_manager.store("hi")
        # Importance of "hi" will be low.  With min_importance=0.99 it should be filtered.
        results = memory_manager.retrieve("hi", min_importance=0.99)
        assert results == []


class TestMemoryManagerDelete:
    def test_delete_removes_memory(self, memory_manager: MemoryManager):
        ids = memory_manager.store("To be removed.")
        assert memory_manager.count() == 1
        memory_manager.delete(ids[0])
        assert memory_manager.count() == 0


class TestMemoryManagerListAll:
    def test_list_all_empty(self, memory_manager: MemoryManager):
        assert memory_manager.list_all() == []

    def test_list_all_returns_stored_memories(self, memory_manager: MemoryManager):
        memory_manager.store("Memory one.")
        memory_manager.store("Memory two.")
        all_memories = memory_manager.list_all()
        assert len(all_memories) == 2

    def test_list_all_respects_limit(self, memory_manager: MemoryManager):
        for i in range(10):
            memory_manager.store(f"Memory {i}.")
        assert len(memory_manager.list_all(limit=3)) == 3
