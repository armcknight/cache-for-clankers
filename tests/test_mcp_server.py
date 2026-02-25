"""Tests for the MCP server tools."""

from __future__ import annotations

import json

import pytest

import cache_for_clankers.mcp_server as mcp_module
from cache_for_clankers.memory import MemoryManager
from conftest import FakeEmbeddingFunction, _EPHEMERAL_CLIENT

import uuid


@pytest.fixture(autouse=True)
def _isolated_manager(monkeypatch):
    """
    Replace the module-level _manager singleton with a fresh in-memory
    MemoryManager for each test so tests don't share state.
    """
    from cache_for_clankers.store import VectorStore

    collection_name = f"mcp_test_{uuid.uuid4().hex}"
    store = VectorStore(
        _client=_EPHEMERAL_CLIENT,
        collection_name=collection_name,
        _embedding_function=FakeEmbeddingFunction(),
    )
    manager = MemoryManager(_store=store)
    monkeypatch.setattr(mcp_module, "_manager", manager)
    return manager


class TestMCPTools:
    def test_count_memories_empty(self):
        result = mcp_module.count_memories()
        assert "0" in result
        assert "memories" in result or "memory" in result

    def test_store_memory_returns_confirmation(self):
        result = mcp_module.store_memory("Alice likes Python.")
        assert "Stored" in result
        assert "chunk" in result.lower()

    def test_store_memory_increments_count(self):
        mcp_module.store_memory("Bob prefers Rust.")
        result = mcp_module.count_memories()
        assert "1" in result

    def test_store_memory_with_session_id(self, _isolated_manager):
        mcp_module.store_memory("Memory with session.", session_id="sess-abc")
        memories = _isolated_manager.list_all()
        assert any(m["metadata"].get("session_id") == "sess-abc" for m in memories)

    def test_retrieve_memories_empty(self):
        result = mcp_module.retrieve_memories("anything")
        assert "No memories found" in result

    def test_retrieve_memories_returns_json(self):
        mcp_module.store_memory("The sky is blue.")
        result = mcp_module.retrieve_memories("sky color")
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) >= 1
        assert "content" in data[0]
        assert "similarity" in data[0]

    def test_retrieve_memories_n_results_limit(self):
        for i in range(6):
            mcp_module.store_memory(f"Distinct fact number {i} about subject {i}.")
        result = mcp_module.retrieve_memories("fact", n_results=3)
        data = json.loads(result)
        assert len(data) <= 3

    def test_list_memories_empty(self):
        result = mcp_module.list_memories()
        assert "No memories stored" in result

    def test_list_memories_returns_stored(self):
        mcp_module.store_memory("Something to list.")
        result = mcp_module.list_memories()
        data = json.loads(result)
        assert isinstance(data, list)
        assert any("Something to list." in m.get("content", "") for m in data)

    def test_list_memories_limit(self):
        for i in range(5):
            mcp_module.store_memory(f"Entry {i}.")
        result = mcp_module.list_memories(limit=2)
        data = json.loads(result)
        assert len(data) <= 2

    def test_delete_memory(self):
        mcp_module.store_memory("To be deleted.")
        # Get the ID via list
        list_result = json.loads(mcp_module.list_memories())
        mem_id = list_result[0]["id"]

        del_result = mcp_module.delete_memory(mem_id)
        assert "Deleted" in del_result
        assert mem_id in del_result

        assert "0" in mcp_module.count_memories()

    def test_count_after_multiple_stores(self):
        mcp_module.store_memory("Fact one.")
        mcp_module.store_memory("Fact two.")
        result = mcp_module.count_memories()
        assert "2" in result

    def test_retrieve_includes_session_id_field(self):
        mcp_module.store_memory("Test session field.", session_id="my-session")
        result = mcp_module.retrieve_memories("session field")
        data = json.loads(result)
        assert data[0]["session_id"] == "my-session"
