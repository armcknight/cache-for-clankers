"""Tests for the CLI entry point."""

from __future__ import annotations

import json

import chromadb
import pytest

from cache_for_clankers.cli import main
from cache_for_clankers.memory import MemoryManager
from cache_for_clankers.store import VectorStore
from conftest import FakeEmbeddingFunction


@pytest.fixture()
def patched_manager(ephemeral_store: VectorStore, monkeypatch) -> MemoryManager:
    """
    Patch MemoryManager.__init__ so the CLI uses our ephemeral in-memory
    store instead of touching the filesystem.
    """
    manager = MemoryManager(_store=ephemeral_store)

    def _fake_init(self, **kwargs):  # noqa: ARG001
        self._store = ephemeral_store
        self.similarity_threshold = 0.92

    monkeypatch.setattr(MemoryManager, "__init__", _fake_init)
    return manager


class TestCLI:
    def test_count_empty(self, patched_manager, capsys):
        rc = main(["count"])
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert out == "0"

    def test_store_and_count(self, patched_manager, capsys):
        main(["store", "Hello from the CLI test."])
        capsys.readouterr()  # flush

        rc = main(["count"])
        assert rc == 0
        out = capsys.readouterr().out.strip()
        assert int(out) >= 1

    def test_store_missing_text_returns_error(self, patched_manager, capsys, monkeypatch):
        monkeypatch.setattr("sys.stdin", __import__("io").StringIO(""))
        rc = main(["store"])
        assert rc == 1

    def test_retrieve_empty_store(self, patched_manager, capsys):
        rc = main(["retrieve", "anything"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No memories found" in out

    def test_store_and_retrieve(self, patched_manager, capsys):
        main(["store", "The user's favourite colour is green."])
        capsys.readouterr()

        rc = main(["retrieve", "favourite colour"])
        assert rc == 0
        out = capsys.readouterr().out
        assert len(out) > 0

    def test_list_empty(self, patched_manager, capsys):
        rc = main(["list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "No memories stored" in out

    def test_store_and_list(self, patched_manager, capsys):
        main(["store", "Something to list."])
        capsys.readouterr()
        rc = main(["list"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Something to list." in out

    def test_store_and_delete(self, patched_manager, capsys):
        main(["store", "To be deleted via CLI."])
        capsys.readouterr()
        main(["count"])
        id_out = capsys.readouterr().out.strip()

        # Grab the actual ID from list output.
        main(["list", "--json"])
        list_out = capsys.readouterr().out
        memories = json.loads(list_out)
        assert memories, "Expected at least one memory"
        mem_id = memories[0]["id"]

        rc = main(["delete", mem_id])
        assert rc == 0
        capsys.readouterr()

        main(["count"])
        assert capsys.readouterr().out.strip() == "0"

    def test_retrieve_json_output(self, patched_manager, capsys):
        main(["store", "Paris is the capital of France."])
        capsys.readouterr()

        main(["retrieve", "--json", "capital"])
        out = capsys.readouterr().out
        results = json.loads(out)
        assert isinstance(results, list)
        assert all("content" in r for r in results)
