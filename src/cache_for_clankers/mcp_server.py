"""
MCP (Model Context Protocol) server for cache-for-clankers.

Exposes the MemoryManager as a set of Claude tools so that Claude can
persist and retrieve semantic memories across sessions and compactions.

Run as a stdio server (Claude Desktop / claude.ai):
    python -m cache_for_clankers.mcp_server

Or via the installed entry-point:
    cache-for-clankers-mcp

Configuration (environment variables):
    CACHE_FOR_CLANKERS_DB_PATH     - path to the ChromaDB store (default: ~/.cache/cache-for-clankers)
    CACHE_FOR_CLANKERS_COLLECTION  - ChromaDB collection name (default: memories)
    CACHE_FOR_CLANKERS_MODEL       - sentence-transformers model (default: all-MiniLM-L6-v2)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .memory import MemoryManager

# ---------------------------------------------------------------------------
# Resolve configuration from environment (with sensible defaults)
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = str(Path.home() / ".cache" / "cache-for-clankers")

_DB_PATH = os.environ.get("CACHE_FOR_CLANKERS_DB_PATH", _DEFAULT_DB_PATH)
_COLLECTION = os.environ.get("CACHE_FOR_CLANKERS_COLLECTION", "memories")
_MODEL = os.environ.get("CACHE_FOR_CLANKERS_MODEL", "all-MiniLM-L6-v2")

# Lazy-initialised singleton so the embedding model is only loaded once.
_manager: MemoryManager | None = None


def _get_manager() -> MemoryManager:
    global _manager
    if _manager is None:
        _manager = MemoryManager(
            db_path=_DB_PATH,
            collection_name=_COLLECTION,
            embedding_model=_MODEL,
        )
    return _manager


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "cache-for-clankers",
    instructions=(
        "Long-term semantic memory for Claude. "
        "Use `store_memory` at the end of important exchanges to save context "
        "that should survive across sessions or compactions. "
        "Use `retrieve_memories` at the start of a session or when you need "
        "to recall relevant past context. "
        "Use `list_memories` to browse all stored entries. "
        "Use `delete_memory` to remove an entry that is no longer relevant. "
        "Use `count_memories` to see how many memories are stored."
    ),
)


@mcp.tool()
def store_memory(
    content: str,
    session_id: str = "unknown",
) -> str:
    """
    Store an important piece of context for later retrieval.

    Long texts are automatically split into chunks; near-duplicate content
    is merged with the existing entry rather than stored twice.

    Args:
        content:    The text to remember (e.g. a key fact, user preference,
                    decision, or summary).
        session_id: Optional identifier for the current Claude session.
                    Useful for grouping related memories.

    Returns:
        A confirmation message with the IDs of the stored memory chunks.
    """
    ids = _get_manager().store(content, session_id=session_id)
    plural = "chunk" if len(ids) == 1 else "chunks"
    return f"Stored {len(ids)} memory {plural}. IDs: {', '.join(ids)}"


@mcp.tool()
def retrieve_memories(
    query: str,
    n_results: int = 5,
    min_importance: float = 0.0,
) -> str:
    """
    Retrieve the most relevant memories for a natural-language query.

    Results are re-ranked by a blend of semantic similarity (70%) and
    information-density importance score (30%).

    Args:
        query:          Natural-language question or topic to search for.
        n_results:      Maximum number of memories to return (default 5).
        min_importance: Only return memories with an importance score at or
                        above this value (0.0 – 1.0, default 0.0).

    Returns:
        JSON array of matching memories, each with fields:
        id, content, similarity, session_id, timestamp.
    """
    results = _get_manager().retrieve(
        query, n_results=n_results, min_importance=min_importance
    )
    if not results:
        return "No memories found."

    simplified = [
        {
            "id": r["id"],
            "content": r["content"],
            "similarity": round(r["similarity"], 4),
            "session_id": r["metadata"].get("session_id", "unknown"),
            "importance": round(float(r["metadata"].get("importance", 0)), 4),
            "timestamp": r["metadata"].get("timestamp"),
        }
        for r in results
    ]
    return json.dumps(simplified, indent=2)


@mcp.tool()
def list_memories(limit: int = 50) -> str:
    """
    List stored memories (no ranking applied).

    Args:
        limit: Maximum number of entries to return (default 50).

    Returns:
        JSON array of memory entries with id, content, and metadata.
    """
    memories = _get_manager().list_all(limit=limit)
    if not memories:
        return "No memories stored."
    return json.dumps(memories, indent=2)


@mcp.tool()
def delete_memory(memory_id: str) -> str:
    """
    Delete a stored memory by its ID.

    Args:
        memory_id: The ID of the memory to delete (as returned by
                   store_memory or list_memories).

    Returns:
        A confirmation message.
    """
    _get_manager().delete(memory_id)
    return f"Deleted memory {memory_id}."


@mcp.tool()
def count_memories() -> str:
    """
    Return the total number of memories currently stored.

    Returns:
        A short message with the count.
    """
    n = _get_manager().count()
    return f"{n} {'memory' if n == 1 else 'memories'} stored."


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the MCP server over stdio (used by Claude Desktop)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
