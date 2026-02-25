"""
cache-for-clankers: A locally hosted vector database memory layer for LLMs.

Provides intelligent storage and retrieval of semantic context across
Claude sessions and compactions.
"""

from .memory import MemoryManager
from .store import VectorStore
from .intelligence import chunk_text, compute_importance, deduplicate_content

__all__ = [
    "MemoryManager",
    "VectorStore",
    "chunk_text",
    "compute_importance",
    "deduplicate_content",
]
