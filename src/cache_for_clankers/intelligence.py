"""
Intelligent logic layer: chunking, importance scoring, and deduplication.

These utilities sit on top of the raw vector store to provide:
  - Semantic chunking of long texts before storage
  - Importance scoring to rank content by information density
  - Deduplication helpers to decide whether new content is novel
"""

from __future__ import annotations

import re
import uuid

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Cosine-similarity threshold above which two documents are considered
#: duplicates.  At 0.92 minor paraphrases are collapsed; truly different
#: content is always stored separately.
SIMILARITY_THRESHOLD: float = 0.92

#: Maximum number of characters per chunk when splitting long texts.
DEFAULT_CHUNK_SIZE: int = 500


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(text: str, max_chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    """
    Split *text* into semantically coherent chunks of at most *max_chunk_size*
    characters.

    Strategy:
      1. Split on blank lines (paragraph boundaries).
      2. Accumulate paragraphs until the next one would exceed *max_chunk_size*.
      3. When a single paragraph is already longer than the limit, split it on
         sentence boundaries instead.

    Returns a non-empty list of non-empty strings.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else [""]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_size = 0

    for para in paragraphs:
        # If a single paragraph is too long, split it on sentence boundaries.
        if len(para) > max_chunk_size:
            # Flush any accumulated content first.
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts, current_size = [], 0
            sentences = _split_sentences(para)
            sentence_buf: list[str] = []
            sentence_size = 0
            for sent in sentences:
                if sentence_size + len(sent) > max_chunk_size and sentence_buf:
                    chunks.append(" ".join(sentence_buf))
                    sentence_buf, sentence_size = [sent], len(sent)
                else:
                    sentence_buf.append(sent)
                    sentence_size += len(sent)
            if sentence_buf:
                chunks.append(" ".join(sentence_buf))
            continue

        if current_size + len(para) > max_chunk_size and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts, current_size = [], 0

        current_parts.append(para)
        current_size += len(para)

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks or [text]


def _split_sentences(text: str) -> list[str]:
    """Naïve sentence splitter on '. ', '! ', '? ' boundaries."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------


def compute_importance(text: str) -> float:
    """
    Estimate the importance / information density of *text* as a float in
    [0.0, 1.0].

    Heuristics used (all normalised to [0, 1]):
      - Vocabulary richness: unique_tokens / total_tokens
      - Length contribution: capped at 100 words
      - Structure bonus: presence of list markers, code fences, colons
      - Fact bonus: presence of numeric data
    """
    words = text.lower().split()
    if not words:
        return 0.0

    unique_ratio = len(set(words)) / len(words)
    length_score = min(len(words) / 100.0, 1.0)

    structure_bonus = 0.1 if re.search(r"(^\s*[-*\d]\.?\s|```|:\s*$)", text, re.MULTILINE) else 0.0
    fact_bonus = 0.1 if re.search(r"\d+", text) else 0.0

    score = unique_ratio * 0.5 + length_score * 0.3 + structure_bonus + fact_bonus
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Deduplication helper
# ---------------------------------------------------------------------------


def deduplicate_content(
    candidate: str,
    existing_documents: list[str],
    existing_distances: list[float],
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[bool, int | None]:
    """
    Decide whether *candidate* is a near-duplicate of any document in
    *existing_documents*.

    *existing_distances* are ChromaDB cosine distances (range [0, 2]).
    Converts to cosine similarity:  similarity = 1 - distance

    Returns:
        (is_duplicate, index_of_most_similar)
        where *index_of_most_similar* is the index into *existing_documents*
        of the closest match, or ``None`` if the store is empty.
    """
    if not existing_distances:
        return False, None

    best_idx = int(min(range(len(existing_distances)), key=lambda i: existing_distances[i]))
    similarity = 1.0 - existing_distances[best_idx]
    return similarity >= threshold, best_idx


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


def generate_id() -> str:
    """Return a new unique memory ID."""
    return str(uuid.uuid4())
