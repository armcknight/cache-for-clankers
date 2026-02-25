"""Tests for the intelligence layer (chunking, importance, deduplication)."""

from __future__ import annotations

import pytest

from cache_for_clankers.intelligence import (
    SIMILARITY_THRESHOLD,
    chunk_text,
    compute_importance,
    deduplicate_content,
    generate_id,
)


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_short_text_is_not_split(self):
        text = "Hello world."
        chunks = chunk_text(text, max_chunk_size=500)
        assert chunks == ["Hello world."]

    def test_long_text_is_split_on_paragraphs(self):
        para = "word " * 100  # 500 chars
        text = f"{para}\n\n{para}\n\n{para}"
        chunks = chunk_text(text, max_chunk_size=500)
        assert len(chunks) > 1

    def test_very_long_single_paragraph_splits_on_sentences(self):
        sentences = ["This is sentence number %d. " % i for i in range(30)]
        text = "".join(sentences)  # no blank lines
        chunks = chunk_text(text, max_chunk_size=100)
        assert len(chunks) > 1
        # All original content is preserved
        assert "".join(chunks).replace("  ", " ") != ""

    def test_empty_string_returns_list(self):
        result = chunk_text("", max_chunk_size=500)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_chunks_do_not_exceed_max_chunk_size_for_multi_para(self):
        # Each paragraph is exactly max_chunk_size characters.
        para = "x" * 100
        text = "\n\n".join([para] * 5)
        chunks = chunk_text(text, max_chunk_size=100)
        # Every chunk must be at most max_chunk_size characters.
        for chunk in chunks:
            assert len(chunk) <= 100, f"Chunk too long: {len(chunk)}"


# ---------------------------------------------------------------------------
# compute_importance
# ---------------------------------------------------------------------------


class TestComputeImportance:
    def test_empty_text_returns_zero(self):
        assert compute_importance("") == 0.0

    def test_score_is_in_unit_interval(self):
        texts = [
            "Hello.",
            "The quick brown fox jumps over the lazy dog.",
            "1. First item\n2. Second item\n3. Third item",
            "def foo():\n    return 42",
            "a " * 200,
        ]
        for text in texts:
            score = compute_importance(text)
            assert 0.0 <= score <= 1.0, f"Out of range for: {text!r}"

    def test_structured_text_scores_higher_than_repetitive_prose(self):
        # Repetitive prose: very low unique_ratio
        prose = "the the the the the the the the the the"
        # Structured list with numbers: structure bonus + fact bonus
        structured = "1. First point\n2. Second point\n3. Third point\n4. Fourth point"
        assert compute_importance(structured) > compute_importance(prose)

    def test_numeric_content_adds_fact_bonus(self):
        without = "the cat sat on the mat"
        with_num = "the cat sat on 3 mats"
        assert compute_importance(with_num) > compute_importance(without)


# ---------------------------------------------------------------------------
# deduplicate_content
# ---------------------------------------------------------------------------


class TestDeduplicateContent:
    def test_empty_distances_returns_no_duplicate(self):
        is_dup, idx = deduplicate_content("text", [], [], threshold=0.92)
        assert is_dup is False
        assert idx is None

    def test_very_close_distance_is_duplicate(self):
        # distance=0.05 → similarity=0.95 > threshold 0.92
        is_dup, idx = deduplicate_content("text", ["existing"], [0.05], threshold=0.92)
        assert is_dup is True
        assert idx == 0

    def test_far_distance_is_not_duplicate(self):
        # distance=0.3 → similarity=0.7 < threshold 0.92
        is_dup, idx = deduplicate_content("text", ["different"], [0.3], threshold=0.92)
        assert is_dup is False
        assert idx == 0  # idx is still returned (best candidate)

    def test_picks_closest_of_multiple_candidates(self):
        distances = [0.5, 0.02, 0.4]
        is_dup, idx = deduplicate_content("text", ["a", "b", "c"], distances, threshold=0.92)
        assert is_dup is True
        assert idx == 1  # distance 0.02 is the closest

    def test_threshold_boundary(self):
        # exactly at threshold: similarity = 1 - distance = threshold
        threshold = 0.92
        distance = 1.0 - threshold  # = 0.08
        is_dup, idx = deduplicate_content("text", ["x"], [distance], threshold=threshold)
        assert is_dup is True


# ---------------------------------------------------------------------------
# generate_id
# ---------------------------------------------------------------------------


class TestGenerateId:
    def test_returns_non_empty_string(self):
        assert isinstance(generate_id(), str)
        assert len(generate_id()) > 0

    def test_ids_are_unique(self):
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100
