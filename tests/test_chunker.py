"""Unit tests for processing/chunker.py"""

import pytest
from unittest.mock import MagicMock
from processing.chunker import TextChunker


def _make_mock_chunker(
    token_ids: list[int],
    window: int = 450,
    overlap: int = 100,
    stride: int = 350,
    min_chunk: int = 50,
) -> TextChunker:
    """Create a TextChunker with a mock tokenizer — no transformers needed."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.encode.return_value = token_ids
    tokenizer.decode.side_effect = lambda ids, **kw: " ".join(str(t) for t in ids)
    return TextChunker(
        tokenizer, window=window, overlap=overlap,
        stride=stride, min_chunk=min_chunk,
    )


# Basic Chunking Tests

class TestBasicChunking:
    """Test fundamental chunking behavior."""

    def test_single_chunk_short_text(self):
        """Text with ≤ window tokens → single chunk."""
        ids = list(range(400))
        chunker = _make_mock_chunker(ids)
        out = chunker.chunk("dummy text", "doc1")
        assert out.total_chunks == 1
        assert out.chunks[0].chunk_index == 1
        assert out.chunks[0].token_count == 400

    def test_two_chunks_exact(self):
        """Text with 800 tokens, stride=350 → spans [0,450] and [350,800]."""
        ids = list(range(800))
        chunker = _make_mock_chunker(ids)
        out = chunker.chunk("dummy", "doc2")
        assert out.total_chunks == 2
        assert out.chunks[0].token_count == 450
        assert out.chunks[1].token_count == 450  # [350, 800] = 450 tokens

    def test_exactly_window_size(self):
        """Text with exactly window tokens → single chunk."""
        ids = list(range(450))
        chunker = _make_mock_chunker(ids)
        out = chunker.chunk("dummy", "doc_exact")
        assert out.total_chunks == 1
        assert out.chunks[0].token_count == 450


# Edge Case Tests

class TestEdgeCases:
    """Test edge cases in chunking algorithm."""

    def test_empty_text(self):
        """Empty text → no chunks."""
        ids = []
        chunker = _make_mock_chunker(ids)
        out = chunker.chunk("", "doc_empty")
        assert out.total_chunks == 0
        assert out.chunks == []

    def test_last_chunk_merge(self):
        """Last chunk < min_chunk (50) tokens → merged into previous chunk."""
        ids = list(range(480))
        # With stride=440: span1=[0,450], span2=[440,480]=40 < 50 → merge
        chunker = _make_mock_chunker(ids, window=450, stride=440, min_chunk=50)
        out = chunker.chunk("dummy", "doc_merge")
        assert out.total_chunks == 1
        assert out.chunks[0].token_count == 480

    def test_last_chunk_no_merge_if_long_enough(self):
        """Last chunk ≥ min_chunk → no merge."""
        ids = list(range(500))
        # stride=350: span1=[0,450], span2=[350,500]=150 ≥ 50 → no merge
        chunker = _make_mock_chunker(ids)
        out = chunker.chunk("dummy", "doc_no_merge")
        assert out.total_chunks == 2

    def test_very_short_text(self):
        """Text shorter than min_chunk → still returns single chunk."""
        ids = list(range(30))
        chunker = _make_mock_chunker(ids)
        out = chunker.chunk("short", "doc_short")
        assert out.total_chunks == 1
        assert out.chunks[0].token_count == 30


# Metadata Tests

class TestChunkMetadata:
    """Test chunk metadata correctness."""

    def test_chunk_indices_sequential(self):
        """Chunk indices should be 1-based and sequential."""
        ids = list(range(900))
        chunker = _make_mock_chunker(ids)
        out = chunker.chunk("dummy", "doc_idx")
        for i, c in enumerate(out.chunks, start=1):
            assert c.chunk_index == i

    def test_document_id_preserved(self):
        """Document ID should be preserved in output."""
        ids = list(range(100))
        chunker = _make_mock_chunker(ids)
        out = chunker.chunk("dummy", "my_doc_123")
        assert out.document_id == "my_doc_123"

    def test_total_tokens_correct(self):
        """Total tokens should match the tokenized length."""
        ids = list(range(750))
        chunker = _make_mock_chunker(ids)
        out = chunker.chunk("dummy", "doc_tokens")
        assert out.total_tokens == 750

    def test_initial_score_and_label(self):
        """Chunks should have default score=0.0 and label=''."""
        ids = list(range(200))
        chunker = _make_mock_chunker(ids)
        out = chunker.chunk("dummy", "doc_defaults")
        for c in out.chunks:
            assert c.score == 0.0
            assert c.label == ""
