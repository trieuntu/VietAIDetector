"""VietAIDetector — Text Chunker"""

from dataclasses import dataclass, field

from config.settings import (
    CHUNK_WINDOW,
    CHUNK_OVERLAP,
    CHUNK_STRIDE,
    CHUNK_MIN_TOKENS,
)
from schemas.models import ChunkDetail


@dataclass
class ChunkOutput:
    """Container for the chunking result of a single document."""
    document_id: str
    total_tokens: int
    total_chunks: int
    chunks: list[ChunkDetail] = field(default_factory=list)


class TextChunker:
    """Sliding window chunker using PhoGPT tokenizer."""

    def __init__(
        self,
        tokenizer_name_or_obj,
        window: int = CHUNK_WINDOW,
        overlap: int = CHUNK_OVERLAP,
        stride: int = CHUNK_STRIDE,
        min_chunk: int = CHUNK_MIN_TOKENS,
    ):
        """Initialize the chunker with a tokenizer and sliding window parameters."""
        if isinstance(tokenizer_name_or_obj, str):
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name_or_obj, trust_remote_code=True
            )
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            # Accept a pre-built tokenizer object (useful for testing/reuse)
            self.tokenizer = tokenizer_name_or_obj

        self.window = window
        self.overlap = overlap
        self.stride = stride
        self.min_chunk = min_chunk

    def chunk(
        self,
        text: str,
        document_id: str,
        window: int = None,
        overlap: int = None,
    ) -> ChunkOutput:
        """Split text into overlapping chunks using the sliding window algorithm."""
        active_window = window if window is not None else self.window
        if window is not None or overlap is not None:
            active_overlap = overlap if overlap is not None else self.overlap
            active_stride = active_window - active_overlap
        else:
            active_stride = self.stride
            
        if active_stride <= 0:
            active_stride = 1  # Prevent infinite loop if overlap >= window
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        N = len(ids)

        if N == 0:
            return ChunkOutput(
                document_id=document_id, total_tokens=0, total_chunks=0
            )

        # Build list of (start, end) token index pairs
        spans: list[tuple[int, int]] = []

        if N <= active_window:
            # Text fits in a single window — no splitting needed
            spans.append((0, N))
        else:
            # Sliding window with active_stride
            start = 0
            while start < N:
                end = min(start + active_window, N)
                spans.append((start, end))
                if end == N:
                    break
                start += active_stride

            # Edge case: merge last chunk if it's too short
            if len(spans) > 1:
                last_start, last_end = spans[-1]
                if (last_end - last_start) < self.min_chunk:
                    spans.pop()
                    prev_start, _ = spans[-1]
                    spans[-1] = (prev_start, N)

        # Decode token spans back to Vietnamese text
        chunks: list[ChunkDetail] = []
        for idx, (s, e) in enumerate(spans, start=1):
            token_slice = ids[s:e]
            decoded = self.tokenizer.decode(token_slice, skip_special_tokens=True)
            chunks.append(
                ChunkDetail(
                    chunk_index=idx,
                    text=decoded,
                    token_count=e - s,
                    score=0.0,
                    label="",
                )
            )

        return ChunkOutput(
            document_id=document_id,
            total_tokens=N,
            total_chunks=len(chunks),
            chunks=chunks,
        )
