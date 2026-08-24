"""VietAIDetector — Score Aggregator"""

from dataclasses import dataclass, field

from config.settings import YOUDEN_THRESHOLD
from schemas.models import ChunkDetail


@dataclass
class AggregationResult:
    """Result of the majority voting aggregation."""
    total_chunks: int
    ai_chunk_count: int
    ai_percentage: float
    applied_threshold: float
    final_decision: str
    chunk_details: list[ChunkDetail] = field(default_factory=list)


class ScoreAggregator:
    """Aggregates per-chunk Binoculars scores using Majority Voting."""

    def aggregate(
        self,
        chunks: list[ChunkDetail],
        threshold: float = YOUDEN_THRESHOLD, #default threshold is Youden's Index
    ) -> AggregationResult:
        """Aggregate chunk scores into a document-level detection result."""
        if not chunks:
            raise ValueError("Chunk list is empty.")

        ai_count = 0
        for chunk in chunks:
            if chunk.score < threshold:
                chunk.label = "AI"
                ai_count += 1
            else:
                chunk.label = "Human"

        total = len(chunks)
        ai_pct = round((ai_count / total) * 100.0, 2)

        # Majority Voting decision rules
        if ai_pct > 50.0:
            decision = "AI-generated"
        elif ai_pct > 0.0:
            decision = "Human-written but contains AI-generated parts"
        else:
            decision = "Human-written"

        return AggregationResult(
            total_chunks=total,
            ai_chunk_count=ai_count,
            ai_percentage=ai_pct,
            applied_threshold=threshold,
            final_decision=decision,
            chunk_details=chunks,
        )
