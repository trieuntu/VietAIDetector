"""VietAIDetector — Data Models (Schemas)"""

from dataclasses import dataclass, field
from config.settings import CHUNK_WINDOW, CHUNK_OVERLAP


@dataclass
class ChunkDetail:
    """Represents a single text chunk with its detection score and label."""
    chunk_index: int
    text: str
    token_count: int
    score: float = 0.0
    label: str = ""


@dataclass
class PreprocessResult:
    """Result of document preprocessing (extraction + normalization)."""
    document_name: str
    source_format: str
    extraction_status: str
    cleaned_text: str
    error_message: str = ""


@dataclass
class DetectionResult:
    """Final detection result for the entire document."""
    document_id: str
    document_name: str
    total_chunks: int
    ai_chunk_count: int
    ai_percentage: float
    applied_threshold: float
    applied_mode: str
    final_decision: str
    chunk_details: list[ChunkDetail] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    chunk_window: int = CHUNK_WINDOW
    chunk_overlap: int = CHUNK_OVERLAP
