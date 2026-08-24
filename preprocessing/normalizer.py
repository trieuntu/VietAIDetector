"""VietAIDetector — Text Normalizer"""

from schemas.models import PreprocessResult
from preprocessing.document_reader import DocumentReader, UnsupportedFormatError
from preprocessing.text_utils import (
    DEHYPHEN_RE,
    LINEBREAK_RE,
    WHITESPACE_RE,
    count_words,
    special_char_ratio,
)


class TextNormalizer:
    """Normalizes Vietnamese text extracted from documents."""

    def __init__(self):
        self.reader = DocumentReader()
        self._ocr_engine = None

    def _get_ocr_engine(self):
        """Lazy-load the VinternOCR engine (only when a scanned PDF is detected)."""
        if self._ocr_engine is None:
            from preprocessing.ocr_engine import VinternOCR
            self._ocr_engine = VinternOCR()
        return self._ocr_engine

    def normalize(self, text: str) -> str:
        """Apply text normalization rules to a single text string."""
        # Step 1: De-hyphenation
        text = DEHYPHEN_RE.sub(r'\1\2', text)
        # Step 2: Remove mid-sentence line breaks
        text = LINEBREAK_RE.sub(' ', text)
        # Step 3: Collapse whitespace
        text = WHITESPACE_RE.sub(' ', text)
        return text.strip()

    def filter_paragraphs(self, text: str) -> list[str]:
        """Filter out non-semantic paragraphs from the text."""
        raw_paragraphs = text.split('\n')
        valid: list[str] = []

        for paragraph in raw_paragraphs:
            # Normalize whitespace within this paragraph only
            paragraph = WHITESPACE_RE.sub(' ', paragraph).strip()
            if not paragraph:
                continue
            if count_words(paragraph) < 5:
                continue
            if special_char_ratio(paragraph) > 0.50:
                continue
            valid.append(paragraph)

        return valid

    def preprocess(self, filename: str, content: bytes | str) -> PreprocessResult:
        """Execute the full preprocessing pipeline."""
        try:
            raw_text, source_format = self.reader.read(
                filename, content, ocr_engine=self._get_ocr_engine()
            )
        except UnsupportedFormatError as exc:
            return PreprocessResult(
                document_name=filename,
                source_format="unsupported",
                extraction_status="error",
                cleaned_text="",
                error_message=str(exc),
            )
        except Exception as exc:
            return PreprocessResult(
                document_name=filename,
                source_format="unknown",
                extraction_status="error",
                cleaned_text="",
                error_message=f"Text extraction error: {exc}",
            )

        # Step 1: De-hyphenation on the whole text (fixes word breaks across lines)
        dehyphened = DEHYPHEN_RE.sub(r'\1\2', raw_text)

        # Step 2: Split into paragraphs and filter (before merging lines)
        paragraphs = self.filter_paragraphs(dehyphened)

        # Step 3: Normalize each surviving paragraph individually
        paragraphs = [self.normalize(p) for p in paragraphs]

        cleaned = "\n".join(paragraphs)

        return PreprocessResult(
            document_name=filename,
            source_format=source_format,
            extraction_status="success",
            cleaned_text=cleaned,
        )
