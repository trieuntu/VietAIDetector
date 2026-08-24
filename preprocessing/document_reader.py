"""VietAIDetector — Document Reader"""

from io import BytesIO
from typing import Optional


class UnsupportedFormatError(Exception):
    """Raised when the input file format cannot be processed."""
    pass


class DocumentReader:
    """Extracts raw text from various document formats."""

    @staticmethod
    def detect_format(filename: str) -> str:
        """Classify the document format based on file extension."""
        lower = filename.lower()
        if lower.endswith(".docx"):
            return "docx"
        if lower.endswith(".pdf"):
            return "pdf"
        return "plain"

    @staticmethod
    def extract_from_docx(content: bytes) -> str:
        """Extract text from a DOCX file."""
        from docx import Document

        doc = Document(BytesIO(content))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    @staticmethod
    def extract_from_pdf(content: bytes) -> tuple[str, str]:
        """Extract text from a native PDF file."""
        import fitz  # PyMuPDF

        all_text_parts: list[str] = []

        with fitz.open(stream=content, filetype="pdf") as pdf:
            for page in pdf:
                page_height = page.rect.height
                header_threshold = page_height * 0.10
                footer_threshold = page_height * 0.90

                blocks = page.get_text("blocks")
                for block in blocks:
                    # block format: (x0, y0, x1, y1, text, block_no, block_type)
                    if len(block) < 5:
                        continue
                    y0, y1, text = block[1], block[3], block[4]
                    if not text.strip():
                        continue
                    # Skip header and footer regions
                    if y1 < header_threshold or y0 > footer_threshold:
                        continue
                    all_text_parts.append(text.strip())

        combined = "\n".join(all_text_parts)

        if not combined.strip():
            # No text layer found — this is a scanned/image-based PDF
            return "", "pdf_scanned"

        return combined, "pdf_native"

    def read(
        self, filename: str, content: bytes | str, ocr_engine=None
    ) -> tuple[str, str]:
        """Read a document and return its raw text content."""
        if isinstance(content, str):
            return content, "plain"

        fmt = self.detect_format(filename)

        if fmt == "docx":
            return self.extract_from_docx(content), "docx"
        elif fmt == "pdf":
            text, pdf_type = self.extract_from_pdf(content)
            if pdf_type == "pdf_scanned":
                if ocr_engine is None:
                    raise UnsupportedFormatError(
                        "This PDF is a scanned (image-based) PDF. "
                        "OCR engine is not available."
                    )
                ocr_text = ocr_engine.extract_from_pdf(content)
                return ocr_text, "pdf_scanned"
            return text, pdf_type
        else:
            return content.decode("utf-8", errors="replace"), "plain"
