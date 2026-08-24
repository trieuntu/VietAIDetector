"""VietAIDetector — PDF Report Generator"""

import os
import urllib.request
from datetime import datetime, timezone, timedelta
from config.settings import FONT_PATH, FONT_URL, APP_NAME
from schemas.models import DetectionResult


def _is_valid_ttf(path: str) -> bool:
    """Check if a file is a valid TrueType/OpenType font by reading magic bytes."""
    try:
        with open(path, "rb") as f:
            header = f.read(4)
        return header in (b"\x00\x01\x00\x00", b"OTTO", b"true")
    except (OSError, IOError):
        return False


def _ensure_font(font_path: str = FONT_PATH) -> str:
    """Download NotoSans font if not present or corrupted at the expected path."""
    if os.path.exists(font_path) and _is_valid_ttf(font_path):
        return font_path

    # Remove corrupted file if present
    if os.path.exists(font_path):
        os.remove(font_path)

    os.makedirs(os.path.dirname(font_path) or "/tmp", exist_ok=True)

    # Try multiple font sources in order of reliability
    urls = [
        FONT_URL,
        (
            "https://github.com/notofonts/noto-fonts/raw/main/hinted/ttf/"
            "NotoSans/NotoSans-Regular.ttf"
        ),
        (
            "https://raw.githubusercontent.com/notofonts/noto-fonts/main/"
            "hinted/ttf/NotoSans/NotoSans-Regular.ttf"
        ),
        (
            "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/"
            "NotoSans/NotoSans-Regular.ttf"
        ),
    ]

    for url in urls:
        try:
            urllib.request.urlretrieve(url, font_path)
            if _is_valid_ttf(font_path):
                return font_path
            # Downloaded file is not a valid font, try next URL
            os.remove(font_path)
        except Exception:
            if os.path.exists(font_path):
                os.remove(font_path)
            continue

    raise RuntimeError(
        "Could not download NotoSans font from any source. "
        "Please manually download NotoSans-Regular.ttf and set FONT_PATH."
    )


class PDFReportGenerator:
    """Generate PDF reports with highlighted AI/Human chunks using fpdf2."""

    # Color definitions (RGB)
    AI_COLOR = (255, 200, 200)       # Light red for AI chunks
    HUMAN_COLOR = (210, 245, 210)    # Light green for Human chunks

    def __init__(self, font_path: str = FONT_PATH):
        """Initialize the report generator."""
        self.font_path = _ensure_font(font_path)

    def _make_pdf(self, result: DetectionResult):
        """Build the PDF document from detection results."""
        from fpdf import FPDF, XPos, YPos

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Register Vietnamese-capable font
        pdf.add_font("NotoSans", "", self.font_path)
        pdf.add_font("NotoSans", "B", self.font_path)

        # Header
        pdf.set_font("NotoSans", "B", 16)
        pdf.cell(
            0, 12,
            f"AI Text Detection Report — {APP_NAME}",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C",
        )

        pdf.set_font("NotoSans", "", 10)
        gmt7 = timezone(timedelta(hours=7))
        pdf.cell(
            0, 7,
            f"Generated: {datetime.now(gmt7).strftime('%Y-%m-%d %H:%M')} (GMT+7 HCM)",
            new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C",
        )
        pdf.ln(6)

        # Summary
        pdf.set_font("NotoSans", "B", 12)
        pdf.cell(0, 9, "Detection Summary", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(150, 150, 150)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

        pdf.set_font("NotoSans", "", 11)
        rows = [
            ("Document:", result.document_name),
            ("AI Ratio:", f"{result.ai_percentage}%"),
            ("Decision:", result.final_decision),
            ("Threshold Mode:", result.applied_mode),
            ("Threshold Value:", str(result.applied_threshold)),
            ("Chunk Window Size:", str(result.chunk_window)),
            ("Chunk Overlap:", str(result.chunk_overlap)),
            ("Total Chunks:", str(result.total_chunks)),
            ("AI Chunks:", str(result.ai_chunk_count)),
            ("Human Chunks:", str(result.total_chunks - result.ai_chunk_count)),
            ("Processing Time:", f"{result.processing_time_seconds:.2f}s"),
        ]
        for label, value in rows:
            pdf.set_font("NotoSans", "B", 11)
            pdf.cell(55, 8, label)
            pdf.set_font("NotoSans", "", 11)
            pdf.cell(0, 8, value, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.ln(8)

        # Chunk Details
        pdf.set_font("NotoSans", "B", 12)
        pdf.cell(0, 9, "Chunk-Level Details", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

        for chunk in result.chunk_details:
            color = self.AI_COLOR if chunk.label == "AI" else self.HUMAN_COLOR
            pdf.set_fill_color(*color)

            # Chunk header bar
            pdf.set_font("NotoSans", "B", 10)
            header_text = (
                f"Chunk {chunk.chunk_index}  |  "
                f"Score: {chunk.score:.4f}  |  "
                f"{chunk.label}  |  "
                f"{chunk.token_count} tokens"
            )
            pdf.cell(0, 8, header_text, fill=True,
                     new_x=XPos.LMARGIN, new_y=YPos.NEXT)

            # Full chunk text content (no truncation)
            pdf.set_font("NotoSans", "", 9)
            pdf.multi_cell(0, 6, chunk.text, fill=True)
            pdf.ln(3)

        return pdf

    def generate_pdf(self, result: DetectionResult) -> bytes:
        """Generate a complete PDF report as bytes."""
        pdf = self._make_pdf(result)
        return bytes(pdf.output())
