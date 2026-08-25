"""
VietAIDetector — Gradio Frontend
"""

import html
import os
import time
import uuid
import tempfile
from typing import Optional

import gradio as gr
import pandas as pd

from config.settings import (
    APP_NAME,
    APP_DESCRIPTION,
    APP_VERSION,
    OBSERVER_MODEL,
    PERFORMER_MODEL,
    THRESHOLD_MODES,
    DEFAULT_MODE,
    MINIMUM_TOKENS,
    MAX_MODEL_TOKENS,
    CHUNK_WINDOW,
    CHUNK_OVERLAP,
    YOUDEN_THRESHOLD,
    CLOSEST_POINT_THRESHOLD,
    FPR_THRESHOLD,
)
from schemas.models import ChunkDetail, DetectionResult
from preprocessing.normalizer import TextNormalizer
from processing.chunker import TextChunker
from processing.aggregator import ScoreAggregator
from reporting.pdf_report import PDFReportGenerator


# SVG Icons (inline, professional, no emoji dependency)

# 24x24 SVG icons used across the UI
SVG_SEARCH = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>'
SVG_FILE_TEXT = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><line x1="10" y1="9" x2="8" y2="9"/></svg>'
SVG_UPLOAD = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>'
SVG_SETTINGS = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>'
SVG_PLAY = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>'
SVG_TRASH = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>'
SVG_BAR_CHART = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/></svg>'
SVG_DOWNLOAD = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>'
SVG_INFO = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>'
SVG_BOOK = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>'
SVG_ROBOT = '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#e74c3c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg>'
SVG_PEN = '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#27ae60" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19l7-7 3 3-7 7-3 0z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18z"/><path d="M2 2l7.586 7.586"/><circle cx="11" cy="11" r="2"/></svg>'
SVG_ALERT = '<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#f39c12" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
SVG_WARNING = '<svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#c0392b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'


# Module-level state (initialized lazily to avoid loading models at import)

_scorer = None
_chunker = None
_normalizer = TextNormalizer()
_aggregator = ScoreAggregator()
_report_gen = PDFReportGenerator()


def _get_scorer():
    """Lazy-load the VietBinocularsScorer (heavy GPU model loading)."""
    global _scorer
    if _scorer is None:
        from core.scorer import VietBinocularsScorer
        _scorer = VietBinocularsScorer(
            observer_name=OBSERVER_MODEL,
            performer_name=PERFORMER_MODEL,
        )
    return _scorer


def _get_chunker():
    """Lazy-load the TextChunker using the scorer's tokenizer."""
    global _chunker
    if _chunker is None:
        scorer = _get_scorer()
        _chunker = TextChunker(tokenizer_name_or_obj=scorer.tokenizer)
    return _chunker


# Backend Logic

def _run_detection(
    text: str,
    file_obj,
    mode: str,
    window: int,
    overlap: int,
    progress=gr.Progress(),
) -> tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """Execute the full detection pipeline.

    Args:
        text: Raw text input from the textbox.
        file_obj: Uploaded file object (gradio UploadFile) or None.
        mode: Detection mode name (key in THRESHOLD_MODES).
        progress: Gradio progress tracker for loading animation.

    Returns:
        Tuple of (summary_html, chunks_dataframe, pdf_file_path).
    """
    start_time = time.time()
    document_name = "Direct Text Input"

    try:
        # ── Step 1: Get input text ──────────────────────────────
        progress(0.05, desc="Reading input data...")

        if file_obj is not None:
            # File upload takes priority
            filepath = file_obj if isinstance(file_obj, str) else file_obj.name
            document_name = os.path.basename(filepath)
            with open(filepath, "rb") as f:
                file_bytes = f.read()
            preprocess_result = _normalizer.preprocess(document_name, file_bytes)
            source_format = preprocess_result.source_format
            if source_format == "pdf_scanned":
                progress(0.10, desc="Running OCR on scanned PDF (Vintern-1B)...")
                gr.Warning(
                    "Scanned PDF detected — text was extracted via OCR (Vintern-1B-v2). "
                    "OCR may introduce hallucinated text, which could affect detection accuracy. "
                    "For the most reliable results, use a text-based PDF or paste text directly."
                )
        elif text and text.strip():
            preprocess_result = _normalizer.preprocess("input.txt", text)
            source_format = preprocess_result.source_format
        else:
            gr.Warning("Please enter text or upload a file.")
            return "", None, None

        # Check preprocessing result
        if preprocess_result.extraction_status == "error":
            raise ValueError(
                f"File processing error: {preprocess_result.error_message}"
            )

        cleaned_text = preprocess_result.cleaned_text
        if not cleaned_text.strip():
            gr.Warning("Could not extract text from the uploaded file.")
            return "", None, None

        # ── Step 2: Chunk text ──────────────────────────────────
        progress(0.15, desc="Chunking text...")

        scorer = _get_scorer()
        chunker = _get_chunker()

        # Check minimum token count
        token_ids = scorer.tokenizer.encode(cleaned_text, add_special_tokens=False)
        if len(token_ids) < MINIMUM_TOKENS:
            gr.Warning(
                f"Text too short ({len(token_ids)} tokens). "
                f"Minimum {MINIMUM_TOKENS} tokens required for accurate detection."
            )
            return "", None, None

        doc_id = str(uuid.uuid4())[:8]
        chunk_output = chunker.chunk(
            cleaned_text, 
            document_id=doc_id, 
            window=int(window), 
            overlap=int(overlap)
        )

        if chunk_output.total_chunks == 0:
            gr.Warning("No segments remained after text processing.")
            return "", None, None

        # ── Step 3: Score chunks ────────────────────────────────
        progress(0.30, desc=f"Analyzing {chunk_output.total_chunks} text segments...")

        scored_chunks = scorer.score_chunks(chunk_output.chunks)

        # ── Step 4: Aggregate scores ────────────────────────────
        progress(0.85, desc="Aggregating results...")

        threshold = THRESHOLD_MODES.get(mode, THRESHOLD_MODES[DEFAULT_MODE])
        agg_result = _aggregator.aggregate(scored_chunks, threshold)

        elapsed = round(time.time() - start_time, 2)

        # Build DetectionResult
        detection_result = DetectionResult(
            document_id=doc_id,
            document_name=document_name,
            total_chunks=agg_result.total_chunks,
            ai_chunk_count=agg_result.ai_chunk_count,
            ai_percentage=agg_result.ai_percentage,
            applied_threshold=agg_result.applied_threshold,
            applied_mode=mode,
            final_decision=agg_result.final_decision,
            chunk_details=agg_result.chunk_details,
            processing_time_seconds=elapsed,
            chunk_window=window,
            chunk_overlap=overlap,
        )

        # ── Step 5: Generate outputs ───────────────────────────
        progress(0.92, desc="Generating report...")

        summary_html = _build_summary_html(detection_result, source_format)
        chunks_df = _build_chunks_dataframe(detection_result.chunk_details)

        # Generate PDF report
        pdf_bytes = _report_gen.generate_pdf(detection_result)
        pdf_path = os.path.join(
            tempfile.gettempdir(),
            f"VietAIDetector_Report_{doc_id}.pdf",
        )
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        progress(1.0, desc="Complete!")
        return summary_html, chunks_df, pdf_path

    except Exception as exc:
        elapsed = round(time.time() - start_time, 2)
        safe_exc = html.escape(str(exc))
        error_html = f"""
        <div style="padding: 20px; background: #fee; border-left: 4px solid #e74c3c;
                    border-radius: 8px; margin: 10px 0;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                {SVG_WARNING}
                <h3 style="color: #c0392b; margin: 0;">Processing Error</h3>
            </div>
            <p style="margin: 0; color: #333;">{safe_exc}</p>
            <p style="margin: 8px 0 0 0; color: #888; font-size: 0.85em;">
                Time elapsed: {elapsed}s
            </p>
        </div>
        """
        gr.Error(f"Error: {str(exc)}")
        return error_html, None, None


# HTML & DataFrame Builders

def _build_summary_html(result: DetectionResult, source_format: str = "") -> str:
    """Build the HTML summary card for detection results."""
    # Escape user input to prevent XSS
    safe_document_name = html.escape(result.document_name)
    safe_source_format = html.escape(source_format) if source_format else ""

    # Determine color and icon SVG based on decision
    if "AI-generated" == result.final_decision:
        color = "#e74c3c"
        bg = "#fdeaea"
        icon_svg = SVG_ROBOT
    elif "Human-written" == result.final_decision:
        color = "#27ae60"
        bg = "#eafaf1"
        icon_svg = SVG_PEN
    else:
        color = "#f39c12"
        bg = "#fef5e7"
        icon_svg = SVG_ALERT

    # Progress bar
    pct = result.ai_percentage
    bar_color = (
        "#27ae60" if pct == 0
        else "#e74c3c" if pct > 50
        else "#f39c12"
    )

    return f"""
    <div style="padding: 24px; background: {bg}; border-left: 5px solid {color};
                border-radius: 12px; margin: 8px 0; font-family: 'Inter', sans-serif;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
            <span style="display: flex; align-items: center;">{icon_svg}</span>
            <div>
                <h2 style="margin: 0; color: {color}; font-size: 1.4em;">
                    {result.final_decision}
                </h2>
                <p style="margin: 4px 0 0 0; color: #555; font-size: 0.95em;">
                    {safe_document_name}
                </p>
            </div>
        </div>

        <div style="background: #eee; border-radius: 8px; height: 28px; margin: 12px 0;
                    overflow: hidden; position: relative;">
            <div style="background: {bar_color}; height: 100%; width: {pct}%;
                        border-radius: 8px; transition: width 0.8s ease;
                        display: flex; align-items: center; justify-content: center;
                        min-width: 60px;">
                <span style="color: white; font-weight: bold; font-size: 0.85em;
                            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                    {pct}% AI
                </span>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                    gap: 12px; margin-top: 16px;">
            <div style="background: white; padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-size: 1.5em; font-weight: bold; color: {color};">
                    {result.ai_percentage}%
                </div>
                <div style="font-size: 0.8em; color: #888;">AI Ratio</div>
            </div>
            <div style="background: white; padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-size: 1.5em; font-weight: bold; color: #333;">
                    {result.total_chunks}
                </div>
                <div style="font-size: 0.8em; color: #888;">Total Chunks</div>
            </div>
            <div style="background: white; padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-size: 1.5em; font-weight: bold; color: #e74c3c;">
                    {result.ai_chunk_count}
                </div>
                <div style="font-size: 0.8em; color: #888;">AI Chunks</div>
            </div>
            <div style="background: white; padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-size: 1.5em; font-weight: bold; color: #27ae60;">
                    {result.total_chunks - result.ai_chunk_count}
                </div>
                <div style="font-size: 0.8em; color: #888;">Human Chunks</div>
            </div>
        </div>

        <div style="margin-top: 12px; padding: 8px 12px; background: white;
                    border-radius: 8px; font-size: 0.85em; color: #666;">
            Threshold: <b>{result.applied_threshold:.4f}</b> ({result.applied_mode})
            &nbsp;|&nbsp; Processing time: <b>{result.processing_time_seconds:.2f}s</b>
            {f'&nbsp;|&nbsp; Source: <b>{safe_source_format}</b>' if safe_source_format else ''}
        </div>
    </div>
    """


def _build_chunks_dataframe(chunks: list[ChunkDetail]) -> pd.DataFrame:
    """Build a pandas DataFrame for the chunks detail table."""
    rows = []
    for c in chunks:
        text_preview = c.text[:150]
        if len(c.text) > 150:
            text_preview += "..."
        rows.append({
            "Chunk": c.chunk_index,
            "Score": round(c.score, 4),
            "Label": f"{'[AI]' if c.label == 'AI' else '[Human]'} {c.label}",
            "Tokens": c.token_count,
            "Content (first 150 chars)": text_preview,
        })
    return pd.DataFrame(rows)


# Gradio App Builder

# Custom CSS for enhanced visual appearance
_CSS = """
.main-header {
    text-align: center;
    padding: 20px 0;
}
.main-header h1 {
    font-size: 2em;
    background: linear-gradient(135deg, #e74c3c, #3498db);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}
.main-header p {
    color: #888;
    font-size: 0.95em;
}
.disclaimer {
    font-size: 0.85em;
    color: #888;
}
.chunks-table-container {
    max-height: 420px;
    overflow-y: auto !important;
}
.chunks-table-container .table-wrap {
    max-height: 400px;
    overflow-y: auto !important;
}
/* Ensure the summary output has enough initial height so the progress bar is immediately visible */
#summary-output {
    min-height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
/* Hide duplicate progress trackers on secondary output components */
.hide-progress > div:has(.progress-text) {
    display: none !important;
}
"""

# Vietnamese sample text for demonstration
_SAMPLE_TEXT = (
    "Trí tuệ nhân tạo (AI) đang ngày càng phát triển mạnh mẽ và được ứng dụng "
    "rộng rãi trong nhiều lĩnh vực của đời sống. Từ y tế, giáo dục đến sản xuất "
    "công nghiệp, AI đã và đang thay đổi cách con người làm việc và tương tác "
    "với thế giới xung quanh. Tuy nhiên, sự phát triển nhanh chóng của AI cũng "
    "đặt ra nhiều thách thức về đạo đức, bảo mật và quyền riêng tư. Việc phát "
    "hiện văn bản do AI sinh ra trở thành một nhu cầu cấp thiết trong bối cảnh "
    "công nghệ hiện đại, đặc biệt trong lĩnh vực giáo dục và truyền thông."
)


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application.

    Returns:
        gr.Blocks application ready to launch.
    """
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.red,
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
    )

    with gr.Blocks(css=_CSS, theme=theme, title=APP_NAME) as app:

        # Header
        gr.HTML(f"""
        <div class="main-header">
            <div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
                {SVG_SEARCH}
                <h1>{APP_NAME}</h1>
            </div>
            <p>{APP_DESCRIPTION}</p>
            <p style="font-size: 0.8em; color: #aaa;"><a href="mailto:trieunh@ntu.edu.vn" style="color: inherit; text-decoration: none; hover: underline;">Developed by Trieu N.H.</a> - v{APP_VERSION}</p>
        </div>
        """)

        # Input Section
        with gr.Row():
            with gr.Column(scale=3):
                input_text = gr.Textbox(
                    value=_SAMPLE_TEXT,
                    placeholder="Enter or paste Vietnamese text to analyze...",
                    lines=10,
                    max_lines=30,
                    label="Input Text",
                    info="Enter text directly or upload a file on the right.",
                )
            with gr.Column(scale=1):
                input_file = gr.File(
                    label="Upload File",
                    file_types=[".docx", ".pdf", ".txt"],
                    type="filepath",
                )
                mode_dropdown = gr.Dropdown(
                    choices=list(THRESHOLD_MODES.keys()),
                    value=DEFAULT_MODE,
                    label="Detection Mode",
                    info="Youden: balanced | Closest Point: near-perfect | Low FPR: fewer false alarms",
                )
                with gr.Accordion("Advanced Settings", open=False):
                    window_slider = gr.Slider(
                        minimum=100, maximum=MAX_MODEL_TOKENS, step=10, value=CHUNK_WINDOW,
                        label="Chunk Window Size",
                        info="Maximum tokens per chunk. Higher reduces speed.",
                    )
                    overlap_slider = gr.Slider(
                        minimum=0, maximum=500, step=10, value=CHUNK_OVERLAP,
                        label="Chunk Overlap",
                        info="Token overlap between consecutive chunks.",
                    )

        # Action Buttons
        with gr.Row():
            submit_btn = gr.Button(
                "Analyze",
                variant="primary",
                size="lg",
            )
            clear_btn = gr.ClearButton(
                value="Clear",
                size="lg",
            )

        # Output Section
        with gr.Row():
            summary_output = gr.HTML(
                label="Detection Summary",
                elem_id="summary-output",
            )

        with gr.Row():
            chunks_table = gr.Dataframe(
                label="Chunk-Level Details",
                interactive=False,
                wrap=True,
                elem_classes="chunks-table-container hide-progress",
            )

        with gr.Row():
            pdf_download = gr.File(
                label="Download PDF Report",
                interactive=False,
                elem_classes="hide-progress",
            )

        # Notes & Explanation
        with gr.Accordion("Notes & Explanation", open=False):
            gr.Markdown("""
            - **Algorithm:** VietBinoculars uses the Perplexity / Cross-Perplexity ratio
              between PhoGPT-4B (base) and PhoGPT-4B-Chat (chat) to detect AI-generated text.
            - **Detection Thresholds:**
                - *Youden (0.8607)*: Optimizes F1-score, balancing true positives and false positives.
                - *Closest Point (0.8729)*: Closest point to perfect classification on the ROC curve.
                - *Low FPR (0.7023)*: Prioritizes reducing false positives — suitable when avoiding false accusations.
            - **Recommendation:** Use 200-500 words for best accuracy.
            - **Limitations:** Very short text (<64 tokens) or text with many tables/formulas
              may affect results.
            - **Disclaimer:** Results are for reference only and do not replace expert judgment.
              We assume no liability for any decisions based on these results.
            """, elem_classes="disclaimer")

        with gr.Accordion("Cite Our Work", open=False):
            gr.Markdown("""
            ```bibtex
            @misc{nguyen2025vietbinocularszeroshotapproachdetecting,
                  title={VietBinoculars: A Zero-Shot Approach for Detecting
                         Vietnamese LLM-Generated Text},
                  author={Trieu Hai Nguyen and Sivaswamy Akilesh},
                  year={2025},
                  eprint={2509.26189},
                  archivePrefix={arXiv},
                  primaryClass={cs.CL},
                  url={https://arxiv.org/abs/2509.26189},
            }
            ```
            """)

        # Event Handlers
        submit_btn.click(
            fn=_run_detection,
            inputs=[input_text, input_file, mode_dropdown, window_slider, overlap_slider],
            outputs=[summary_output, chunks_table, pdf_download],
            show_progress="full",
            api_name=False,
        )

        clear_btn.add([
            input_text, input_file, summary_output, chunks_table, pdf_download,
        ])

    return app
