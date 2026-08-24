# VietAIDetector — Complete Project Structure

## Overview

A system for detecting AI-generated Vietnamese text using the VietBinoculars algorithm (PPL/X-PPL ratio) with the PhoGPT-4B model pair. Includes OCR support for scanned PDFs via Vintern-1B-v2 and long documents. Features a Gradio UI and is optimized for dual-GPU (e.g., Kaggle T4x2).

## Algorithm

```
VietBinoculars Score = perplexity(performer) / cross_entropy(observer → performer)
Low score → AI-generated  |  High score → Human-written
Default threshold: YOUDEN_THRESHOLD $\approx 0.9280$ (Youden's J)
```

## Directory Structure

```
VietAIDetector/
├── README.md                         ← Usage guide
├── requirements.txt                  ← Python dependencies
├── app.py                            ← Entry point: launch Gradio app
├── run_kaggle.sh                     ← Kaggle T4x2 deployment script
├── pyproject.toml                    ← Pytest configuration
├── mkdocs.yml                        ← MkDocs configuration
│
├── docs/                             ← Documentation (MkDocs)
│   ├── STRUCTURE.md                  ← This file
│   ├── ARCHITECTURE.md               ← Technical architecture & design
│   └── DEVELOPER_MANUAL.md           ← Complete developer guide
│
├── diagram/                          ← Architecture diagrams (Graphviz)
│
├── examples/                         ← Sample documents for testing
│
├── config/
│   ├── __init__.py
│   └── settings.py                   ← Constants: models, thresholds, chunking, OCR config, devices
│
├── schemas/
│   ├── __init__.py
│   └── models.py                     ← Dataclasses: ChunkDetail, DetectionResult, PreprocessResult
│
├── core/
│   ├── __init__.py
│   ├── metrics.py                    ← perplexity() + cross_entropy() (from original Binoculars)
│   └── scorer.py                     ← VietBinocularsScorer: PhoGPT-4B dual-GPU scoring
│
├── preprocessing/
│   ├── __init__.py
│   ├── text_utils.py                 ← Regex constants, Vietnamese text helpers
│   ├── document_reader.py            ← Read DOCX, PDF (native + scanned) → raw text
│   ├── normalizer.py                 ← Pipeline: de-hyphen, linebreaks, whitespace, filtering
│   └── ocr_engine.py                 ← VinternOCR: Vintern-1B-v2 for scanned PDF → text
│
├── processing/
│   ├── __init__.py
│   ├── chunker.py                    ← TextChunker: sliding window processing
│   └── aggregator.py                 ← ScoreAggregator: majority voting → final decision
│
├── reporting/
│   ├── __init__.py
│   └── pdf_report.py                 ← PDFReportGenerator: fpdf2 + NotoSans font
│
├── frontend/
│   ├── __init__.py
│   └── gradio_app.py                 ← Gradio UI: advanced settings, OCR progress, results
│
├── benchmarks/
│   ├── run_eval.py                   ← CLI benchmark evaluation script
│   ├── datasets/                     ← LLM-generated test corpora (JSON)
│   └── train_datasets/               ← Update datasets/ with new data
│       ├── final_updated_dataset_20000.csv
│       ├── train_dataset_from_VietBinoculars.csv
│       └── *.jsonl
│
└── tests/
    ├── __init__.py
    ├── conftest.py                    ← Test configuration + sys.path setup
    ├── test_text_utils.py             ← Unit tests: regex normalization
    ├── test_normalizer.py             ← Unit tests: preprocessing pipeline
    ├── test_chunker.py                ← Unit tests: sliding window chunking
    ├── test_aggregator.py             ← Unit tests: majority voting aggregation
    ├── test_ocr_engine.py             ← Unit tests: VinternOCR + scanned PDF handling
    └── test_report_generator.py       ← Unit tests: PDF generation
```

## Data Flow Pipeline

```
[Gradio Frontend]
  User inputs text / uploads file (.docx, .pdf, .txt)
  Selects detection mode (Youden / Closest Point / Low FPR)
  Adjusts Chunk Window and Overlap (Advanced Settings)
       ↓ Submit
[Backend Pipeline]
  ① DocumentReader
  ├─ Detect format (docx / pdf / plain)
  ├─ If PDF: attempt native text extraction (PyMuPDF)
  │   ├─ Text found → "pdf_native" → continue
  │   └─ No text layer → "pdf_scanned" → delegate to OCR
  │       └── VinternOCR (Vintern-1B-v2 on cuda:1)
  │           ├─ Convert pages to 300 DPI images
  │           ├─ Dynamic high-resolution tiling (448×448)
  │           ├─ Anti-hallucination: greedy decode, hard-coded prompt
  │           ├─ Post-validation: skip pages < 10 chars
  │           └─ Return concatenated page text
  ├─ If DOCX: extract via python-docx
  └─ If plain: decode UTF-8
       ↓
  ② TextNormalizer
  ├─ De-hyphenation (fix broken words)
  ├─ Remove mid-sentence line breaks
  ├─ Collapse whitespace
  └─ Filter short or noisy paragraphs
       ↓
  ③ TextChunker (PhoGPT tokenizer)
  ├─ Apply dynamic Window (W) and Overlap (O) from UI
  ├─ Calculate Stride (D = W - O)
  ├─ Tokenize full text → token IDs
  ├─ Sliding window generation
  └─ Merge short trailing chunks
       ↓
  ④ VietBinocularsScorer (PhoGPT-4B + PhoGPT-4B-Chat)
  ├─ Observer (cuda:0): compute logits
  ├─ Performer (cuda:1): compute logits
  ├─ PPL = CE(performer logits, next tokens)
  ├─ X-PPL = CE(observer probs → performer logits)
  └─ score = PPL / X-PPL (per chunk, batch_size=8)
       ↓
  ⑤ ScoreAggregator
  ├─ score < threshold → "AI"
  ├─ ai_pct = ai_chunks / total × 100
  ├─ ai_pct > 50% → "AI-generated"
  ├─ 0 < ai_pct ≤ 50% → "Human-written but contains AI-generated parts"
  └─ ai_pct = 0% → "Human-written"
       ↓
  ⑥ PDFReportGenerator
  └─ Generate PDF with AI (red) / Human (green) highlights
       ↓
[Gradio Frontend]
  Render: Summary card (with source format) + chunk details table
  Enable PDF report download
```

## Models Used

| Role | Model | Default Device | VRAM (bf16) |
|------|-------|----------------|-------------|
| Observer (base) | vinai/PhoGPT-4B | cuda:0 | ~9 GB |
| Performer (chat) | vinai/PhoGPT-4B-Chat | cuda:1 | ~9 GB |
| OCR (scanned PDF) | 5CD-AI/Vintern-1B-v2 | cuda:1 (lazy) | ~2.5 GB |

> **GPU Budget (Kaggle T4x2)**: cuda:0 = ~9/15 GB, cuda:1 = ~11.5/15 GB (when OCR is active)

## Detection Thresholds (VietBinoculars Paper)

Centralized in [`config/settings.py`](../config/settings.py) ($\text{Score} < \text{threshold} \implies \text{AI}$, $\text{Score} \ge \text{threshold} \implies \text{Human}$):

| Mode (`THRESHOLD_MODES`) | Constant | Value | Description |
|---|---|---|---|
| **Youden (Balanced F1)** *(Default)* | `YOUDEN_THRESHOLD` | $\approx 0.9280$ | Maximizes Youden's J (optimal balanced F1) |
| **Closest Point (Near-Perfect)** | `CLOSEST_POINT_THRESHOLD` | $\approx 0.9251$ | Closest to $(0,1)$ on ROC curve |
| **Low FPR (Fewer False Alarms)** | `FPR_THRESHOLD` | $\approx 0.8993$ | Low False Positive Rate mode ($\text{TPR@0.05FPR}$) |

## Dynamic Chunking Parameters

Values can be modified in the UI via the **Advanced Settings** accordion:

| Parameter | Default | Description |
|---------|---------|----------|
| Window (W) | 450 | Max tokens per chunk. UI Slider: 100 - 1024. |
| Overlap (O) | 100 | Token overlap between chunks. UI Slider: 0 - 500. |
| Stride (D) | W - O | Automatically computed sliding step size. |
| Min Chunk | 50 | (Internal logic) If a chunk is smaller, it merges to the previous one. |

## OCR Anti-Hallucination Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `do_sample` | `False` | Greedy decoding — no randomness |
| `temperature` | `0.0` | Eliminates sampling variability |
| `num_beams` | `1` | No beam search creativity |
| `repetition_penalty` | `1.0` | Preserves naturally repeated text |
| `max_new_tokens` | `2048` | Caps output, prevents runaway generation |
| Min page chars | `10` | Pages below this threshold are skipped |
