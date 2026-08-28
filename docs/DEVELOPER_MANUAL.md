# VietAIDetector — Developer Manual

**Version:** 1.1.1  
**Language:** Python 3.10+  
**License:** MIT  
**Contact:** trieunh@ntu.edu.vn

---

## Table of Contents

- [1. Abstract](#1-abstract)
- [2. Motivation and Problem Statement](#2-motivation-and-problem-statement)
  - [2.1 Limitations of Existing Approaches](#21-limitations-of-existing-approaches)
  - [2.2 Core Contributions and Design Principles](#22-core-contributions-and-design-principles)
- [3. Software Description](#3-software-description)
  - [3.1 Architecture Overview](#31-architecture-overview)
  - [3.2 Module-by-Module Reference](#32-module-by-module-reference)
- [4. Mathematical Foundations](#4-mathematical-foundations)
- [5. Installation and Deployment](#5-installation-and-deployment)
- [6. Usage Guide](#6-usage-guide)
- [7. Testing](#7-testing)
- [8. Benchmark Evaluation](#8-benchmark-evaluation)
- [9. Performance and Hardware Notes](#9-performance-and-hardware-notes)
- [10. Known Limitations and Responsible Use](#10-known-limitations-and-responsible-use)
- [11. Reuse and Extensibility](#11-reuse-and-extensibility)
- [12. Related Work and References](#12-related-work-and-references)

---

## 1. Abstract

**VietAIDetector** is an open-source software tool designed specifically for detecting AI-generated Vietnamese text using a zero-shot scoring approach. The system builds upon the *VietBinoculars* and *Binoculars* algorithmic frameworks, computing the ratio of *Perplexity* (PPL) to *Cross-Perplexity* (X-PPL) between two PhoGPT-4B language models — a base model acting as the observer and a chat-tuned model acting as the performer. A low PPL/X-PPL score indicates machine-generated content, whereas a higher score signifies human authorship.

The software provides a production-ready Gradio web interface that accepts inputs ranging from raw text to common document formats (`.docx`, `.pdf`, `.txt`), including scanned (image-based) documents processed via an integrated OCR pipeline powered by the Vintern-1B-v2 Vision-Language Model (VLM). To overcome the context size limitations of the underlying language models, VietAIDetector implements a configurable sliding-window chunking mechanism with short-trailing-chunk post-processing, enabling full-length analysis of exceptionally long documents without sequence truncation or feature dilution. Chunk-level scores are aggregated into a document-level verdict using a majority-voting strategy ($P_{\text{AI}}$) alongside transparent, color-coded chunk highlights.

Users can select optimal detection thresholds calibrated for balanced F1-score (Youden's $J$), high precision (Closest Point on ROC), or minimized false alarms ($\text{TPR@0.05FPR}$), which is critical for high-stakes academic and legal contexts. Upon completion, the system generates a downloadable, color-coded PDF report embedding full configuration metadata for auditability and provides structured JSON serialization for downstream NLP pipelines. The system is optimized for dual-GPU environments (e.g., Kaggle T4×2, 32 GB VRAM total) and includes comprehensive unit test coverage for all non-GPU components.

---

## 2. Motivation and Problem Statement

The rapid advancement and widespread adoption of Large Language Models (LLMs) — including OpenAI's GPT series, Google Gemini, and Anthropic Claude — have led to a surge in AI-generated text across academic, journalistic, and professional domains. This widespread adoption presents pressing challenges in distinguishing between machine-generated content and human-written text:

- **Academic Integrity**: In higher education, students may misuse generative AI tools to complete assignments and essays without attribution, undermining critical thinking and authentic learning.
- **Information Authenticity**: Malicious actors can exploit LLMs on digital platforms to generate synthetic disinformation, fake news, and manipulated content at scale, influencing public perception and discourse.

### 2.1 Limitations of Existing Approaches

Considerable research has produced open-source and commercial detection tools, including Binoculars, DetectGPT, GLTR, Radar, Ghostbuster, GPTZero, and Turnitin. However, existing solutions exhibit several critical limitations when applied to Vietnamese:

1. **English-Centric Bias**: The vast majority of detection tools are developed and calibrated for high-resource languages (primarily English). Vietnamese, despite having approximately 100 million speakers, remains significantly underserved with no dedicated open-source detection tools.
2. **Impracticality of Supervised Classifiers**: Supervised detection methods rely on large-scale labeled datasets that are costly and difficult to curate for Vietnamese. Moreover, with frequent updates and releases of new LLMs, supervised models suffer from rapid domain obsolescence and risk overfitting to specific generator styles or artifacts.
3. **Inability to Handle Long Documents**: Standard zero-shot algorithms typically truncate input texts exceeding the model's maximum context length ($L_{\text{max}}$) or suffer from statistical feature dilution and attention degradation on long sequences. Existing zero-shot tools lack automated sliding-window tokenization and document-level aggregation for long essays or multi-page reports.
4. **Lack of Multi-Format and Scanned Document Support**: Real-world educational workflows involve diverse document formats (`.docx`, native `.pdf`, scanned `.pdf`). Most research implementations only accept short, plain-text strings and fail on scanned documents containing diacritics and complex Vietnamese typography.

### 2.2 Core Contributions and Design Principles

**VietAIDetector** bridges the gap between theoretical zero-shot detection research and real-world deployment through six core design decisions:

1. **Zero-Shot Scoring Mechanism**: Eliminates the need for labeled training data or periodic model retraining by computing the perplexity-to-cross-perplexity ratio across an observer-performer model pair, adapting the VietBinoculars formulation.
2. **Sliding-Window Chunking for Long Documents**: Employs a dynamic sliding window ($W$) and stride ($D = W - O$) tokenization with an admissible minimum chunk constraint ($m$) to eliminate high-variance trailing artifacts while ensuring 100% document coverage without sequence truncation.
3. **Vietnamese-First Dual-Model Architecture & VLM OCR**: Integrates PhoGPT-4B and PhoGPT-4B-Chat natively pre-trained on Vietnamese text, paired with on-demand Vintern-1B-v2 OCR using deterministic decoding and anti-hallucination prompt engineering for scanned documents.
4. **Calibrated Threshold Selection & False-Alarm Minimization**: Supports flexible threshold modes, notably $\text{TPR@0.05FPR}$ to constrain false accusations below 5%, which is essential when misclassifying student work carries severe ethical or disciplinary consequences.
5. **End-to-End Operational Usability & Auditable Reporting**: Features an interactive Gradio web interface, dynamic parameter tuning, auditable PDF reports with color-coded chunk visualization and embedded run metadata, and hierarchical JSON outputs for automated downstream integration.
6. **Accessible and Reproducible Deployment**: Provides a turnkey shell script (`run_kaggle.sh`) optimized for free-tier dual NVIDIA T4 GPUs on Kaggle, lowering the computational barrier for educational institutions and researchers.

---

## 3. Software Description

### 3.1 Architecture Overview

VietAIDetector is organized into **five distinct layers**, each responsible for a specific stage of the pipeline. This separation of concerns enables independent testing, model swapping, and modular extension.

```
+----------------------------------------------------------+
|  Layer 1 -- Presentation (frontend/gradio_app.py)        |
|  Gradio UI: input, settings, results, PDF download       |
+------------------+---------------------------------------+
                   |  filename + bytes / raw text
+------------------v---------------------------------------+
|  Layer 2 -- Data Ingestion & Preprocessing               |
|  document_reader.py -> normalizer.py -> [ocr_engine.py]  |
|  Supported: .docx, .pdf (native + scanned), .txt         |
+------------------+---------------------------------------+
                   |  cleaned normalized text
+------------------v---------------------------------------+
|  Layer 3 -- Processing Layer                             |
|  chunker.py    (sliding window tokenization)             |
|  aggregator.py (majority voting verdict)                 |
+------------------+---------------------------------------+
                   |  chunks
+------------------v---------------------------------------+
|  Layer 4 -- Core Detection Layer (VietBinoculars)        |
|  scorer.py     (VietBinoculars dual-GPU scoring)         |
|  metrics.py    (PPL / X-PPL functions)                   |
+------------------+---------------------------------------+
                   |  DetectionResult
+------------------v---------------------------------------+
|  Layer 5 -- Reporting (reporting/pdf_report.py)          |
|  Downloadable PDF with color-coded highlights            |
+----------------------------------------------------------+
```

> **Note:** The execution order crosses Layer 3 and Layer 4: `chunker` (Layer 3) → `scorer` (Layer 4) → `aggregator` (Layer 3). The chunker splits text into overlapping windows *before* scoring; the aggregator applies majority voting *after* all chunks are scored.

**Data flow (pipeline sequence):**

```
User Input (text / file)
        |
        v
[DocumentReader] --> detect format (docx / pdf / txt)
        |
        +--[pdf_native]--> extract text blocks, remove header/footer
        |
        +--[pdf_scanned]--> [VinternOCR] --> 300 DPI render --> tiled VLM inference
        |
        +--[docx]---------> extract paragraphs via python-docx
        |
        v
[TextNormalizer] --> de-hyphenate, filter short paragraphs, collapse whitespace
        |
        v
[TextChunker] --> sliding window (W=450, O=100, D=350) --> List[ChunkDetail]
        |
        v
[VietBinocularsScorer]
    |                   |
    v                   v
PhoGPT-4B          PhoGPT-4B-Chat
(cuda:0, PPL)      (cuda:1, X-PPL)
    |                   |
    +-------------------+
            |
            v
    score = PPL / X-PPL  (per chunk)
            |
            v
[ScoreAggregator] --> majority vote --> final_decision + ai_percentage
            |
            v
[PDFReportGenerator] --> color-highlighted PDF bytes
            |
            v
     Gradio UI (summary card + chunks table + PDF download)
```

---

### 3.2 Module-by-Module Reference

#### `config/settings.py` — Application Configuration

All constants and tunable parameters are centralized in this file. No magic numbers appear elsewhere in the codebase.

| Constant | Type | Default | Description |
|----------|------|---------|-------------|
| `OBSERVER_MODEL` | `str` | `"vinai/PhoGPT-4B"` | HuggingFace model ID for the base observer |
| `PERFORMER_MODEL` | `str` | `"vinai/PhoGPT-4B-Chat"` | HuggingFace model ID for the chat performer |
| `HF_TOKEN` | `str \| None` | `env:HF_TOKEN` | HuggingFace API token (for gated models) |
| `YOUDEN_THRESHOLD` | `float` | `0.9280` | Default detection threshold (Youden's J) |
| `CLOSEST_POINT_THRESHOLD` | `float` | `0.9251` | ROC closest-to-(0,1) threshold |
| `FPR_THRESHOLD` | `float` | `0.8993` | Low False Positive Rate threshold |
| `THRESHOLD_MODES` | `dict` | (3 modes) | UI-friendly names mapped to threshold values |
| `DEFAULT_MODE` | `str` | `"Youden..."` | Default detection mode in Gradio |
| `CHUNK_WINDOW` | `int` | `450` | Max tokens per chunk (W) |
| `CHUNK_OVERLAP` | `int` | `100` | Token overlap between chunks (O) |
| `CHUNK_STRIDE` | `int` | `350` | Stride = W - O |
| `CHUNK_MIN_TOKENS` | `int` | `50` | Min tokens; shorter chunks are merged |
| `DEVICE_1` | `str` | `"cuda:0"` | GPU for observer model |
| `DEVICE_2` | `str` | `"cuda:1"` | GPU for performer model (falls back to `cuda:0`) |
| `USE_BFLOAT16` | `bool` | `True` | bfloat16 precision (saves VRAM) |
| `MAX_MODEL_TOKENS` | `int` | `768` | Hard token limit for model forward pass |
| `SCORER_BATCH_SIZE` | `int` | `8` | Chunks per inference batch |
| `MINIMUM_TOKENS` | `int` | `64` | Min tokens for a valid analysis |
| `OCR_MODEL` | `str` | `"5CD-AI/Vintern-1B-v2"` | VLM for scanned PDF OCR |
| `OCR_DEVICE` | `str` | `DEVICE_2` | GPU device for OCR model (shares cuda:1 with performer) |
| `OCR_MAX_NEW_TOKENS` | `int` | `2048` | Max OCR output tokens per page |
| `OCR_MAX_IMAGE_TILES` | `int` | `12` | Max tiles for dynamic high-res tiling |
| `OCR_PROMPT` | `str` | `"Trích xuất chính xác..."` | Vietnamese OCR prompt (anti-hallucination design) |
| `OCR_MIN_PAGE_CHARS` | `int` | `10` | Skip pages with fewer characters |
| `FONT_PATH` | `str` | `/tmp/NotoSans-Regular.ttf` | Path to Vietnamese TTF font |
| `FONT_URL` | `str` | `"https://github.com/..."` | URL to download NotoSans font if missing |
| `APP_NAME` | `str` | `"VietAIDetector"` | Application name |
| `APP_VERSION` | `str` | `"1.1.1"` | Software version string |

All of these can be overridden via environment variables (for `HF_TOKEN`, `FONT_PATH`) or by editing the file directly.

---

#### `schemas/models.py` — Data Models

Three Python `dataclass` objects serve as the typed data contracts exchanged between modules. These replace Pydantic to minimize dependencies.

**`ChunkDetail`** — Represents a single text chunk with its detection result.

```python
@dataclass
class ChunkDetail:
    chunk_index: int        # 1-based position in the document
    text: str               # decoded Vietnamese text of this chunk
    token_count: int        # number of BPE tokens
    score: float = 0.0      # VietBinoculars score (PPL / X-PPL); lower -> AI
    label: str = ""         # "AI" or "Human" (set by ScoreAggregator)
```

**`PreprocessResult`** — Carries the output of the preprocessing pipeline.

```python
@dataclass
class PreprocessResult:
    document_name: str      # original filename
    source_format: str      # "plain" | "docx" | "pdf_native" | "pdf_scanned"
    extraction_status: str  # "success" | "error"
    cleaned_text: str       # normalized text ready for chunking
    error_message: str      # populated if extraction_status == "error"
```

**`DetectionResult`** — The final, fully-populated result object passed to the UI and PDF generator.

```python
@dataclass
class DetectionResult:
    document_id: str               # short UUID for this run
    document_name: str
    total_chunks: int
    ai_chunk_count: int
    ai_percentage: float           # (ai_chunk_count / total_chunks) * 100
    applied_threshold: float       # threshold value used
    applied_mode: str              # threshold mode name
    final_decision: str            # "AI-generated" | "Human-written" |
                                   # "Human-written but contains AI-generated parts"
    chunk_details: list[ChunkDetail] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    chunk_window: int = CHUNK_WINDOW    # default: 450
    chunk_overlap: int = CHUNK_OVERLAP  # default: 100
```

---

#### `core/metrics.py` — Binoculars Metric Functions

Implements the two fundamental mathematical operations of the Binoculars algorithm. These functions are stateless and operate directly on PyTorch tensors; they contain no model loading logic.

**`perplexity(encoding, logits, median, temperature)`**

Computes per-sample perplexity from the **performer** model's output logits. Internally, it uses `CrossEntropyLoss(reduction="none")`, shifts logits and labels by one position (standard language modeling), and masks padded positions via `attention_mask`. Aggregation is mean by default; set `median=True` for robustness to outliers.

**`entropy(p_logits, q_logits, encoding, pad_token_id, ...)`**

Computes per-sample cross-perplexity. The observer's logits (`p_logits`) are converted to a probability distribution via softmax, and then the cross-entropy from that distribution to the performer's logits (`q_logits`) is computed. This captures how much the chat-tuned model diverges from the base model's next-token predictions.

---

#### `core/scorer.py` — VietBinocularsScorer

The central inference class that orchestrates dual-model loading and chunk scoring.

**Class: `VietBinocularsScorer`**

```python
class VietBinocularsScorer:
    observer:   AutoModelForCausalLM   # PhoGPT-4B on cuda:0
    performer:  AutoModelForCausalLM   # PhoGPT-4B-Chat on cuda:1
    tokenizer:  AutoTokenizer          # shared; loaded from observer
```

**`__init__(observer_name, performer_name, device1, device2, use_bf16, max_tokens)`**

Loads both models from HuggingFace Hub onto their respective GPU devices using `device_map={"": device}` for explicit single-device placement (avoiding `accelerate` auto-sharding, which is incompatible with cross-device tensor operations). Both models are set to `.eval()` and gradient computation is globally disabled via `torch.set_grad_enabled(False)`.

**`compute_score(text) -> float | list[float]`**

Accepts a single string or a list of strings. Tokenizes the batch, runs a forward pass through both models, calls `perplexity()` and `entropy()`, and returns `PPL / X-PPL` per sample.

**`score_chunks(chunks) -> list[ChunkDetail]`**

Processes a list of `ChunkDetail` objects in batches of `SCORER_BATCH_SIZE` (default: 8) to avoid OOM on T4 GPUs. Populates the `score` field of each `ChunkDetail` in-place and rounds to 6 decimal places.

---

#### `preprocessing/text_utils.py` — Vietnamese Text Utilities

Defines compiled regex patterns and helper functions used throughout the preprocessing pipeline.

| Symbol | Purpose |
|--------|---------|
| `DEHYPHEN_RE` | Fix hyphenated line-breaks from PDF text extraction ("phát tri-\nển" -> "phát triển") |
| `LINEBREAK_RE` | Remove mid-sentence line breaks; preserve only after sentence-ending punctuation (`.!?…`) |
| `WHITESPACE_RE` | Collapse spaces, tabs, NBSP (`\u00a0`), and BOM (`\ufeff`) into a single space |
| `NON_SEMANTIC_RE` | Detect non-semantic characters for noise ratio calculation |

The `À-ỹ` Unicode range (U+00C0–U+1EF9) in the patterns covers all Vietnamese diacritics and tones, ensuring Vietnamese text is not incorrectly classified as special characters.

Helper functions:
- `count_words(text) -> int` — whitespace-split word count
- `special_char_ratio(text) -> float` — ratio of non-semantic characters; values > 0.5 indicate non-prose content (tables, formulas, code)

---

#### `preprocessing/document_reader.py` — Document Reader

Handles text extraction from multiple document formats. Format detection is based on file extension.

**Class: `DocumentReader`**

**`detect_format(filename) -> str`**  
Returns `"docx"`, `"pdf"`, or `"plain"` based on the lowercase file extension.

**`extract_from_docx(content: bytes) -> str`**  
Uses `python-docx` to traverse the document paragraph tree and join non-empty paragraphs with newlines. Only `doc.paragraphs` is traversed; images, charts, tables, headers, and footers from Word's XML schema are not included.

**`extract_from_pdf(content: bytes) -> tuple[str, str]`**  
Uses PyMuPDF (`fitz`) to extract text blocks page-by-page. Header and footer regions are excluded by filtering blocks whose bounding box lies in the top 10% or bottom 10% of the page height:

```python
if y1 < page_height * 0.10 or y0 > page_height * 0.90:
    continue  # skip header / footer block
```

Returns `("", "pdf_scanned")` if no text layer is found, triggering OCR fallback.

**`read(filename, content, ocr_engine) -> tuple[str, str]`**  
The main entry point. Routes to the appropriate extraction method and delegates scanned PDFs to the provided `VinternOCR` instance. Raises `UnsupportedFormatError` if a scanned PDF is detected but no OCR engine is provided.

---

#### `preprocessing/normalizer.py` — Text Normalizer

Orchestrates the complete preprocessing pipeline: document reading → de-hyphenation → paragraph filtering → per-paragraph normalization.

**Class: `TextNormalizer`**

Holds a `DocumentReader` instance and a lazily-initialized `VinternOCR` engine. The OCR engine is not loaded until the first scanned PDF is encountered.

**`normalize(text) -> str`**  
Applies three normalization rules in sequence:
1. De-hyphenation (`DEHYPHEN_RE`)
2. Mid-sentence linebreak removal (`LINEBREAK_RE`)
3. Whitespace collapsing (`WHITESPACE_RE`)

**`filter_paragraphs(text) -> list[str]`**  
Splits text on `\n`, then discards any paragraph that:
- Has fewer than **5 words** (page numbers, figure captions, short labels)
- Has `special_char_ratio > 0.50` (tables, code blocks, formula-heavy content)

This step is crucial: table data and code snippets produce anomalously low perplexity values that would distort the detection score if not removed.

**`preprocess(filename, content) -> PreprocessResult`**  
Top-level pipeline method:
1. `DocumentReader.read()` → raw text + source format
2. `DEHYPHEN_RE` on the full text
3. `filter_paragraphs()` → list of valid paragraphs
4. `normalize()` on each surviving paragraph
5. `"\n".join(paragraphs)` → cleaned text

Returns a `PreprocessResult` dataclass. On any exception, returns with `extraction_status="error"` and a descriptive `error_message`.

---

#### `preprocessing/ocr_engine.py` — Vintern OCR Engine

Extracts Vietnamese text from scanned (image-based) PDFs using the Vintern-1B-v2 Vision-Language Model.

**Image preprocessing pipeline (from InternVL2/Vintern architecture):**

1. `_build_transform(input_size=448)` — torchvision pipeline: RGB conversion → bicubic resize to 448×448 → tensor → ImageNet normalization
2. `_find_closest_aspect_ratio(...)` — selects the best tile layout `(w_tiles, h_tiles)` from a candidate set
3. `_dynamic_preprocess(image, max_num=12)` — splits the image into up to 12 tiles of 448×448 pixels, optionally appending a global thumbnail tile for global context
4. `load_image(image) -> Tensor` — returns a `(num_tiles, 3, 448, 448)` tensor

**Class: `VinternOCR`**

**`__init__(device)`**  
Initializes with `model = None`; model is not loaded until `_ensure_loaded()` is first called.

**`_ensure_loaded()`**  
Lazy-loads Vintern-1B-v2 using `AutoModel` with `attn_implementation="eager"` (avoids the optional `flash_attn` dependency). Includes a monkey-patch on `transformers.dynamic_module_utils.check_imports` to bypass the flash_attn import check in the remote model code without modifying the HuggingFace source.

**`extract_page(image: PIL.Image) -> str`**  
Runs VLM inference on a single page image. The generation configuration is **hard-coded** and cannot be overridden by the caller:

```python
generation_config = dict(
    max_new_tokens=2048,
    do_sample=False,       # greedy decoding -- no randomness
    temperature=0.0,       # eliminates sampling variability
    num_beams=1,           # no beam search
    repetition_penalty=1.0 # no penalty (preserves naturally repeated source text)
)
```

Before each inference call, `torch.cuda.set_device(self.device)` is called explicitly. This is required because the Vintern-1B-v2 remote modeling code contains hardcoded `.cuda()` calls that would otherwise target the wrong device in a multi-GPU environment.

Output validation: if the extracted text is fewer than `OCR_MIN_PAGE_CHARS` (default: 10) characters, the page is considered blank and skipped with a warning log.

**`extract_from_pdf(pdf_bytes: bytes) -> str`**  
Iterates over all PDF pages, renders each at **300 DPI** using `fitz.Matrix(300/72, 300/72)`, converts to PIL Images, and calls `extract_page()` on each. Returns the concatenated text of all non-blank pages.

**Anti-Hallucination Design:**

The OCR prompt (`OCR_PROMPT` in `config/settings.py`) is written in **Vietnamese**, not English, because Vintern-1B-v2 is fine-tuned on Vietnamese instruction datasets (`Viet-OCR-VQA`, `Viet-Doc-VQA`). Vietnamese prompts are in-distribution for this model and produce more faithful extraction. English prompts are out-of-distribution and increase hallucination risk.

The prompt instructs the model to: extract all text exactly as-is, preserve paragraph structure, punctuation, and any spelling errors present in the original, and add no explanations, reformatting, or additional content.

---

#### `processing/chunker.py` — Text Chunker

Splits normalized Vietnamese text into overlapping chunks for processing by the language models.

**`ChunkOutput`** (dataclass)

```python
@dataclass
class ChunkOutput:
    document_id: str
    total_tokens: int
    total_chunks: int
    chunks: list[ChunkDetail]
```

**Class: `TextChunker`**

**`__init__(tokenizer_name_or_obj, window, overlap, stride, min_chunk)`**  
Accepts either a HuggingFace model ID string (auto-loads tokenizer) or a pre-built tokenizer object. The latter is used in production to reuse the tokenizer already loaded by `VietBinocularsScorer`, avoiding a redundant download.

**`chunk(text, document_id, window=None, overlap=None) -> ChunkOutput`**  
Implements the sliding window algorithm:

1. Tokenize: `ids = tokenizer.encode(text, add_special_tokens=False)` → token ID list of length `N`
2. If `N == 0`: return empty `ChunkOutput`
3. If `N <= W`: single chunk spanning `(0, N)`
4. If `N > W`: generate spans with stride `D`:
   - `start = 0`; `end = min(start + W, N)`; append `(start, end)`; advance `start += D`; stop when `end == N`
5. **Short chunk merging**: if the final span has fewer than `min_chunk` tokens, pop it and extend the previous span's end to `N`
6. Decode each span: `tokenizer.decode(ids[s:e], skip_special_tokens=True)`
7. Wrap in `ChunkDetail(chunk_index=idx, text=decoded, token_count=e-s)`

Per-call `window` and `overlap` overrides are supported (used by the UI's Advanced Settings sliders); the stride is recomputed as `window - overlap` for that specific call.

---

#### `processing/aggregator.py` — Score Aggregator

**`AggregationResult`** (dataclass)

```python
@dataclass
class AggregationResult:
    total_chunks: int
    ai_chunk_count: int
    ai_percentage: float      # (ai_chunk_count / total_chunks) * 100
    applied_threshold: float
    final_decision: str
    chunk_details: list[ChunkDetail]
```

**Class: `ScoreAggregator`**

**`aggregate(chunks, threshold=YOUDEN_THRESHOLD) -> AggregationResult`**

1. For each `ChunkDetail`: if `chunk.score < threshold` → `chunk.label = "AI"`, increment `ai_count`; otherwise `chunk.label = "Human"`
2. Compute `ai_pct = round((ai_count / total) * 100, 2)`
3. Apply three-class decision:
   - `ai_pct > 50.0` → `"AI-generated"`
   - `0 < ai_pct <= 50.0` → `"Human-written but contains AI-generated parts"`
   - `ai_pct == 0.0` → `"Human-written"`

Raises `ValueError` if the input list is empty.

---

#### `reporting/pdf_report.py` — PDF Report Generator

**Class: `PDFReportGenerator`**

Uses `fpdf2` for lightweight PDF generation without heavy dependencies. Requires a Vietnamese-capable TTF font (NotoSans-Regular.ttf) for correct diacritic rendering.

**`_ensure_font(font_path) -> str`**  
Validates the font at `font_path` using TrueType/OpenType magic bytes (`_is_valid_ttf`). If the file is missing or corrupted (e.g. an HTML 404 response), it automatically cleans up and re-downloads `NotoSans-Regular.ttf` from GitHub (`notofonts/noto-fonts`) with multiple fallback URLs.

**`generate_pdf(result: DetectionResult) -> bytes`**  
Builds and returns the complete PDF as bytes. The PDF contains:

1. **Header**: Application name + timestamp (GMT+7)
2. **Summary table**: document name, AI ratio, decision, threshold mode/value, chunk window/overlap, total/AI/human chunk counts, processing time
3. **Chunk-Level Details**: for each chunk, a color-coded header bar followed by the full chunk text (no truncation):
   - AI chunk: background `(255, 200, 200)` (light red)
   - Human chunk: background `(210, 245, 210)` (light green)

---

#### `frontend/gradio_app.py` — Gradio Web Application

**Module-level state (lazy initialization):**

```python
_scorer     = None              # VietBinocularsScorer (heavy GPU load)
_chunker    = None              # TextChunker (reuses scorer tokenizer)
_normalizer = TextNormalizer()  # lightweight, initialized at import
_aggregator = ScoreAggregator()
_report_gen = PDFReportGenerator()
```

Models are **not** initialized at import time, avoiding GPU memory allocation during testing or CPU-only environments.

**`_run_detection(text, file_obj, mode, window, overlap, progress)`**  
Main backend handler triggered on form submission:

| Step | Progress | Action |
|------|----------|--------|
| 1 | 5% | Read input (text or file). Call `TextNormalizer.preprocess()`. Emit `gr.Warning()` for scanned PDFs. |
| 2 | 15% | Tokenize and chunk via `TextChunker.chunk()` with UI-specified window/overlap. Enforce `MINIMUM_TOKENS` (64). |
| 3 | 30% | Score all chunks via `VietBinocularsScorer.score_chunks()`. |
| 4 | 85% | Aggregate via `ScoreAggregator.aggregate()`. Build `DetectionResult`. |
| 5 | 92%→100% | Build HTML summary, chunks DataFrame, generate PDF to temp file. |

Returns `(summary_html, chunks_dataframe, pdf_file_path)`.

**`create_app() -> gr.Blocks`**  
Builds the Gradio UI using the `Soft` theme with `Inter` (Google Font). Key UI components:

- `gr.Textbox` — direct text input (pre-loaded with a Vietnamese sample text)
- `gr.File` — file upload (`.docx`, `.pdf`, `.txt`)
- `gr.Dropdown` — detection mode selector (Youden / Closest Point / Low FPR)
- `gr.Accordion("Advanced Settings")` — Chunk Window (100–1024) and Overlap (0–500) sliders
- `gr.Button("Analyze")` / `gr.ClearButton("Clear")`
- `gr.HTML` — summary result card with progress bar and metrics grid
- `gr.Dataframe` — chunk-level detail table (scrollable, 420px max height)
- `gr.File` — PDF report download

---

## 4. Mathematical Foundations

### 4.1 VietBinoculars Score

For a text chunk $C_k$, the VietBinoculars score is:

$$\text{score}(C_k) = \frac{\text{PPL}(C_k)}{\text{X-PPL}(C_k)}$$

where $\text{PPL}(C_k)$ is the cross-entropy loss of the **performer** model on $C_k$, and $\text{X-PPL}(C_k)$ is the cross-entropy from the **observer** distribution to the **performer** logits. A **low score** indicates AI-generated text; a **high score** indicates human-written text.

### 4.2 Perplexity (Performer)

Let $C_k = (x_1, x_2, \ldots, x_n)$ be a sequence of $n$ tokens. Let $q_\theta$ be the performer model. The cross-entropy loss (used as PPL proxy) shifts the logits to predict the next token:

$$\text{PPL}(C_k) = \frac{1}{\sum_{t=2}^{n} m_t} \sum_{t=2}^{n} \ell\!\left(q_\theta(x_{<t}),\; x_t\right) \cdot m_t$$

where $\ell(\cdot)$ is cross-entropy loss and $m_t$ is the attention mask (1 for real tokens, 0 for padding). Note that the summation is over $n-1$ shifted positions.

### 4.3 Cross-Perplexity (Observer → Performer)

Let $p_\phi$ be the observer model. Unlike standard perplexity, the cross-perplexity evaluates the divergence between the two models' next-token probability distributions at *every* position $t = 1 \dots n$ (including the prediction for the unseen $(n+1)$-th token):

$$\text{X-PPL}(C_k) = \frac{1}{\sum_{t=1}^{n} m_t} \sum_{t=1}^{n} \text{CE}\!\left(q_\theta(x_{\le t}),\; \text{softmax}(p_\phi(x_{\le t}))\right) \cdot m_t$$

Here, $\text{softmax}(p_\phi(x_{\le t}))$ converts the observer's target logits into a probability distribution, and $\text{CE}$ computes the cross-entropy from that target distribution to the performer's predicted logits $q_\theta(x_{\le t})$.

**Intuition:** For AI-generated text (produced by a model from the same family as the performer), both $p_\phi$ and $q_\theta$ assign high confidence to the same tokens, so the ratio $\text{PPL} / \text{X-PPL} \approx 1$ is low. For human text, the performer's per-token uncertainty is higher (larger PPL), while the observer-performer divergence increases proportionally less — yielding a higher score.

### 4.4 Sliding Window Chunking

Let the full text tokenize to $S = (x_1, \ldots, x_N)$ with parameters:
- $W$ = window size (default: 450)
- $O$ = overlap (default: 100)
- $D = W - O$ = stride (default: 350)
- $M$ = minimum chunk size (default: 50)

**Chunk generation:**

$$C_k = \bigl(x_{(k-1)D+1},\; \ldots,\; x_{\min((k-1)D+W,\; N)}\bigr), \quad k = 1, 2, \ldots, K$$

**Short chunk merging:** If $|C_K| < M$ (last chunk too short):

$$C_{K-1}^* = (x_{(K-2)D+1}, \ldots, x_N), \quad C_K \text{ removed}$$

### 4.5 Majority Voting Aggregation

Let $t^*$ be the selected threshold. For each chunk $C_k$, define the indicator:

$$I(C_k) = \begin{cases} 1 & \text{if } \text{score}(C_k) < t^* \quad (\text{"AI"}) \\ 0 & \text{otherwise} \quad (\text{"Human"}) \end{cases}$$

AI vote count and percentage:

$$\text{Vote}(S) = \sum_{k=1}^{K} I(C_k), \qquad P_\text{AI} = \frac{\text{Vote}(S)}{K} \times 100\%$$

**Three-class decision rule:**

$$\text{Decision}(S) = \begin{cases} \textit{AI-generated} & \text{if } P_\text{AI} > 50\% \\ \textit{Human-written but contains AI-generated parts} & \text{if } 0\% < P_\text{AI} \leq 50\% \\ \textit{Human-written} & \text{if } P_\text{AI} = 0\% \end{cases}$$

### 4.6 Detection Thresholds

Detection thresholds are centralized in [`config/settings.py`](../config/settings.py) and derived from VietBinoculars benchmark experiments:

- **Classification rule**:
  $$\text{Label}(\text{chunk}) = \begin{cases} \text{AI-generated} & \text{if } \text{score} < \text{threshold} \\ \text{Human-written} & \text{if } \text{score} \ge \text{threshold} \end{cases}$$

| Mode (`THRESHOLD_MODES`) | Constant | Threshold $t^*$ | Derivation / Purpose |
|---|---|---|---|
| **Youden (Balanced F1)** *(Default)* | `YOUDEN_THRESHOLD` | $\approx 0.9280$ | Maximizes Youden's J ($J = \text{TPR} - \text{FPR}$, optimal balanced F1) |
| **Closest Point (Near-Perfect)** | `CLOSEST_POINT_THRESHOLD` | $\approx 0.9251$ | Minimizes Euclidean distance from ROC curve to ideal point $(0, 1)$ |
| **Low FPR (Fewer False Alarms)** | `FPR_THRESHOLD` | $\approx 0.8993$ | Low False Positive Rate threshold ($\text{TPR@0.05FPR}$, minimizes false alarms) |

---

## 5. Installation and Deployment

### 5.1 Prerequisites

| Requirement | Specification |
|-------------|--------------|
| Python | >= 3.10 |
| CUDA | >= 11.8 |
| GPU VRAM | >= 24 GB total (recommended: 2x NVIDIA T4, 32 GB) |
| Disk space | ~20 GB (model weights: ~9 GB + ~9 GB + ~2.5 GB) |
| Internet | Required for HuggingFace model download on first run |

### 5.2 Local Installation

```bash
# Step 1: Clone the repository
git clone https://github.com/trieuntu/VietAIDetector.git
cd VietAIDetector

# Step 2: Create a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate   # Linux / macOS

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Set environment variables (optional)
export HF_TOKEN="hf_your_token_here"       # for gated models
export FONT_PATH="/tmp/NotoSans-Regular.ttf"  # auto-downloaded if missing

# Step 5: Launch
python app.py
```

The application is accessible at `http://localhost:7860`. With `share=True` in `app.py`, a temporary public URL is also printed to the console.

### 5.3 Kaggle Notebook Deployment

VietAIDetector is optimized for Kaggle T4x2 (2x NVIDIA T4, 15 GB VRAM each).

**Steps:**
1. Create a new Kaggle Notebook
2. **Settings → Accelerator**: select `GPU T4 x2`
3. **Settings → Internet**: enable
4. Upload `VietAIDetector/` folder to `/kaggle/working/`
5. In the first code cell, run:

```bash
!bash /kaggle/working/VietAIDetector/run_kaggle.sh
```

The `run_kaggle.sh` script performs five actions automatically:
1. Sets performance environment variables (`TORCHDYNAMO_DISABLE`, `TOKENIZERS_PARALLELISM`, `TRANSFORMERS_NO_ADVISORY_WARNINGS`)
2. Installs all dependencies: `pip install -q -r requirements.txt`
3. Downloads NotoSans font to `/tmp/NotoSans-Regular.ttf`
4. Verifies GPU count and VRAM
5. Launches `python3 app.py`

### 5.4 Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Optional | `None` | HuggingFace API token. Required if PhoGPT models are gated. |
| `FONT_PATH` | Optional | `/tmp/NotoSans-Regular.ttf` | Path to NotoSans TTF font. Auto-downloaded if not present. |
| `TORCHDYNAMO_DISABLE` | Optional | unset | Set to `1` to disable TorchDynamo compilation (improves stability). |
| `TOKENIZERS_PARALLELISM` | Optional | unset | Set to `false` to prevent tokenizer deadlocks in multi-threaded environments. |
| `TRANSFORMERS_NO_ADVISORY_WARNINGS` | Optional | unset | Set to `1` to suppress HuggingFace Transformers advisory warnings. |

### 5.5 First-Run Behavior

On first launch, HuggingFace models are downloaded and cached to `~/.cache/huggingface/`:
- PhoGPT-4B (~9 GB) and PhoGPT-4B-Chat (~9 GB): approximately **3–5 minutes**
- Vintern-1B-v2 (~2.5 GB): downloaded only on first scanned PDF submission (lazy loading)

---

## 6. Usage Guide

### 6.1 Web User Interface

Navigate to `http://localhost:7860` after launching.

**Input methods:**

| Method | Description |
|--------|-------------|
| Direct text | Paste Vietnamese text into the textbox (file upload takes priority if both are provided) |
| File upload | Upload `.docx`, `.pdf`, or `.txt` |

**Detection modes:**

| Mode | Threshold | Recommended use case |
|------|-----------|---------------------|
| Youden (Balanced F1) | `YOUDEN_THRESHOLD` $\approx 0.9280$ | General-purpose — balanced accuracy |
| Closest Point (Near-Perfect) | `CLOSEST_POINT_THRESHOLD` $\approx 0.9251$ | High-precision applications |
| Low FPR (Fewer False Alarms) | `FPR_THRESHOLD` $\approx 0.8993$ | When false accusations are costly (e.g., academic proceedings) |

**Advanced Settings:**
- **Chunk Window Size** (100–768 tokens, default: 450): Maximum tokens per analysis window. While PhoGPT-4B supports up to 8192 tokens, this parameter is hard-capped at `MAX_MODEL_TOKENS` (768) to prevent OOM errors on Kaggle T4 GPUs (15GB VRAM) given the $O(N^2)$ attention mechanism. Larger windows provide more context but increase processing time.
- **Chunk Overlap** (0–500 tokens, default: 100): Token overlap between consecutive windows. Higher overlap reduces boundary artifacts.

**Interpreting results:**

| Verdict | Condition |
|---------|-----------|
| AI-generated | > 50% of chunks scored below threshold |
| Human-written but contains AI-generated parts | 1–50% of chunks below threshold |
| Human-written | 0% of chunks below threshold |

The summary card displays: verdict icon, AI ratio progress bar, chunk statistics (AI / Human / Total), applied threshold, processing time, and detected source format.

### 6.2 Programmatic API

**End-to-end detection pipeline:**

```python
from preprocessing.normalizer import TextNormalizer
from processing.chunker import TextChunker
from processing.aggregator import ScoreAggregator
from core.scorer import VietBinocularsScorer
from config.settings import OBSERVER_MODEL, PERFORMER_MODEL, YOUDEN_THRESHOLD
import uuid

# Initialize once at startup
scorer     = VietBinocularsScorer(OBSERVER_MODEL, PERFORMER_MODEL)
normalizer = TextNormalizer()
chunker    = TextChunker(tokenizer_name_or_obj=scorer.tokenizer)  # reuse tokenizer (can set window=600, overlap=200 globally here)
aggregator = ScoreAggregator()

# Run pipeline on a text string
result = normalizer.preprocess("sample.txt", "Văn bản tiếng Việt cần kiểm tra...")
# You can override chunking parameters per document here:
chunk_output = chunker.chunk(result.cleaned_text, document_id=str(uuid.uuid4())[:8], window=600, overlap=200)
scored_chunks = scorer.score_chunks(chunk_output.chunks)
agg = aggregator.aggregate(scored_chunks, threshold=YOUDEN_THRESHOLD)

print(f"Decision : {agg.final_decision}")
print(f"AI ratio : {agg.ai_percentage:.1f}%")
for c in agg.chunk_details:
    print(f"  Chunk {c.chunk_index}: score={c.score:.4f}  label={c.label}")
```

**Score a single string:**

```python
score = scorer.compute_score("Trí tuệ nhân tạo đang phát triển mạnh mẽ.")
print(f"Score: {score:.4f}")  # < YOUDEN_THRESHOLD -> likely AI; > YOUDEN_THRESHOLD -> likely human
```

**Score a batch:**

```python
scores = scorer.compute_score(["Text 1...", "Text 2...", "Text 3..."])
# returns list[float]
```

**Process a PDF file:**

```python
with open("document.pdf", "rb") as f:
    result = normalizer.preprocess("document.pdf", f.read())
print(result.source_format)    # "pdf_native" or "pdf_scanned"
print(result.cleaned_text[:500])
```

**Generate a PDF report:**

```python
from reporting.pdf_report import PDFReportGenerator
from schemas.models import DetectionResult

gen = PDFReportGenerator()
pdf_bytes = gen.generate_pdf(detection_result)  # a DetectionResult object
with open("report.pdf", "wb") as f:
    f.write(pdf_bytes)
```

### 6.3 Classification Decision Reference

Chunk-level classification is binary based on the active threshold $t^*$:
- $\text{score} < t^* \implies \textbf{AI-generated}$
- $\text{score} \ge t^* \implies \textbf{Human-written}$

The overall document verdict is determined by the percentage of AI chunks ($P_{\text{AI}}$) via majority voting:

| Overall Verdict | Condition | Meaning |
|-----------------|-----------|---------|
| **AI-generated** | $P_{\text{AI}} > 50\%$ | More than half of the text chunks are classified as AI |
| **Human-written but contains AI-generated parts** | $0\% < P_{\text{AI}} \leq 50\%$ | Mixed authorship — document contains both AI and human segments |
| **Human-written** | $P_{\text{AI}} = 0\%$ | All analyzed text chunks are classified as human-written |

### 6.4 Source Format Labels

| `source_format` | Meaning |
|-----------------|---------|
| `plain` | Input was a raw text string |
| `docx` | Extracted from Microsoft Word (.docx) |
| `pdf_native` | PDF with an embedded text layer |
| `pdf_scanned` | Image-based PDF processed via Vintern-1B-v2 OCR |

---

## 7. Testing

### 7.1 Test Suite Overview

The `tests/` directory contains **6 test modules** covering all non-GPU-dependent components. GPU-dependent components (the scorer) are excluded to allow CI execution without hardware requirements.

| Test file | Component under test |
|-----------|---------------------|
| `test_text_utils.py` | `DEHYPHEN_RE`, `LINEBREAK_RE`, `WHITESPACE_RE`, `count_words()`, `special_char_ratio()` |
| `test_normalizer.py` | `TextNormalizer.normalize()`, `filter_paragraphs()`, `preprocess()` |
| `test_chunker.py` | `TextChunker.chunk()` — single chunk, multi-chunk, boundary merging, empty input, stride override |
| `test_aggregator.py` | `ScoreAggregator.aggregate()` — all three decisions, threshold boundaries, empty input error |
| `test_report_generator.py` | `PDFReportGenerator.generate_pdf()` — output validity, color scheme correctness |
| `test_ocr_engine.py` | `VinternOCR` with mock models, `_dynamic_preprocess()`, `load_image()`, output validation |

`tests/conftest.py` inserts the project root into `sys.path`, enabling absolute package imports without installation.

### 7.2 Running Tests

```bash
# Run all tests
pytest

# Verbose output
pytest -v

# Specific file
pytest tests/test_chunker.py -v

# Keyword filter
pytest -k "normalizer" -v
```

**Test configuration** (`pyproject.toml`):
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

### 7.3 Test Design Notes

- **`TextChunker` tests** use a whitespace-tokenizing mock tokenizer for fast, deterministic execution without loading PhoGPT-4B.
- **`VinternOCR` tests** mock `AutoModel` and `AutoTokenizer` to isolate image preprocessing and output validation logic from actual VLM inference.
- **`PDFReportGenerator` tests** verify that the output is non-empty bytes and that the correct RGB fill colors are applied to AI vs. Human chunks.

---

## 8. Benchmark Evaluation

The `benchmarks/` directory contains datasets and a CLI evaluation script for measuring detection accuracy across different LLM-generated corpora.

### 8.1 Directory Structure

```
benchmarks/
├── run_eval.py                              ← CLI evaluation script
├── datasets/
│   ├── claude-sonnet-4.6.json               ← 20 AI-generated documents
│   ├── gemini-3.7-flash.json                ← 20 AI-generated documents
│   └── gpt-5.6-luna.json                    ← 20 AI-generated documents
└── train_datasets/
    ├── final_updated_dataset_20000.csv
    ├── train_dataset_from_VietBinoculars.csv
    ├── gemini-3.7-flash-vietnamese_news_ai_dataset_5000.jsonl
    └── gpt-5.6-luna-vietnamese_news_ai_dataset_5000.jsonl
```

### 8.2 Dataset Format

Each JSON file in `benchmarks/datasets/` is a list of objects with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique document identifier |
| `text` | `str` | Full document text (Vietnamese) |
| `is_ai_generated` | `bool` | Ground-truth label |
| `model` | `str` | Generator model name (e.g., `"claude-sonnet-4.6"`) |
| `topic` | `str` | Document topic category |
| `sub_topic` | `str` | Document sub-topic category |
| `word_count` | `int` | Number of words in the document |

The training datasets in `benchmarks/train_datasets/` follow JSONL format (one JSON object per line) and were used to derive the detection thresholds reported in the VietBinoculars paper.

### 8.3 Running Evaluations

`benchmarks/run_eval.py` processes a dataset through the full detection pipeline and outputs per-document results as CSV.

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | *(required)* | Path to a `.json` dataset file or directory |
| `-o, --output` | auto-generated | Output CSV path |
| `-m, --mode` | `Youden (Balanced F1)` | Detection threshold mode |
| `-w, --window` | `450` | Chunk window size (tokens) |
| `--overlap` | `100` | Chunk overlap (tokens) |
| `--resume` | `false` | Skip documents already present in the output CSV |

**Examples:**

```bash
# Evaluate a single dataset with default parameters
python benchmarks/run_eval.py benchmarks/datasets/claude-sonnet-4.6.json

# Custom chunking parameters
python benchmarks/run_eval.py benchmarks/datasets/gpt-5.6-luna.json \
    --window 600 --overlap 200

# Evaluate all datasets in the directory
python benchmarks/run_eval.py benchmarks/datasets/

# Resume an interrupted run
python benchmarks/run_eval.py benchmarks/datasets/gemini-3.7-flash.json --resume
```

**Output CSV Schema (`CSV_COLUMNS`):**

| Column | Type | Description |
|--------|------|-------------|
| `document_id` | `str` | Unique document identifier from dataset or generated UUID |
| `ai_percentage` | `float` | Percentage of chunks classified as AI-generated ($0.0 - 100.0$) |
| `final_decision` | `str` | Overall verdict (`"AI-generated"`, `"Human-written"`, `"Human-written but contains AI-generated parts"`) |
| `applied_threshold` | `float` | Numeric threshold value used for chunk classification |
| `applied_mode` | `str` | Threshold mode name (e.g., `"Youden (Balanced F1)"`) |
| `chunk_window` | `int` | Token window size used ($W$) |
| `chunk_overlap` | `int` | Token overlap size used ($O$) |
| `total_chunks` | `int` | Total number of chunks evaluated |
| `ai_chunk_count` | `int` | Number of chunks classified as AI |
| `is_ai_generated` | `bool / str` | Ground-truth label from dataset (if present) |
| `model` | `str` | Generator model name from dataset metadata (if present) |
| `topic` | `str` | Document topic category from dataset metadata (if present) |
| `processing_time_seconds` | `float` | Total pipeline processing time in seconds |

> **Note:** Evaluation requires GPU access (dual T4 or equivalent). Results are written to `benchmarks/results/`, which is excluded from version control.

---

## 9. Performance and Hardware Notes

### 9.1 VRAM Budget

| Model | Device | VRAM (bfloat16) |
|-------|--------|-----------------|
| PhoGPT-4B (observer) | cuda:0 | ~9 GB |
| PhoGPT-4B-Chat (performer) | cuda:1 | ~9 GB |
| Vintern-1B-v2 (OCR, lazy-loaded) | cuda:1 | ~2.5 GB |

**On Kaggle T4x2 (15 GB per GPU):**
- `cuda:0`: ~9 / 15 GB (observer only)
- `cuda:1`: ~9 / 15 GB → ~11.5 / 15 GB when OCR is active

### 9.2 Single GPU Fallback

`DEVICE_2` falls back automatically when only one GPU is available:

```python
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1
```

Both models will share the same GPU, requiring at least **18 GB VRAM** (without OCR).

### 9.3 Processing Time Estimates (Kaggle T4x2)

| Input type | Approx. time |
|------------|-------------|
| Short text (< 450 tokens, 1 chunk) | 5–10 seconds |
| Medium document (~2,000 tokens, ~6 chunks) | 20–40 seconds |
| Long document (~10,000 tokens, ~29 chunks) | 2–4 minutes |
| Scanned PDF (5 pages, OCR + detection) | 3–6 minutes |

Estimates exclude first-run model download time.

### 9.4 Batch Size Tuning

`SCORER_BATCH_SIZE = 8` is conservative for T4 GPUs (16 GB each). On GPUs with larger VRAM (e.g., A100 80 GB), increasing this to 16 or 32 will reduce the number of forward-pass iterations proportionally.

---

## 10. Known Limitations and Responsible Use

### 10.1 Minimum Text Length

The system requires at least **64 tokens** of input text (`MINIMUM_TOKENS`). Below this threshold, language models have insufficient context to establish reliable perplexity estimates. For best accuracy, inputs of **200–500 words** are recommended.

### 10.2 OCR Accuracy Caveats

When processing scanned PDFs via Vintern-1B-v2:
- **Handwritten text** is not supported; the model targets printed documents
- **Low-resolution scans** (< 150 DPI) may produce garbled or incomplete output
- **Tables, charts, and diagrams** are typically not extracted correctly
- Despite anti-hallucination measures, VLMs can still occasionally insert plausible-looking text absent from the source image

When a scanned PDF is detected, a warning is displayed in the UI advising users to verify results against the source document.

### 10.3 Zero-Shot Detection Limitations

VietBinoculars is a zero-shot method: no labeled training data is used. This provides broad generalizability but also means:

- **Model-dependent performance**: detection accuracy is highest for text generated by models with similar distribution characteristics to PhoGPT-4B-Chat. Text from architecturally distinct models may yield less reliable scores.
- **Content sensitivity**: documents with high proportions of tables, formulas, code listings, or mixed-language content may produce atypical scores, even after the paragraph filtering step.

### 10.4 Ethical Disclaimer

VietAIDetector is a **research and decision-support tool**. Its output is probabilistic and must not be used as sole evidence in any consequential decision, including academic misconduct proceedings, employment decisions, or legal actions. Results must be reviewed by qualified human experts familiar with the limitations of AI-text detection technology. The authors assume no liability for decisions made on the basis of this software's output.

---

## 11. Reuse and Extensibility

### 11.1 Swapping Language Models

To use a different observer/performer pair, modify `config/settings.py`:

```python
OBSERVER_MODEL  = "your-org/your-base-model"
PERFORMER_MODEL = "your-org/your-chat-model"
```

Both models must be compatible with `AutoModelForCausalLM` and `AutoTokenizer`, share the same tokenizer vocabulary, and be available on HuggingFace Hub. No other code changes are required.

### 11.2 Adding a New Document Format

1. Extend `detect_format()` in `preprocessing/document_reader.py`:
   ```python
   if lower.endswith(".odt"):
       return "odt"
   ```
2. Add a static extraction method (e.g., `extract_from_odt(content: bytes) -> str`)
3. Add a branch in `read()`:
   ```python
   elif fmt == "odt":
       return self.extract_from_odt(content), "odt"
   ```
4. Update `gr.File(file_types=...)` in `frontend/gradio_app.py`

### 11.3 Adding a New Threshold Mode

1. Define the value in `config/settings.py`:
   ```python
   MY_THRESHOLD: float = 0.8200
   ```
2. Add to `THRESHOLD_MODES`:
   ```python
   THRESHOLD_MODES = {
       ...
       "My Custom Mode": MY_THRESHOLD,
   }
   ```
The new mode appears automatically in the UI dropdown. No other changes required.

### 11.4 Integrating into a Larger System

All pipeline components accept and return typed Python dataclasses, making them straightforward to serialize (e.g., via `dataclasses.asdict()`) for REST APIs or queue-based architectures. Each component is independently instantiable and stateless at the method level (state is limited to loaded model weights in `VietBinocularsScorer` and `VinternOCR`).

**Minimal integration pattern:**

```python
# Initialize once at startup
scorer     = VietBinocularsScorer(OBSERVER_MODEL, PERFORMER_MODEL)
normalizer = TextNormalizer()
chunker    = TextChunker(scorer.tokenizer)
aggregator = ScoreAggregator()

# Per-request processing
def analyze(filename: str, content: bytes) -> dict:
    prep   = normalizer.preprocess(filename, content)
    chunks = chunker.chunk(prep.cleaned_text, "run-001")
    scored = scorer.score_chunks(chunks.chunks)
    result = aggregator.aggregate(scored)
    return dataclasses.asdict(result)
```

---

## 12. Related Work and References

### Algorithm

[1] Nguyen, T. H., & Hoang, V.-D. (2026). *VietAIDetector: An Open-Source Zero-Shot Detector for Vietnamese AI-Generated Text*. arXiv:2608.25478 [cs.CL]. https://arxiv.org/abs/2608.25478

[2] Nguyen, T. H., & Akilesh, S. (2025). *VietBinoculars: A Zero-Shot Approach for Detecting Vietnamese LLM-Generated Text*. arXiv:2509.26189 [cs.CL]. https://arxiv.org/abs/2509.26189

[3] Hans, A., Schwarzschild, A., Cherepanova, V., Kazemi, H., Saha, A., Goldblum, M., Geiping, J., & Goldstein, T. (2024). *Spotting LLMs With Binoculars: Zero-Shot Detection of Machine-Generated Text*. arXiv:2401.12070 [cs.CL]. https://arxiv.org/abs/2401.12070

### Pre-trained Models

[4] VinAI Research. (2024). *PhoGPT: Generative Pre-training for Vietnamese*. HuggingFace: `vinai/PhoGPT-4B`, `vinai/PhoGPT-4B-Chat`.

[5] 5CD-AI. (2024). *Vintern-1B: An Efficient Multimodal Small Language Model for Vietnamese*. HuggingFace: `5CD-AI/Vintern-1B-v2`.


### Libraries

[6] Abid, A., et al. (2019). *Gradio: Hassle-Free Sharing and Testing of ML Models in the Wild*. https://gradio.app

[7] Wolf, T., et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP 2020 (System Demonstrations). https://github.com/huggingface/transformers

[8] Artifex Software. *PyMuPDF — Python Bindings for MuPDF*. https://pymupdf.readthedocs.io

[9] Léon Dica et al. *fpdf2 — PDF generation library for Python*. https://pyfpdf.github.io/fpdf2/

---

*VietAIDetector v1.1.1 — Developer Manual*  
*For bug reports and contributions, please open an issue or pull request on the project repository.*
