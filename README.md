# VietAIDetector

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2509.26189-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.26189)
[![Platform](https://img.shields.io/badge/Platform-Kaggle%20T4%C3%972-f9ab00?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![Version](https://img.shields.io/badge/Version-1.1.0-6366f1)](docs/DEVELOPER_MANUAL.md)

**Vietnamese AI-generated text detection software** — a zero-shot, end-to-end system based on the *VietBinoculars* algorithm with the PhoGPT-4B model pair.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Detection Algorithm](#detection-algorithm)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Web UI](#web-ui)
  - [Programmatic API](#programmatic-api)
- [Deployment on Kaggle T4×2](#deployment-on-kaggle-t42)
- [Model Architecture](#model-architecture)
- [Detection Thresholds](#detection-thresholds)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Testing](#testing)
- [Citation](#citation)
- [Authors & Contact](#authors--contact)
- [License](#license)

---

## Overview

VietAIDetector is an open-source tool for detecting AI-generated Vietnamese text using a **zero-shot** scoring approach — no labeled training data or model fine-tuning is required. The system implements the *VietBinoculars* algorithm, which computes the ratio of *Perplexity* (PPL) to *Cross-Perplexity* (X-PPL) between two PhoGPT-4B language models: a base model (observer) and a chat-tuned model (performer). A low PPL/X-PPL ratio indicates that the text is likely AI-generated; a high ratio indicates human authorship.

The software accepts direct text input and file uploads (DOCX, native PDF, TXT). Scanned (image-based) PDFs are automatically detected and processed via an integrated OCR pipeline powered by the Vintern-1B-v2 Vision-Language Model. Long documents are split into overlapping chunks using a configurable sliding-window algorithm, and chunk-level scores are aggregated using a majority voting strategy to produce a document-level verdict. A downloadable PDF report with color-coded AI/Human highlights is generated upon completion. The system is optimized for dual-GPU deployment (Kaggle T4×2, 32 GB VRAM total).

---

## Key Features

**Input**
- Direct text input or file upload (`.docx`, `.pdf`, `.txt`)
- Automatic scanned PDF detection with OCR fallback (Vintern-1B-v2)

**Detection**
- Zero-shot VietBinoculars scoring — no training data required
- Three detection modes: Youden (balanced F1), Closest Point, Low FPR
- Configurable sliding-window chunking (window size and overlap adjustable from the UI)
- Chunk-level scores and AI/Human labels for full transparency

**Output**
- Summary verdict: *AI-generated* / *Human-written* / *Human-written but contains AI-generated parts*
- AI ratio progress bar with chunk statistics
- Downloadable PDF report with color-coded highlights (red = AI, green = Human)

**Deployment**
- Gradio web interface — no frontend coding required
- Optimized for Kaggle T4×2 (dual-GPU, 32 GB VRAM)
- Single-GPU fallback supported ($\ge 24$ GB VRAM)

---

## Detection Algorithm

VietAIDetector implements the **VietBinoculars** scoring algorithm ([Nguyen & Akilesh, 2025](https://arxiv.org/abs/2509.26189)), adapted from the Binoculars method ([Hans et al., 2024](https://arxiv.org/abs/2401.12070)):

**Step 1 — Per-chunk scoring:**

$$\text{score}(C_k) = \frac{\text{PPL}(\text{performer}, C_k)}{\text{X-PPL}(\text{observer} \to \text{performer}, C_k)}$$

Where:
- $\text{PPL}$ = cross-entropy loss of PhoGPT-4B-Chat on chunk $C_k$
- $\text{X-PPL}$ = cross-entropy from PhoGPT-4B probability distribution to PhoGPT-4B-Chat logits

A **low score** $\to$ AI-generated text | A **high score** $\to$ human-written text

**Step 2 — Majority Voting aggregation:**

$$P_{\text{AI}} = \frac{\text{number of chunks with } \text{score} < \text{threshold}}{\text{total chunks}} \times 100\%$$

**Step 3 — Document-level decision:**

| Condition | Verdict |
|---|---|
| $P_{\text{AI}} > 50\%$ | AI-generated |
| $0\% < P_{\text{AI}} \le 50\%$ | Human-written but contains AI-generated parts |
| $P_{\text{AI}} = 0\%$ | Human-written |

For the full mathematical derivation, see [DEVELOPER_MANUAL.md § 4 — Mathematical Foundations](docs/DEVELOPER_MANUAL.md#4-mathematical-foundations).

---

## System Requirements

| Requirement | Specification |
|-------------|--------------|
| Python | $\ge 3.10$ |
| CUDA | $\ge 11.8$ |
| GPU VRAM (recommended) | $2 \times$ NVIDIA T4 — 32 GB total |
| GPU VRAM (minimum) | $\ge 24$ GB (single GPU) |
| Disk space | $\approx 20$ GB (model weights) |
| Internet | Required on first run (HuggingFace model download) |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/trieuntu/VietAIDetector.git
cd VietAIDetector

# 2. Create and activate a virtual environment
python3.10 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set environment variables
export HF_TOKEN="hf_your_token_here"          # HuggingFace token for gated models
export FONT_PATH="/tmp/NotoSans-Regular.ttf"  # Auto-downloaded if not set
```

> **First run:** PhoGPT-4B (~9 GB) and PhoGPT-4B-Chat (~9 GB) are downloaded from HuggingFace Hub on first launch (~3–5 min). The OCR model (Vintern-1B-v2, ~2.5 GB) is only downloaded when a scanned PDF is first submitted.

---

## Quick Start

### Web UI

```bash
python app.py
```

Open `http://localhost:7860` in a browser. Paste Vietnamese text (or upload a file), select a detection mode, and click **Analyze**.

### Programmatic API

```python
from preprocessing.normalizer import TextNormalizer
from processing.chunker import TextChunker
from processing.aggregator import ScoreAggregator
from core.scorer import VietBinocularsScorer
from config.settings import OBSERVER_MODEL, PERFORMER_MODEL, YOUDEN_THRESHOLD
import uuid

# Initialize once at startup (model loading happens here)
scorer     = VietBinocularsScorer(OBSERVER_MODEL, PERFORMER_MODEL)
normalizer = TextNormalizer()
chunker    = TextChunker(tokenizer_name_or_obj=scorer.tokenizer)  # reuse tokenizer (can set window=600, overlap=200 globally here)
aggregator = ScoreAggregator()

# Run the detection pipeline
text = "Trí tuệ nhân tạo đang ngày càng phát triển mạnh mẽ và được ứng dụng rộng rãi..."
prep   = normalizer.preprocess("input.txt", text)
# You can override chunking parameters per document here:
chunks = chunker.chunk(prep.cleaned_text, document_id=str(uuid.uuid4())[:8], window=600, overlap=200)
scored = scorer.score_chunks(chunks.chunks)
result = aggregator.aggregate(scored, threshold=YOUDEN_THRESHOLD)

print(f"Decision  : {result.final_decision}")
print(f"AI ratio  : {result.ai_percentage:.1f}%")
print(f"Chunks    : {result.total_chunks} total / {result.ai_chunk_count} AI")
```

For the full API reference, see [DEVELOPER_MANUAL.md § 6 — Usage Guide](docs/DEVELOPER_MANUAL.md#6-usage-guide).

---

## Deployment on Kaggle T4×2

1. Create a new Kaggle Notebook
2. **Settings → Accelerator**: select `GPU T4 x2`
3. **Settings → Internet**: enable
4. Upload the `VietAIDetector/` folder to `/kaggle/working/`
5. In the first code cell, run:

```bash
!bash /kaggle/working/VietAIDetector/run_kaggle.sh
```

The script installs dependencies, downloads the NotoSans font, verifies GPU availability, and launches the Gradio app. A public Gradio URL is printed to the console upon startup.

---

## Model Architecture

| Role | Model | Device | VRAM (bf16) |
|------|-------|--------|-------------|
| Observer (base) | `vinai/PhoGPT-4B` | `cuda:0` | ~9 GB |
| Performer (chat-tuned) | `vinai/PhoGPT-4B-Chat` | `cuda:1` | ~9 GB |
| OCR (scanned PDF) | `5CD-AI/Vintern-1B-v2` | `cuda:1` (lazy) | ~2.5 GB |

> **Lazy loading:** The OCR model is only loaded when a scanned PDF is detected, keeping startup fast and VRAM usage minimal. On Kaggle T4×2: `cuda:0` uses ~9/15 GB; `cuda:1` uses ~9/15 GB (up to ~11.5/15 GB with OCR active).

For a full architecture diagram and data-flow description, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Detection Thresholds

Detection thresholds and mode mappings are defined in [`config/settings.py`](config/settings.py):

- **Classification rule**:
  - $\text{Score} < \text{threshold} \implies \textbf{AI-generated}$
  - $\text{Score} \ge \text{threshold} \implies \textbf{Human-written}$

| Mode (`THRESHOLD_MODES`) | Constant | Value | Description |
|---|---|---|---|
| **Youden (Balanced F1)** *(Default)* | `YOUDEN_THRESHOLD` | $\approx 0.9280$ | Maximizes Youden's J statistic (optimal balanced F1-score) |
| **Closest Point (Near-Perfect)** | `CLOSEST_POINT_THRESHOLD` | $\approx 0.9251$ | Closest-to-$(0,1)$ point on ROC curve |
| **Low FPR (Fewer False Alarms)** | `FPR_THRESHOLD` | $\approx 0.8993$ | Low False Positive Rate mode ($\text{TPR@0.05FPR}$, minimizes false alarms) |

---

## Project Structure

VietAIDetector follows a modular architecture organized by separation of concerns:

- **`config/`** — Centralized settings, model identifiers, threshold parameters, and device configurations.
- **`schemas/`** — Core data models (`ChunkDetail`, `DetectionResult`, `PreprocessResult`).
- **`core/`** — Dual-model inference scoring (`VietBinocularsScorer`) and metric calculations.
- **`preprocessing/`** — Text normalization, document extraction (DOCX, native/scanned PDF), and OCR engine (`VinternOCR`).
- **`processing/`** — Sliding-window token chunking (`TextChunker`) and score aggregation (`ScoreAggregator`).
- **`reporting/`** — Color-coded PDF report generation (`PDFReportGenerator`).
- **`frontend/`** — Interactive web interface (`gradio_app.py`).
- **`benchmarks/`** — Evaluation scripts and benchmark corpora.
- **`tests/`** — Comprehensive pytest test suite.

For the full directory tree, module descriptions, and data-flow pipeline, see [STRUCTURE.md](docs/STRUCTURE.md).

---

## Documentation

| Document | Description |
|----------|-------------|
| [DEVELOPER_MANUAL.md](docs/DEVELOPER_MANUAL.md) | Full developer reference: API docs, mathematical foundations, programmatic usage, extensibility guide |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Five-layer architecture overview with Mermaid data-flow diagram |
| [STRUCTURE.md](docs/STRUCTURE.md) | Detailed directory structure and end-to-end data-flow pipeline |

---

## Testing

Unit tests cover all non-GPU-dependent components (preprocessing, chunking, aggregation, PDF generation, OCR preprocessing). GPU-dependent inference is excluded to support CI environments without hardware requirements.

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v
```

Test configuration is defined in `pyproject.toml`. See [DEVELOPER_MANUAL.md § 7 — Testing](docs/DEVELOPER_MANUAL.md#7-testing) for details on test design and coverage.

---

## Citation

If you use VietAIDetector in your research, please cite both the algorithm paper and the software:

**Algorithm paper (VietBinoculars):**

```bibtex
@misc{nguyen2025vietbinoculars,
  title     = {VietBinoculars: A Zero-Shot Approach for Detecting Vietnamese LLM-Generated Text},
  author    = {Trieu Hai Nguyen and Sivaswamy Akilesh},
  year      = {2025},
  eprint    = {2509.26189},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url       = {https://arxiv.org/abs/2509.26189}
}
```

**Software (VietAIDetector):**

```bibtex
@software{nguyen2025vietaidetector,
  author    = {Trieu Hai Nguyen and Sivaswamy Akilesh},
  title     = {{ VietAIDetector }: Vietnamese AI-Generated Text Detection Software},
  year      = {2025},
  version   = {1.1.0},
  url       = {https://github.com/trieuntu/VietAIDetector},
  license   = {MIT}
}
```

## License

This software is released under the [MIT License](LICENSE).

> **Note on model licenses:** The pre-trained models used by this software are subject to their own licenses:
> - `vinai/PhoGPT-4B` and `vinai/PhoGPT-4B-Chat`: see [VinAI Research terms](https://huggingface.co/vinai/PhoGPT-4B)
> - `5CD-AI/Vintern-1B-v2`: see [5CD-AI model card](https://huggingface.co/5CD-AI/Vintern-1B-v2)
>
> Users are responsible for complying with the applicable model license terms.

