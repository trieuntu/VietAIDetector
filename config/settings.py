"""
Application Configuration.
Centralized constants, model paths, thresholds, and system parameters.
"""

import os
import torch

# Model Configuration

OBSERVER_MODEL: str = "vinai/PhoGPT-4B"
PERFORMER_MODEL: str = "vinai/PhoGPT-4B-Chat"
HF_TOKEN: str | None = os.environ.get("HF_TOKEN", None)

# Detection Thresholds (from VietBinoculars paper)
# Score < threshold → classified as AI-generated
# Score ≥ threshold → classified as Human-written

YOUDEN_THRESHOLD: float = 0.9279661178588868      # Optimal F1-score
CLOSEST_POINT_THRESHOLD: float = 0.925110161304474  # Near-perfect classification
FPR_THRESHOLD: float = 0.8992537260055542         # Low False Positive Rate

# Mapping from user-friendly mode names to threshold values
THRESHOLD_MODES: dict[str, float] = {
    "Youden (Balanced F1)": YOUDEN_THRESHOLD,
    "Closest Point (Near-Perfect)": CLOSEST_POINT_THRESHOLD,
    "Low FPR (Fewer False Alarms)": FPR_THRESHOLD,
}

DEFAULT_MODE: str = "Youden (Balanced F1)"

# Text Chunking Parameters

CHUNK_WINDOW: int = 450
CHUNK_OVERLAP: int = 100
CHUNK_STRIDE: int = CHUNK_WINDOW - CHUNK_OVERLAP
CHUNK_MIN_TOKENS: int = 50

# Device Configuration

DEVICE_1: str = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2: str = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

# Inference Configuration

USE_BFLOAT16: bool = True
MAX_MODEL_TOKENS: int = 768
SCORER_BATCH_SIZE: int = 8
MINIMUM_TOKENS: int = 64

# Report Configuration

FONT_PATH: str = os.environ.get("FONT_PATH", "/tmp/NotoSans-Regular.ttf")
FONT_URL: str = (
    "https://github.com/notofonts/noto-fonts/raw/main/hinted/ttf/"
    "NotoSans/NotoSans-Regular.ttf"
)

# OCR Configuration

OCR_MODEL: str = "5CD-AI/Vintern-1B-v2"
OCR_DEVICE: str = DEVICE_2
OCR_MAX_NEW_TOKENS: int = 2048
OCR_MAX_IMAGE_TILES: int = 12
OCR_PROMPT: str = (
    "Trích xuất chính xác toàn bộ văn bản có trong hình ảnh này. "
    "Yêu cầu bắt buộc: Giữ nguyên cấu trúc đoạn văn, dấu câu "
    "và các lỗi chính tả nếu có. Tuyệt đối không giải thích, "
    "không định dạng lại và không thêm bất kỳ từ ngữ nào ngoài nội dung gốc."
)
OCR_MIN_PAGE_CHARS: int = 10

# Application Metadata

APP_NAME: str = "VietAIDetector"
APP_VERSION: str = "1.1.1"
APP_DESCRIPTION: str = (
    "Vietnamese AI-generated text detection software, "
    "powered by the VietBinoculars algorithm with PhoGPT-4B model pair."
)
